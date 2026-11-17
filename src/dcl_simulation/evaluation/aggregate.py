from dataclasses import asdict, dataclass
import hashlib
import math
from statistics import median
from typing import Sequence

import numpy as np

from .input import EvaluationEpisode, EvaluationInputError
from .interaction import EpisodeEvaluation, evaluate_interaction


@dataclass(frozen=True)
class BootstrapConfig:
    samples: int = 2000
    seed: int = 4242

    def __post_init__(self):
        if type(self.samples) is not int or self.samples != 2000:
            raise ValueError('bootstrap.samples must be 2000')
        if type(self.seed) is not int:
            raise ValueError('bootstrap.seed must be an integer')


@dataclass(frozen=True)
class AggregateMetric:
    condition_id: str
    model_id: str
    bin_index: int
    bin_start_s: float
    bin_end_s: float
    metric: str
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    status: str
    reason: str | None
    ci_status: str
    event_count: int
    episode_count: int
    eligible_episode_count: int
    bootstrap_valid_samples: int
    numerator: float | None = None
    denominator: float | None = None


@dataclass(frozen=True)
class ModelCoverage:
    model_id: str
    planned_episode_count: int
    completed_episode_count: int
    excluded_episode_count: int
    excluded_episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatasetEvaluation:
    condition_id: str
    bootstrap: BootstrapConfig
    minimum_fto_events: int
    comparison_conditions: dict[str, object]
    coverage: tuple[ModelCoverage, ...]
    episodes: tuple[EpisodeEvaluation, ...]
    metrics: tuple[AggregateMetric, ...]


def _require(condition, message):
    if not condition:
        raise EvaluationInputError(message)


def _conditions(episodes):
    _require(bool(episodes), 'At least one episode is required for comparison')
    keys, directories = set(), set()
    plans, models, audio_specs = {}, {}, {}
    common_policy, common_services = None, None
    for episode in episodes:
        key = (episode.model_id, episode.episode_id)
        directory = episode.directory.resolve()
        _require(key not in keys and directory not in directories, f'Duplicate evaluation input: {key}')
        keys.add(key)
        directories.add(directory)
        policy = asdict(episode.config)
        scenario = {'persona': policy.pop('persona'), 'topic': policy.pop('topic')}
        plans.setdefault(episode.model_id, {})[episode.episode_id] = scenario
        if common_policy is None:
            common_policy = policy
        _require(policy == common_policy, f'Conversation policy settings differ: {key}')
        if episode.summary.status == 'failed':
            continue
        provenance = episode.provenance
        _require(provenance is not None, f'Missing provenance for comparison: {key}')
        for role in ('model', 'llm', 'tts', 'vad', 'asr'):
            info = getattr(provenance, role)
            _require(info is not None and info.revision is not None,
                     f'Missing {role} name or revision for comparison: {key}')
        _require(provenance.vad_timestamp_resolution_s is not None, f'Missing VAD timestamp resolution: {key}')
        data = asdict(provenance)
        model = data.pop('model')
        previous = models.setdefault(episode.model_id, model)
        _require(previous == model, f'Versions or settings differ for the same model_id: {episode.model_id}')
        previous_spec = audio_specs.setdefault(episode.model_id, asdict(episode.audio_spec))
        _require(previous_spec == asdict(episode.audio_spec), f'Audio interfaces differ for the same model_id: {episode.model_id}')
        if common_services is None:
            common_services = data
        _require(data == common_services, f'LLM/TTS/VAD/ASR comparison conditions differ: {key}')
    reference = next(iter(plans.values()))
    for model_id, plan in plans.items():
        _require(set(plan) == set(reference), f'Planned episode sets differ across models: {model_id}')
        _require(plan == reference, f'Persona or topic differs for the same episode_id: {model_id}')
    return {'policy': common_policy, 'scenarios': dict(sorted(reference.items())),
            'services': common_services,
            'models': {name: models.get(name) for name in sorted(plans)},
            'audio_specs': {name: audio_specs.get(name) for name in sorted(plans)},
            'provenance_scope': 'completed_episodes'}


def _interval(clusters, *, config, domain, statistic):
    digest = hashlib.sha256(f'dcl:evaluation:bootstrap:{config.seed}:{domain}'.encode('utf-8')).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    values = []
    for _ in range(config.samples):
        indices = rng.integers(0, len(clusters), size=len(clusters))
        value = statistic([clusters[int(index)] for index in indices])
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if not values:
        return None, None, 0
    lower, upper = np.percentile(np.asarray(values, dtype=np.float64), (2.5, 97.5))
    return float(lower), float(upper), len(values)


def _mean(values):
    return sum(values) / len(values) if values else None


def _performance_end(episode):
    end = episode.config.target_duration_s
    if episode.summary.termination_reason == 'response_collapse':
        end = min(end, episode.summary.collapse_decision_time_s)
    return end


def evaluate_dataset(
    episodes: Sequence[EvaluationEpisode], *, condition_id: str = 'long_session',
    bootstrap: BootstrapConfig = BootstrapConfig(), minimum_fto_events: int = 20,
) -> DatasetEvaluation:
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise ValueError('condition_id must be nonempty')
    if type(minimum_fto_events) is not int or minimum_fto_events <= 0:
        raise ValueError('minimum_fto_events must be a positive integer')
    if not isinstance(bootstrap, BootstrapConfig):
        raise TypeError('bootstrap must be BootstrapConfig')
    ordered = sorted(episodes, key=lambda e: (e.model_id, e.episode_id))
    conditions = _conditions(ordered)
    analyses = tuple(evaluate_interaction(episode) for episode in ordered)
    groups = {}
    for episode, analysis in zip(ordered, analyses):
        groups.setdefault(episode.model_id, []).append((episode, analysis))
    metrics, coverage = [], []
    for model_id, group in sorted(groups.items()):
        completed = [pair for pair in group if pair[1].status == 'ok']
        excluded = tuple(e.episode_id for e, a in group if a.status == 'excluded')
        coverage.append(ModelCoverage(model_id, len(group), len(completed), len(excluded), excluded))
        count = math.ceil(group[0][0].config.target_duration_s / 60)
        for index in range(count):
            observed = [pair for pair in completed if _performance_end(pair[0]) > index * 60 + 1e-9]
            complete = [pair for pair in observed if _performance_end(pair[0]) + 1e-9 >= (index + 1) * 60]

            def row(name, estimate, clusters, statistic, contributors, event_count=0,
                    numerator=None, denominator=None, reason=None):
                status = 'excluded' if not completed else ('ok' if estimate is not None else 'not_observed')
                lower, upper, valid = None, None, 0
                ci_status = status
                if status == 'ok':
                    if len(clusters) < 2:
                        ci_status = 'insufficient_episodes'
                    else:
                        lower, upper, valid = _interval(
                            clusters, config=bootstrap, domain=f'{condition_id}:{model_id}:{index}:{name}',
                            statistic=statistic,
                        )
                        ci_status = 'ok' if valid else 'not_observed'
                metrics.append(AggregateMetric(
                    condition_id, model_id, index, index * 60.0, (index + 1) * 60.0, name,
                    estimate, lower, upper, status,
                    'all_episodes_failed' if not completed else (reason if estimate is None else None),
                    ci_status, event_count, contributors, len(clusters), valid, numerator, denominator,
                ))

            for name in ('user_speech_duration_s', 'model_speech_duration_s'):
                def duration(selected, field=name):
                    return _mean([getattr(a.observations[index], field) for _, a in selected])
                row(name, duration(complete), complete, duration, len(complete))
            def fto_values(selected):
                return [event.fto_ms for _, analysis in selected for event in analysis.fto_events
                        if event.bin_index == index]
            def fto_statistic(selected):
                values = fto_values(selected)
                return median(values) if values else None
            values = fto_values(observed)
            row('fto_median_ms', median(values) if len(values) >= minimum_fto_events else None,
                observed, fto_statistic,
                sum(any(event.bin_index == index for event in a.fto_events) for _, a in observed),
                event_count=len(values), reason=f'fewer_than_{minimum_fto_events}_fto_events')
            def counts(selected):
                return (sum(a.observations[index].no_response_count for _, a in selected),
                        sum(a.observations[index].response_opportunity_count for _, a in selected))
            def rate(selected):
                numerator, denominator = counts(selected)
                return numerator / denominator if denominator else None
            numerator, denominator = counts(completed)
            row('no_response_rate', rate(completed), completed, rate,
                sum(a.observations[index].response_opportunity_count > 0 for _, a in completed),
                event_count=numerator, numerator=float(numerator), denominator=float(denominator),
                reason='no_resolved_opportunity')
            def collapse(selected):
                return _mean([float(a.observations[index].cumulative_collapse) for _, a in selected])
            row('cumulative_collapse', collapse(completed), completed, collapse, len(completed),
                event_count=sum(a.observations[index].cumulative_collapse for _, a in completed))
    return DatasetEvaluation(condition_id, bootstrap, minimum_fto_events, conditions,
                             tuple(coverage), analyses, tuple(metrics))
