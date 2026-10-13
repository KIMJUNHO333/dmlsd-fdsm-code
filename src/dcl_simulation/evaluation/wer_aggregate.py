from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .aggregate import BootstrapConfig, _conditions, _interval
from .input import EvaluationEpisode, EvaluationInputError
from .wer import EpisodeWER, WERConfig, WERDependencyError, evaluate_wer


@dataclass(frozen=True)
class WERCoverage:
    model_id: str
    episode_count: int
    status_counts: dict[str, int]
    unscored_episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class WERAggregateMetric:
    condition_id: str
    model_id: str
    bin_index: int
    metric: str
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    status: str
    ci_status: str
    episode_count: int
    event_count: int
    bootstrap_valid_samples: int
    word_count: int
    speech_duration_s: float
    status_counts: dict[str, int]
    excluded_episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class WERDatasetEvaluation:
    condition_id: str
    bootstrap: BootstrapConfig
    minimum_episodes: int
    profiles: dict[str, str]
    comparison_conditions: dict[str, object]
    coverage: tuple[WERCoverage, ...]
    episodes: tuple[EpisodeWER, ...]
    metrics: tuple[WERAggregateMetric, ...]


def evaluate_wer_dataset(
    episodes: Sequence[EvaluationEpisode], *, condition_id: str = 'long_session',
    bootstrap: BootstrapConfig = BootstrapConfig(), minimum_episodes: int = 10,
    profiles: Mapping[str, str] | None = None,
) -> WERDatasetEvaluation:
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise ValueError('condition_id must be nonempty')
    if type(minimum_episodes) is not int or minimum_episodes <= 0:
        raise ValueError('minimum_episodes must be a positive integer')
    if not isinstance(bootstrap, BootstrapConfig):
        raise TypeError('bootstrap must be BootstrapConfig')
    ordered = sorted(episodes, key=lambda e: (e.model_id, e.episode_id))
    conditions = _conditions(ordered)
    models = {e.model_id for e in ordered}
    if profiles is not None and not isinstance(profiles, Mapping):
        raise ValueError('profiles must map model IDs to WER profiles')
    profiles = dict(profiles or {})
    if set(profiles) - models:
        raise EvaluationInputError(f'WER profile refers to a model absent from the input: {set(profiles) - models}')
    configs = {model: WERConfig(profiles.get(model, 'standard')) for model in sorted(models)}
    results, grouped = [], {}
    for episode in ordered:
        config = configs[episode.model_id]
        try:
            result = evaluate_wer(episode, config=config)
        except WERDependencyError:
            raise
        except (ValueError, RuntimeError) as exc:
            result = EpisodeWER(episode.model_id, episode.episode_id, config, 'evaluator_error',
                                f'{type(exc).__name__}: {exc}', {}, (), (), None, None, None, None, None)
        results.append(result)
        grouped.setdefault(episode.model_id, []).append(result)
    coverage, metrics = [], []
    bin_count = math.ceil(ordered[0].config.target_duration_s / 60)
    for model, group in sorted(grouped.items()):
        counts = {}
        for result in group:
            counts[result.status] = counts.get(result.status, 0) + 1
        coverage.append(WERCoverage(model, len(group), counts,
                                    tuple(r.episode_id for r in group if r.status != 'ok')))
        for index in range(bin_count):
            valid, statuses, excluded = [], {}, []
            for result in group:
                status = result.status
                if status == 'ok':
                    minute = result.minutes[index]
                    status = minute.status
                    if status == 'ok' and minute.wer is not None:
                        valid.append(minute)
                if status != 'ok':
                    excluded.append(result.episode_id)
                statuses[status] = statuses.get(status, 0) + 1
            sufficient = len(valid) >= minimum_episodes
            def statistic(selected):
                return float(np.mean([m.wer for m in selected])) if selected else None
            estimate = statistic(valid) if sufficient else None
            lower, upper, samples = None, None, 0
            if sufficient:
                status = 'ok'
            elif valid:
                status = 'insufficient_episodes'
            else:
                status = next(iter(statuses)) if len(statuses) == 1 else 'insufficient'
            ci_status = status
            if sufficient:
                if len(valid) < 2:
                    ci_status = 'insufficient_episodes'
                else:
                    lower, upper, samples = _interval(valid, config=bootstrap,
                        domain=f'quality:{condition_id}:{model}:{index}:wer', statistic=statistic)
                    ci_status = 'ok' if samples else 'not_observed'
            metrics.append(WERAggregateMetric(
                condition_id, model, index, 'wer', estimate, lower, upper, status, ci_status,
                len(valid), sum(m.sentence_count for m in valid), samples,
                sum(m.reference_words for m in valid), sum(m.model_speech_duration_s for m in valid),
                statuses, tuple(excluded),
            ))
    return WERDatasetEvaluation(condition_id, bootstrap, minimum_episodes,
                                {m: c.profile for m, c in configs.items()}, conditions,
                                tuple(coverage), tuple(results), tuple(metrics))
