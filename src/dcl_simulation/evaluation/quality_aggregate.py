from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..records import ComponentInfo
from .aggregate import BootstrapConfig, _conditions, _interval
from .input import EvaluationEpisode, EvaluationInputError
from .judge import ConversationJudge, EpisodeJudge, evaluate_judge
from .speaker import SpeakerEmbedder, EpisodeSpeakerIdentity, evaluate_speaker


@dataclass(frozen=True)
class QualityCoverage:
    model_id: str
    episode_count: int
    status_counts: dict[str, int]
    unscored_episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class QualityMetric:
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
    speech_duration_s: float
    status_counts: dict[str, int]
    excluded_episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class QualityDatasetEvaluation:
    metric: str
    condition_id: str
    bootstrap: BootstrapConfig
    minimum_episodes: int
    evaluator: ComponentInfo
    comparison_conditions: dict[str, object]
    coverage: tuple[QualityCoverage, ...]
    episodes: tuple[EpisodeSpeakerIdentity | EpisodeJudge, ...]
    metrics: tuple[QualityMetric, ...]


def evaluate_speaker_dataset(episodes: Sequence[EvaluationEpisode], embedder: SpeakerEmbedder, *,
        evaluator: ComponentInfo, condition_id: str = 'long_session',
        bootstrap: BootstrapConfig = BootstrapConfig(), minimum_episodes: int = 10) -> QualityDatasetEvaluation:
    return _evaluate(episodes, embedder, evaluator=evaluator, metric='speaker_similarity',
                     condition_id=condition_id, bootstrap=bootstrap, minimum_episodes=minimum_episodes)


def evaluate_judge_dataset(episodes: Sequence[EvaluationEpisode], judge: ConversationJudge, *,
        evaluator: ComponentInfo, condition_id: str = 'long_session',
        bootstrap: BootstrapConfig = BootstrapConfig(), minimum_episodes: int = 10) -> QualityDatasetEvaluation:
    return _evaluate(episodes, judge, evaluator=evaluator, metric='conversation_quality',
                     condition_id=condition_id, bootstrap=bootstrap, minimum_episodes=minimum_episodes)


def _evaluate(episodes, service, *, evaluator, metric, condition_id, bootstrap, minimum_episodes):
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise ValueError('condition_id must be nonempty')
    if type(minimum_episodes) is not int or minimum_episodes < 1:
        raise ValueError('minimum_episodes must be a positive integer')
    if not isinstance(bootstrap, BootstrapConfig):
        raise TypeError('bootstrap must be BootstrapConfig')
    if not isinstance(evaluator, ComponentInfo) or evaluator.revision is None:
        raise EvaluationInputError('Evaluator name and revision are required for comparison')
    ordered = sorted(episodes, key=lambda e: (e.model_id, e.episode_id))
    conditions = _conditions(ordered)
    results = []
    for episode in ordered:
        if metric == 'conversation_quality':
            result = evaluate_judge(episode, service, evaluator=evaluator)
        else:
            result = evaluate_speaker(episode, service, evaluator=evaluator)
        results.append(result)
    metrics, coverage = [], []
    first = 1 if metric == 'speaker_similarity' else 0
    count = math.ceil(ordered[0].config.target_duration_s / 60)
    for model in sorted({e.model_id for e in ordered}):
        group = [result for result in results if result.model_id == model]
        counts = {}
        for result in group:
            counts[result.status] = counts.get(result.status, 0) + 1
        coverage.append(QualityCoverage(model, len(group), counts,
                                        tuple(r.episode_id for r in group if r.status != 'ok')))
        for index in range(first, count):
            valid, statuses, excluded, target_statuses = [], {}, [], []
            for result in group:
                minute = next((m for m in result.minutes if m.bin_index == index), None)
                status = minute.status if minute is not None else result.status
                if minute is not None:
                    target_statuses.append(minute.status)
                if status == 'ok' and minute is not None:
                    value = minute.similarity if metric == 'speaker_similarity' else minute.score
                    if value is not None:
                        valid.append(minute)
                else:
                    excluded.append(result.episode_id)
                statuses[status] = statuses.get(status, 0) + 1
            sufficient = len(valid) >= minimum_episodes
            status = ('ok' if sufficient else 'insufficient_episodes' if valid else
                      next(iter(statuses)) if len(statuses) == 1 else 'insufficient')
            def statistic(rows):
                return float(np.mean([m.similarity if metric == 'speaker_similarity' else m.score
                                      for m in rows])) if rows else None
            lower, upper, samples = None, None, 0
            ci_status = status
            if sufficient and len(valid) >= 2:
                lower, upper, samples = _interval(valid, config=bootstrap,
                    domain=f'quality:{condition_id}:{model}:{index}:{metric}', statistic=statistic)
                ci_status = 'ok' if samples else 'not_observed'
            elif sufficient:
                ci_status = 'insufficient_episodes'
            metrics.append(QualityMetric(condition_id, model, index, metric,
                statistic(valid) if sufficient else None, lower, upper, status, ci_status,
                len(valid), sum(m.segment_count for m in valid) if metric == 'speaker_similarity' else len(valid),
                samples, sum(m.valid_speech_duration_s for m in valid) if metric == 'speaker_similarity' else 0.0,
                statuses, tuple(excluded)))
            if metric == 'conversation_quality':
                for state in ('ok', 'unscorable', 'evaluator_error'):
                    label = 'scorable' if state == 'ok' else state
                    total = target_statuses.count(state)
                    metrics.append(QualityMetric(condition_id, model, index, f'judge_segment_{label}_count',
                        float(total), None, None, 'ok', 'not_applicable', len(target_statuses), total, 0, 0.0, {}, ()))
    return QualityDatasetEvaluation(metric, condition_id, bootstrap, minimum_episodes, evaluator,
                                    conditions, tuple(coverage), tuple(results), tuple(metrics))
