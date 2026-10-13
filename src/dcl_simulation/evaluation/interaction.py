from dataclasses import dataclass
import math
from statistics import median

from .input import EvaluationEpisode


@dataclass(frozen=True)
class FTOEvent:
    bin_index: int
    opportunity_id: int
    user_utterance_id: int
    model_segment_id: int
    user_end_s: float
    model_start_s: float
    fto_ms: float


@dataclass(frozen=True)
class ModelBehavior:
    bin_index: int
    turn_id: int
    segment_ids: tuple[int, ...]
    user_utterance_id: int
    behavior: str
    start_s: float
    end_s: float
    overlap_duration_s: float


@dataclass(frozen=True)
class BinObservation:
    bin_index: int
    user_speech_duration_s: float
    model_speech_duration_s: float
    response_opportunity_count: int
    no_response_count: int
    unresolved_opportunity_count: int
    cumulative_collapse: int


@dataclass(frozen=True)
class Metric:
    bin_index: int
    bin_start_s: float
    bin_end_s: float
    metric: str
    value: float | None
    status: str
    reason: str | None = None
    numerator: float | None = None
    denominator: float | None = None
    event_count: int = 0


@dataclass(frozen=True)
class EpisodeEvaluation:
    model_id: str
    episode_id: str
    status: str
    reason: str | None
    fto_events: tuple[FTOEvent, ...]
    model_behaviors: tuple[ModelBehavior, ...]
    observations: tuple[BinObservation, ...]
    metrics: tuple[Metric, ...]
    unassigned_opportunity_count: int


def _bin(time_s, target):
    return int(time_s // 60) if 0 <= time_s and time_s + 1e-9 < target else None


def _overlap(start, end, left, right):
    return max(0.0, min(end, right) - max(start, left))


def _durations(spans, count):
    merged = []
    for start, end in sorted((start, end) for start, end in spans if end > start):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return [sum(_overlap(start, end, i * 60, (i + 1) * 60) for start, end in merged)
            for i in range(count)]


def evaluate_interaction(episode: EvaluationEpisode) -> EpisodeEvaluation:
    if episode.summary.status == 'failed':
        return EpisodeEvaluation(episode.model_id, episode.episode_id, 'excluded', 'simulation_failed',
                                 (), (), (), (), 0)
    target = episode.config.target_duration_s
    count = math.ceil(target / 60)
    users = {u.id: u for u in episode.user_utterances}
    segments = {s.id: s for s in episode.speech_segments}
    fto, behaviors = [], []
    for opportunity in episode.response_opportunities:
        if opportunity.outcome != 'response' or opportunity.response_classification != 'normal':
            continue
        user = users[opportunity.user_utterance_id]
        segment = segments[opportunity.response_segment_id]
        index = _bin(user.end_s, target)
        if index is not None:
            fto.append(FTOEvent(index, opportunity.id, user.id, segment.id,
                                user.end_s, segment.start_s, (segment.start_s - user.end_s) * 1000))
    for turn in episode.model_turns:
        if turn.kind not in ('backchannel', 'interrupting'):
            continue
        grouped = [segments[sid] for sid in turn.segment_ids]
        index = _bin(grouped[0].start_s, target)
        if index is None:
            continue
        user = users[turn.user_utterance_id]
        behaviors.append(ModelBehavior(
            index, turn.id, turn.segment_ids, user.id,
            'backchannel' if turn.kind == 'backchannel' else 'interrupt',
            grouped[0].start_s, grouped[-1].end_s,
            sum(_overlap(s.start_s, s.end_s, user.start_s, user.end_s) for s in grouped),
        ))
    user_durations = _durations(((u.start_s, u.end_s) for u in users.values() if u.delivered_sample_count), count)
    model_durations = _durations(((s.start_s, s.end_s) for s in segments.values()), count)
    opportunities, misses, unresolved = [0] * count, [0] * count, [0] * count
    unassigned = 0
    for opportunity in episode.response_opportunities:
        if opportunity.start_s is None:
            unassigned += 1
            continue
        index = _bin(opportunity.start_s, target)
        if index is None:
            continue
        if opportunity.outcome in ('response', 'no_response'):
            opportunities[index] += 1
            misses[index] += int(opportunity.outcome == 'no_response')
        else:
            unresolved[index] += 1
    onset = episode.summary.collapse_onset_time_s
    collapse_bin = _bin(onset, target) if onset is not None else None
    observations = tuple(BinObservation(i, user_durations[i], model_durations[i], opportunities[i],
                                       misses[i], unresolved[i], int(collapse_bin is not None and i >= collapse_bin))
                         for i in range(count))
    end = target
    if episode.summary.termination_reason == 'response_collapse':
        end = min(target, episode.summary.collapse_decision_time_s)
    metrics = []
    for observation in observations:
        index = observation.bin_index
        partial = end > index * 60 + 1e-9
        complete = end + 1e-9 >= (index + 1) * 60
        def row(name, value, status, **kwargs):
            metrics.append(Metric(index, index * 60.0, (index + 1) * 60.0, name, value, status, **kwargs))
        for name, value in (('user_speech_duration_s', observation.user_speech_duration_s),
                            ('model_speech_duration_s', observation.model_speech_duration_s)):
            row(name, value if complete else None, 'ok' if complete else 'not_observed',
                reason=None if complete else 'minute_not_fully_observed')
        values = [event.fto_ms for event in fto if event.bin_index == index]
        row('fto_median_ms', median(values) if partial and values else None,
            'ok' if partial and values else 'not_observed', event_count=len(values),
            reason=None if partial and values else ('no_normal_response' if partial else 'minute_not_observed'))
        denominator = observation.response_opportunity_count
        row('no_response_rate', observation.no_response_count / denominator if denominator else None,
            'ok' if denominator else 'not_observed', reason=None if denominator else 'no_resolved_opportunity',
            numerator=float(observation.no_response_count), denominator=float(denominator),
            event_count=observation.no_response_count)
        row('cumulative_collapse', float(observation.cumulative_collapse), 'ok',
            event_count=observation.cumulative_collapse)
    return EpisodeEvaluation(episode.model_id, episode.episode_id, 'ok', None, tuple(fto), tuple(behaviors),
                             observations, tuple(metrics), unassigned)
