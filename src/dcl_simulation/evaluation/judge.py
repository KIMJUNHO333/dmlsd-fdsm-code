from dataclasses import dataclass
import math
from typing import Protocol

from ..records import ComponentInfo
from .aggregate import _performance_end
from .input import EvaluationEpisode, _json
from ._judge_text import system_prompt, response_schema, _segment_rows, _is_model_row


@dataclass(frozen=True)
class JudgeTask:
    model_id: str
    episode_id: str
    system_prompt: str
    user_prompt: str
    segment_statuses: tuple[str, ...]

    @property
    def response_schema(self) -> dict[str, object]:
        return response_schema(len(self.segment_statuses))


class ConversationJudge(Protocol):
    def complete(self, task: JudgeTask) -> str:
        ...


@dataclass(frozen=True)
class JudgeMinute:
    bin_index: int
    status: str
    reason: str | None
    rationale: str | None
    score: int | None


@dataclass(frozen=True)
class EpisodeJudge:
    model_id: str
    episode_id: str
    evaluator: ComponentInfo
    status: str
    reason: str | None
    task: JudgeTask | None
    raw_response: str | None
    minutes: tuple[JudgeMinute, ...]


def build_judge_task(episode: EvaluationEpisode) -> JudgeTask:
    if episode.summary.status == 'failed':
        raise ValueError('Cannot build a judge request for a failed episode')
    count = math.ceil(episode.config.target_duration_s / 60)
    observation_end = _performance_end(episode)
    segments, statuses = ['<dialogue>'], []
    for index in range(count):
        start, end = index * 60.0, (index + 1) * 60.0
        observed = max(0.0, min(end, observation_end) - start)
        rows = _segment_rows(episode, start, end)
        if observed <= 0:
            status = 'unobserved'
        elif observed + 1e-9 < 30.0:
            status = 'insufficient_observation'
        elif not any(_is_model_row(row) for row in rows):
            status = 'model_speech_missing'
        else:
            status = 'scorable'
        statuses.append(status)
        if rows:
            segments.extend([f'<segment id="{index}" status="{status}">', *rows, '</segment>'])
        else:
            segments.append(f'<segment id="{index}" status="{status}" />')
        if index != count - 1:
            segments.append('')
    segments.append('</dialogue>')
    return JudgeTask(episode.model_id, episode.episode_id, system_prompt(count),
                     '\n'.join(segments), tuple(statuses))


def parse_judgment(task: JudgeTask, content: str) -> tuple[JudgeMinute, ...]:
    if not isinstance(content, str):
        raise ValueError('Judge response must be a JSON string')
    value = _json(content)
    if not isinstance(value, dict) or set(value) != {'segments'}:
        raise ValueError('Judge response root does not match the schema')
    rows = value['segments']
    if not isinstance(rows, list) or len(rows) != len(task.segment_statuses):
        raise ValueError('Judge response segment count does not match the request')
    results = []
    for index, (status, row) in enumerate(zip(task.segment_statuses, rows)):
        if not isinstance(row, dict) or set(row) != {'segment_id', 'rationale', 'score'}:
            raise ValueError(f'Judge segment fields do not match: {index}')
        if type(row['segment_id']) is not int or row['segment_id'] != index:
            raise ValueError(f'Invalid judge segment order or ID type: {index}')
        if status != 'scorable':
            if row['rationale'] is not None or row['score'] is not None:
                raise ValueError(f'Unscorable segments must use null values: {index}')
            results.append(JudgeMinute(index, 'unscorable', status, None, None))
            continue
        rationale, score = row['rationale'], row['score']
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f'Empty judge rationale: {index}')
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError(f'Judge score must be an integer from 1 to 5: {index}')
        results.append(JudgeMinute(index, 'ok', None, rationale.strip(), score))
    return tuple(results)


def evaluate_judge(episode: EvaluationEpisode, judge: ConversationJudge, *,
                   evaluator: ComponentInfo) -> EpisodeJudge:
    if not isinstance(evaluator, ComponentInfo):
        raise TypeError('evaluator must be ComponentInfo')
    if episode.summary.status == 'failed':
        return EpisodeJudge(episode.model_id, episode.episode_id, evaluator, 'excluded',
                            'simulation_failed', None, None, ())
    task = build_judge_task(episode)
    if 'scorable' not in task.segment_statuses:
        return EpisodeJudge(episode.model_id, episode.episode_id, evaluator, 'ok', None, task, None,
            tuple(JudgeMinute(i, 'unscorable', status, None, None)
                  for i, status in enumerate(task.segment_statuses)))
    raw, reason = None, None
    try:
        content = judge.complete(task)
        raw = content if isinstance(content, str) else None
        minutes = parse_judgment(task, content)
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
        minutes = tuple(JudgeMinute(i, 'evaluator_error' if status == 'scorable' else 'unscorable',
                                   reason if status == 'scorable' else status, None, None)
                        for i, status in enumerate(task.segment_statuses))
    return EpisodeJudge(episode.model_id, episode.episode_id, evaluator,
                        'evaluator_error' if reason else 'ok', reason, task, raw, minutes)
