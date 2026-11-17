from dataclasses import dataclass, fields, is_dataclass
import json
from fractions import Fraction
import math
from pathlib import Path
import types
from typing import get_args, get_origin, get_type_hints, Literal

from ..artifacts import FORMAT_NAME, SCHEMA_VERSION
from ..audio import AudioSpec
from ..conversation import ConversationConfig, ConversationEvent, ModelTurn, UserUtterance
from ..episode import EpisodeFailure
from ..model import ModelMeasurements
from ..records import EvaluationProvenance, ResponseOpportunity, SpeechSegment
from ..services import DialogueEntry


class EvaluationInputError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeSummary:
    status: Literal['completed', 'failed']
    termination_reason: str | None
    stop_reason: str | None
    requested_duration_s: float
    logical_duration_s: float
    planned_tick_count: int
    recorded_tick_count: int
    model_calls: int
    no_response_count: int
    collapse_onset_time_s: float | None
    collapse_decision_time_s: float | None
    failures: tuple[EpisodeFailure, ...]


@dataclass(frozen=True)
class SavedTick:
    tick_index: int
    time_start_s: float
    time_end_s: float
    input_sample_start: int
    input_sample_end: int
    output_sample_start: int
    output_sample_end: int
    native_text: str | None
    generation_unit_id: str | None
    measurements: ModelMeasurements | None


@dataclass(frozen=True)
class EvaluationEpisode:
    directory: Path
    model_id: str
    episode_id: str
    config: ConversationConfig
    summary: EpisodeSummary
    audio_spec: AudioSpec | None
    provenance: EvaluationProvenance | None
    ticks: tuple[SavedTick, ...]
    events: tuple[ConversationEvent, ...]
    dialogue: tuple[DialogueEntry, ...]
    user_utterances: tuple[UserUtterance, ...]
    model_turns: tuple[ModelTurn, ...]
    speech_segments: tuple[SpeechSegment, ...]
    response_opportunities: tuple[ResponseOpportunity, ...]


def _require(condition, message):
    if not condition:
        raise EvaluationInputError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(text):
    value = json.loads(text, object_pairs_hook=_object)
    json.dumps(value, allow_nan=False)
    return value


def _decode(cls, value):
    origin, args = get_origin(cls), get_args(cls)
    if origin is types.UnionType:
        for candidate in args:
            try:
                return _decode(candidate, value)
            except (ValueError, TypeError):
                pass
        raise EvaluationInputError(f'{cls} has an invalid type')
    if origin is Literal:
        _require(value in args, f'Unsupported value: {value!r}')
        return value
    if is_dataclass(cls):
        _require(isinstance(value, dict), f'{cls.__name__} must be an object')
        hints = get_type_hints(cls)
        names = {f.name for f in fields(cls)}
        _require(set(value) == names, f'{cls.__name__} fields do not match: {set(value) ^ names}')
        return cls(**{key: _decode(hints[key], item) for key, item in value.items()})
    if origin is tuple:
        _require(isinstance(value, list), 'Tuple records must be JSON arrays')
        return tuple(_decode(args[0], item) for item in value)
    if origin is dict:
        _require(isinstance(value, dict), 'Expected an object')
        return {_decode(args[0], key): _decode(args[1], item) for key, item in value.items()}
    if cls is object:
        return value
    if cls is float:
        _require(type(value) in (int, float) and math.isfinite(value), 'Expected a finite number')
        return float(value)
    _require(type(value) is cls, f'{cls} has an invalid type: {value!r}')
    return value


def _file(root, name):
    path = root / name
    _require(path.is_file() and path.resolve().is_relative_to(root), f'Result file is missing or outside the episode directory: {name}')
    return path


def load_episode(directory: str | Path) -> EvaluationEpisode:
    root = Path(directory).expanduser().resolve()
    try:
        manifest = _json(_file(root, 'manifest.json').read_text(encoding='utf-8'))
        _require(manifest['format'] == FORMAT_NAME and type(manifest['schema_version']) is int
                 and manifest['schema_version'] == SCHEMA_VERSION, 'Unsupported result schema')
        for key in ('model_id', 'episode_id'):
            _require(isinstance(manifest[key], str) and manifest[key].strip(), f'{key} is empty')
        record_types = dict(ticks=SavedTick, events=ConversationEvent, dialogue=DialogueEntry,
                            user_utterances=UserUtterance, model_turns=ModelTurn,
                            speech_segments=SpeechSegment, response_opportunities=ResponseOpportunity)
        records = {}
        for key, cls in record_types.items():
            name = key + '.jsonl'
            _require(manifest['files'][key] == name, f'{key} filename does not match the schema')
            rows = []
            for index, line in enumerate(_file(root, name).read_text(encoding='utf-8').splitlines(), 1):
                if line.strip():
                    try:
                        rows.append(_decode(cls, _json(line)))
                    except (ValueError, TypeError) as exc:
                        raise EvaluationInputError(f'{name}:{index}: {exc}') from exc
            records[key] = tuple(rows)
        episode = EvaluationEpisode(
            root, manifest['model_id'], manifest['episode_id'],
            _decode(ConversationConfig, manifest['config']), _decode(EpisodeSummary, manifest['summary']),
            _decode(AudioSpec | None, manifest['audio_spec']),
            _decode(EvaluationProvenance | None, manifest['provenance']), **records,
        )
        _validate_execution(episode, manifest['audio'])
        if episode.summary.status == 'completed':
            _validate_conversation(episode)
        return episode
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise EvaluationInputError(f'{root}: {exc}') from exc


def _close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-6)


def _validate_execution(e, audio):
    s, spec = e.summary, e.audio_spec
    _require(bool(s.failures) == (s.status == 'failed'), 'Execution status does not match the failure records')
    _require(s.recorded_tick_count == len(e.ticks) and s.model_calls >= len(e.ticks)
             and s.planned_tick_count >= len(e.ticks), 'Tick counts do not match the execution summary')
    _require(_close(s.requested_duration_s, e.config.target_duration_s + e.config.termination_grace_s),
             'Requested duration does not match the conversation config')
    _require(s.no_response_count >= 0, 'No-response count is negative')
    if s.status == 'failed':
        _require(s.termination_reason is None and s.stop_reason is None, 'Failed execution has a normal termination reason')
    if spec is None:
        _require(not e.ticks and audio == {} and s.logical_duration_s == 0 and s.status == 'failed',
                 'Result without an audio spec contains audio or a completed status')
        return
    duration = float(spec.tick_duration)
    _require(_close(s.logical_duration_s, len(e.ticks) * duration), 'Logical duration does not match the tick count')
    planned = math.ceil((Fraction(str(e.config.target_duration_s))
                         + Fraction(str(e.config.termination_grace_s))) / spec.tick_duration)
    _require(s.planned_tick_count == planned,
             'Planned tick count does not match the duration settings')
    for index, tick in enumerate(e.ticks):
        _require(tick.tick_index == index and _close(tick.time_start_s, index * duration)
                 and _close(tick.time_end_s, (index + 1) * duration), 'Tick indices or times are not contiguous')
        for prefix, size in (('input', spec.input_chunk_samples), ('output', spec.output_chunk_samples)):
            _require(getattr(tick, prefix + '_sample_start') == index * size
                     and getattr(tick, prefix + '_sample_end') == (index + 1) * size, 'Tick sample ranges do not match')
        _require(tick.generation_unit_id is None or tick.generation_unit_id.strip(), 'Empty generation unit ID')
    import soundfile as sf
    _require(set(audio) == {'user', 'model'}, 'Input and output audio entries do not match')
    for name, rate, size in (('user', spec.input_sample_rate, spec.input_chunk_samples),
                             ('model', spec.output_sample_rate, spec.output_chunk_samples)):
        expected = dict(path=name + '.wav', sample_rate=rate, channels=1,
                        sample_count=len(e.ticks) * size, format='WAV', subtype='FLOAT')
        _require(audio[name] == expected, f'{name} audio metadata does not match')
        info = sf.info(_file(e.directory, expected['path']))
        _require((info.samplerate, info.channels, info.frames, info.format, info.subtype)
                 == (rate, 1, expected['sample_count'], 'WAV', 'FLOAT'), f'{name} audio header does not match the records')
    if s.status == 'completed':
        _require(s.stop_reason in ('driver_finished', 'duration_reached'), 'Missing completion reason')
        _require(s.termination_reason in ('target_drained', 'grace_period_elapsed', 'response_collapse'),
                 'Unsupported conversation termination reason')
        _require(s.model_calls == len(e.ticks), 'Inference count does not match the completed execution')
        _require(s.logical_duration_s <= s.planned_tick_count * duration + 1e-6, 'Execution exceeds the planned duration')
        if s.termination_reason != 'response_collapse':
            _require(s.logical_duration_s + 1e-6 >= e.config.target_duration_s, 'Result completed normally before the target duration')
    else:
        _require(s.termination_reason is None and s.stop_reason is None, 'Failed execution has a normal termination reason')


def _index(rows, label):
    result = {}
    for row in rows:
        _require(row.id > 0 and row.id not in result, f'{label} IDs are duplicated or out of range')
        result[row.id] = row
    return result


def _span(start, end, limit, label):
    _require(0 <= start and start <= end + 1e-6 and 0 <= end <= limit + 1e-6, f'{label} time range is invalid')


def _validate_conversation(e):
    limit, rate = e.summary.logical_duration_s, e.audio_spec.output_sample_rate
    users = _index(e.user_utterances, 'user utterance')
    turns = _index(e.model_turns, 'model turn')
    segments = _index(e.speech_segments, 'VAD segment')
    _index(e.response_opportunities, 'response opportunity')
    previous_end = 0
    for user in e.user_utterances:
        _span(user.start_s, user.end_s, limit, 'user utterance')
        _require(user.start_s + 1e-6 >= previous_end, 'User utterances overlap')
        previous_end = user.end_s
        _require(user.chain > 0 and user.attempt > 0 and user.sample_rate == e.audio_spec.input_sample_rate
                 and 0 <= user.delivered_sample_count <= user.synthesized_sample_count,
                 'Invalid user playback samples or retry records')
        _require(_close(user.end_s - user.start_s, user.delivered_sample_count / user.sample_rate),
                 'User playback duration does not match the sample count')
        _require(user.completion in ('natural', 'model_interrupt', 'episode_end', 'pre_play_cancelled'),
                 'Invalid user utterance completion reason')
        _require(user.alignment in ('word_timestamps', 'word_ratio'), 'Invalid TTS alignment method')
        _require(bool(user.words) == (user.alignment == 'word_timestamps'), 'TTS alignment metadata does not match')
        if user.completion == 'natural':
            _require(user.delivered_sample_count == user.synthesized_sample_count, 'Invalid audio length for a naturally completed utterance')
        previous_char, previous_time = 0, 0
        for word in user.words:
            _require(previous_char <= word.char_start < word.char_end <= len(user.text)
                     and previous_time <= word.start_s <= word.end_s
                     <= user.synthesized_sample_count / user.sample_rate + 1e-6, 'Invalid TTS word alignment')
            previous_char, previous_time = word.char_end, word.end_s
    previous_end = 0
    for segment in e.speech_segments:
        _span(segment.start_s, segment.end_s, limit, 'VAD segment')
        _require(segment.start_s + 1e-6 >= previous_end, 'VAD segments overlap or are out of order')
        previous_end = segment.end_s
        _require(segment.turn_id in turns and segment.kind in ('normal', 'backchannel', 'interrupting'),
                 'VAD segment has no turn or classification')
        _require(segment.audio_sample_start == round(segment.start_s * rate)
                 and segment.audio_sample_end == round(segment.end_s * rate), 'Invalid VAD sample positions')
        _span(segment.start_s, segment.observed_start_s, limit, 'VAD start observation')
        if segment.observed_end_s is None:
            _require(segment.truncated and _close(segment.end_s, limit), 'Invalid end record for an open VAD segment')
        else:
            _span(segment.end_s, segment.observed_end_s, limit, 'VAD end observation')
            _require(not segment.truncated, 'VAD segment with an end event is marked as truncated')
        _require(segment.user_utterance_id is None or segment.user_utterance_id in users, 'Unknown overlapping user utterance')
        if segment.kind != 'normal':
            _require(segment.user_utterance_id in users, 'Backchannel or interruption has no linked user utterance')
    claimed = set()
    for turn in e.model_turns:
        _span(turn.start_s, turn.end_s, limit, 'model turn')
        _require(turn.segment_ids and len(set(turn.segment_ids)) == len(turn.segment_ids), 'Invalid turn segment list')
        grouped = []
        for sid in turn.segment_ids:
            _require(sid in segments and sid not in claimed, 'Unknown segment or duplicate turn link')
            claimed.add(sid)
            segment = segments[sid]
            _require(segment.turn_id == turn.id and segment.kind == turn.kind
                     and segment.user_utterance_id == turn.user_utterance_id, 'Turn and segment links or classifications differ')
            grouped.append(segment)
        _require([s.id for s in grouped] == [s.id for s in sorted(grouped, key=lambda s: s.start_s)], 'Turn segments are out of order')
        _require(_close(turn.start_s, grouped[0].start_s) and _close(turn.end_s, grouped[-1].end_s), 'Invalid turn boundaries')
        _require(turn.audio_sample_start == grouped[0].audio_sample_start
                 and turn.audio_sample_end == grouped[-1].audio_sample_end, 'Invalid turn sample positions')
        _require(turn.truncated == grouped[-1].truncated, 'Turn truncation does not match its segment')
        _span(turn.end_s, turn.observed_at_s, limit, 'model turn observation')
        for word in turn.words or ():
            _require(word.start_sample >= turn.audio_sample_start
                     and turn.audio_sample_start <= word.end_sample <= turn.audio_sample_end, 'Invalid ASR word sample positions')
    _require(claimed == set(segments), 'VAD segment is not linked to a turn')
    _require({o.user_utterance_id for o in e.response_opportunities}
             == {u.id for u in users.values() if u.completion in ('natural', 'model_interrupt')},
             'Completed user utterances do not match response opportunities')
    for opportunity in e.response_opportunities:
        _require(opportunity.user_utterance_id in users, 'Response opportunity refers to an unknown user utterance')
        user = users[opportunity.user_utterance_id]
        _require(opportunity.id == user.id and (opportunity.chain, opportunity.attempt) == (user.chain, user.attempt),
                 'Invalid retry link for a response opportunity')
        _require(opportunity.outcome in (None, 'response', 'no_response'), 'Invalid response outcome')
        _require((opportunity.start_s is None) == (opportunity.deadline_s is None), 'Response wait start and deadline are inconsistent')
        if opportunity.start_s is not None:
            _span(0, opportunity.start_s, limit, 'response wait start')
            _require(_close(opportunity.deadline_s, opportunity.start_s + e.config.no_response_s), 'Invalid response deadline')
        if opportunity.outcome is None:
            _require(opportunity.resolved_s is None and opportunity.observed_at_s is None, 'Unresolved response has a resolution timestamp')
        else:
            _require(opportunity.start_s is not None and opportunity.resolved_s is not None
                     and opportunity.observed_at_s is not None, 'Resolved response is missing timestamps')
            _span(0, opportunity.resolved_s, limit, 'response resolution')
            _span(opportunity.resolved_s, opportunity.observed_at_s, limit, 'response observation')
        if opportunity.outcome == 'response':
            _require(opportunity.response_segment_id in segments and opportunity.response_turn_id in turns,
                     'Missing response segment or turn link')
            segment = segments[opportunity.response_segment_id]
            turn = turns[opportunity.response_turn_id]
            _require(segment.id == turn.segment_ids[0] and segment.turn_id == turn.id
                     and opportunity.response_classification == segment.kind
                     and segment.kind in ('normal', 'interrupting'), 'Response classification or anchor segment does not match')
            _require((turn.user_utterance_id is None and segment.start_s + 1e-6 >= user.end_s)
                     or turn.user_utterance_id == user.id, 'Response is linked to a different user utterance')
        else:
            _require(opportunity.response_segment_id is None and opportunity.response_turn_id is None
                     and opportunity.response_classification is None, 'No-response outcome has a response link')
            if opportunity.outcome == 'no_response':
                _require(_close(opportunity.resolved_s, opportunity.deadline_s), 'No-response decision does not match the deadline')
    for row in e.dialogue:
        _span(row.start_s, row.end_s, limit, 'dialogue history')
        _require((row.speaker == 'user' and row.user_utterance_id in users and row.model_turn_id is None)
                 or (row.speaker == 'model' and row.model_turn_id in turns and row.user_utterance_id is None),
                 'Invalid utterance link in dialogue history')
    for event in e.events:
        _span(0, event.time_s, limit, 'event')
        _span(event.time_s, event.observed_at_s, limit, 'event observation')
    misses = [o for o in e.response_opportunities if o.outcome == 'no_response']
    _require(len(misses) == e.summary.no_response_count == sum(x.kind == 'no_response' for x in e.events),
             'No-response summary does not match the records')
    if e.summary.termination_reason == 'response_collapse':
        onset, decision = e.summary.collapse_onset_time_s, e.summary.collapse_decision_time_s
        _require(onset is not None and decision is not None, 'Missing collapse timestamps')
        _span(onset, decision, limit, 'collapse')
        chains = {o.chain for o in misses}
        _require(any(len(group := [o for o in misses if o.chain == chain]) == e.config.max_no_response_attempts
                     and sorted(o.attempt for o in group) == list(range(1, e.config.max_no_response_attempts + 1))
                     and _close(min(o.start_s for o in group), onset)
                     and _close(max(o.resolved_s for o in group), decision) for chain in chains),
                 'Collapse does not match the consecutive no-response chain')
    else:
        _require(e.summary.collapse_onset_time_s is None and e.summary.collapse_decision_time_s is None,
                 'Normal termination contains collapse timestamps')
