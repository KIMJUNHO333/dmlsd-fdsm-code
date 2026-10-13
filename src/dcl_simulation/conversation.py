from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
import math

import numpy as np

from ._playback import Utterance, prepare_speech
from ._prompts import OPENING_GOAL as _OPEN, BRIDGE_GOAL as _BRIDGE, RESPONSE_GOAL as _RESPOND, RETRY_GOAL
from .episode import EpisodeFailure, EpisodeResult, TickContext, TickRecord, run_episode
from .model import DuplexModel
from .records import EvaluationProvenance, ResponseOpportunity, SpeechSegment
from .services import (
    DialogueEntry, SpeechDetector, SpeechRecognizer, SpeechSynthesizer,
    SpeechUpdate, UtteranceGenerator, UtteranceRequest, RecognizedWord, Transcription, WordTiming,
)

_EPS = 1e-9


@dataclass(frozen=True)
class ConversationConfig:
    persona: str
    topic: str
    target_duration_s: float = 600
    initial_listen_s: float = 2
    turn_end_silence_s: float = 0.8
    interrupt_overlap_s: float = 2
    no_response_s: float = 3
    max_no_response_attempts: int = 3
    termination_grace_s: float = 15

    def __post_init__(self):
        if (not isinstance(self.persona, str) or not isinstance(self.topic, str)
                or not self.persona.strip() or not self.topic.strip()):
            raise ValueError("persona and topic must be nonempty")
        for name in ("target_duration_s", "initial_listen_s", "turn_end_silence_s",
                     "interrupt_overlap_s", "no_response_s", "termination_grace_s"):
            value = getattr(self, name)
            minimum_inclusive = name in {"initial_listen_s", "termination_grace_s"}
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (value == 0 and not minimum_inclusive)):
                raise ValueError(f"invalid conversation setting: {name}")
        if (isinstance(self.max_no_response_attempts, bool)
                or not isinstance(self.max_no_response_attempts, int)
                or self.max_no_response_attempts < 1):
            raise ValueError("max_no_response_attempts must be a positive integer")


@dataclass(frozen=True)
class ConversationEvent:
    kind: str
    time_s: float
    observed_at_s: float
    details: dict[str, object]


@dataclass(frozen=True)
class UserUtterance:
    id: int
    text: str
    delivered_text: str
    start_s: float
    end_s: float
    completion: str
    chain: int
    attempt: int
    alignment: str
    sample_rate: int
    synthesized_sample_count: int
    delivered_sample_count: int
    words: tuple[WordTiming, ...]
    goal: str


@dataclass(frozen=True)
class ModelTurn:
    id: int
    start_s: float
    end_s: float
    text: str
    kind: str
    user_utterance_id: int | None
    observed_at_s: float
    truncated: bool
    segment_ids: tuple[int, ...]
    audio_sample_start: int
    audio_sample_end: int
    words: tuple[RecognizedWord, ...] | None


@dataclass(frozen=True)
class ConversationResult:
    episode: EpisodeResult
    termination_reason: str | None
    dialogue: tuple[DialogueEntry, ...]
    utterances: tuple[UserUtterance, ...]
    model_turns: tuple[ModelTurn, ...]
    events: tuple[ConversationEvent, ...]
    config: ConversationConfig
    speech_segments: tuple[SpeechSegment, ...]
    response_opportunities: tuple[ResponseOpportunity, ...]
    provenance: EvaluationProvenance | None


@dataclass
class _Turn:
    id: int
    start_s: float
    user: Utterance | None
    end_s: float | None = None
    kind: str | None = None
    segment_ids: list[int] = field(default_factory=list)


@dataclass
class _Segment:
    id: int
    turn_id: int
    start_s: float
    user_utterance_id: int | None
    observed_start_s: float
    end_s: float | None = None
    observed_end_s: float | None = None


@dataclass
class _Opportunity:
    user: Utterance
    start_s: float | None
    deadline_s: float | None
    outcome: str | None = None
    response_turn_id: int | None = None
    response_segment_id: int | None = None
    response_classification: str | None = None
    resolved_s: float | None = None
    observed_at_s: float | None = None

    def snapshot(self):
        return ResponseOpportunity(
            self.user.id, self.user.id, self.user.chain, self.user.attempt,
            self.start_s, self.deadline_s, self.outcome, self.response_turn_id,
            self.response_segment_id, self.response_classification, self.resolved_s, self.observed_at_s,
        )


def _call(name, method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"{name}: {type(exc).__name__}: {exc}") from exc


def _transcription(value, audio_start, audio_end):
    if isinstance(value, str):
        return value.strip(), None
    if not isinstance(value, Transcription) or not isinstance(value.text, str):
        raise TypeError("ASR must return Transcription or a string")
    if value.words is None:
        return value.text.strip(), None
    if not isinstance(value.words, tuple):
        raise TypeError("ASR words must be a tuple or None")
    words = []
    for word in value.words:
        if (not isinstance(word, RecognizedWord) or not isinstance(word.text, str)
                or type(word.start_sample) is not int or type(word.end_sample) is not int
                or word.start_sample < 0 or not 0 <= word.end_sample <= audio_end - audio_start):
            raise ValueError("ASR word sample positions are invalid")
        words.append(replace(word, start_sample=word.start_sample + audio_start,
                             end_sample=word.end_sample + audio_start))
    return value.text.strip(), tuple(words)


class ConversationDriver:

    def __init__(self, config: ConversationConfig, *, llm: UtteranceGenerator,
                 tts: SpeechSynthesizer, vad: SpeechDetector, asr: SpeechRecognizer,
                 provenance: EvaluationProvenance | None = None):
        if provenance is not None and not isinstance(provenance, EvaluationProvenance):
            raise TypeError("provenance must be EvaluationProvenance or None")
        self.provenance = deepcopy(provenance)
        self.config, self.llm, self.tts, self.vad, self.asr = config, llm, tts, vad, asr
        self.events: list[ConversationEvent] = []
        self.turns: list[ModelTurn] = []
        self.utterances: list[Utterance] = []
        self.termination_reason: str | None = None
        self._spec = None
        self._vad_session = None
        self._now = 0.0
        self._active: Utterance | None = None
        self._pending: tuple[Utterance | None, int] | None = None
        self._turn: _Turn | None = None
        self._speech_open = False
        self._last_event_time = 0.0
        self._activity = []
        self._activity_ends = []
        self._output_chunks = []
        self._opportunity: _Opportunity | None = None
        self._opportunities: list[_Opportunity] = []
        self._segments: list[_Segment] = []
        self._retry: tuple[Utterance, int] | None = None
        self._chain_onsets: dict[int, float] = {}
        self._chain_count = 0
        self._turn_count = 0
        self._seen_model = False
        self._bridge_pending = False
        self._response_ready = False
        self._target_reached = False

    def _event(self, kind, time_s, **details):
        self.events.append(ConversationEvent(kind, time_s, self._now, details))

    def next_input(self, context: TickContext):
        if self._spec is None:
            self._spec = context.audio_spec
            self._vad_session = _call(
                "VAD open", self.vad.open_session, sample_rate=self._spec.output_sample_rate,
            )
        if self._pending is not None:
            raise RuntimeError("previous input has not been observed")
        if self.termination_reason is not None:
            return None
        now = float(context.time_start)
        if not self._target_reached and self._active is None and not self._speech_open:
            if self._retry is not None:
                previous, attempt = self._retry
                self._retry = None
                self._generate(now, previous.goal, chain=previous.chain, attempt=attempt)
            elif self._response_ready and self._turn is None:
                self._response_ready = False
                self._generate(now, _BRIDGE if self._bridge_pending else _RESPOND)
            elif not self.utterances and not self._seen_model and now + _EPS >= self.config.initial_listen_s:
                self._generate(now, _OPEN)
        audio = np.zeros(self._spec.input_chunk_samples, dtype=np.float32)
        take = 0
        if self._active is not None:
            take = min(audio.size, self._active.speech.audio.size - self._active.cursor)
            audio[:take] = self._active.speech.audio[self._active.cursor:self._active.cursor + take]
        self._pending = (self._active, take)
        return audio

    def _generate(self, now, goal, *, chain=None, attempt=1):
        if chain is None:
            self._chain_count += 1
            chain = self._chain_count
        request_goal = goal if attempt == 1 else RETRY_GOAL.format(content_goal=goal)
        request = UtteranceRequest(self.config.persona, self.config.topic, self.dialogue,
                                   request_goal, attempt)
        text = _call("LLM", self.llm.generate, request)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM must return nonempty utterance text")
        text = text.strip()
        speech = _call("TTS", self.tts.synthesize, text, sample_rate=self._spec.input_sample_rate)
        words = prepare_speech(text, speech, self._spec.input_sample_rate)
        speech = replace(speech, audio=speech.audio.copy())
        self._active = Utterance(len(self.utterances) + 1, text, goal, chain, attempt,
                                 speech, now, words)
        self.utterances.append(self._active)
        self._event("user_generated", now, utterance_id=self._active.id, chain=chain, attempt=attempt)

    def observe(self, record: TickRecord):
        self._now = float(record.context.time_end)
        user, take = self._pending
        self._pending = None
        if user is not None and take:
            if user.cursor == 0:
                self._event("user_started", user.start_s, utterance_id=user.id)
            user.cursor += take
            if user.cursor == user.speech.audio.size:
                self._complete_user(user, user.natural_end_s, "natural")
        self._output_chunks.append(record.output.audio)
        update = _call("VAD", self._vad_session.process, record.output.audio)
        self._consume_update(update, allow_policy=True)
        self._classify_overlap()
        if self._turn is not None and not self._speech_open:
            if self._now + _EPS >= self._turn.end_s + self.config.turn_end_silence_s:
                self._finish_turn(transcribe=True, allow_policy=True)
        if not self._speech_open:
            self._expire_opportunity(self._now)
        self._update_termination()

    def _complete_user(self, user, end_s, completion, *, allow_policy=True):
        user.end_s, user.completion = end_s, completion
        self._active = None
        self._event("user_completed", end_s, utterance_id=user.id,
                    completion=completion, delivered_text=user.delivered_text)
        if completion == "natural" and user.goal == _BRIDGE:
            self._bridge_pending = False
        if not allow_policy:
            return
        self._opportunity = _Opportunity(user, end_s, end_s + self.config.no_response_s)
        self._opportunities.append(self._opportunity)
        self._event("response_opportunity", end_s, utterance_id=user.id,
                    chain=user.chain, attempt=user.attempt,
                    deadline_s=self._opportunity.deadline_s)
        if self._turn is not None and self._turn.user is user:
            if self._turn.kind in {"normal", "interrupting"}:
                self._resolve_response(end_s)
            else:
                self._opportunity.start_s = self._opportunity.deadline_s = None
                self._event("response_classification_pending", self._now, utterance_id=user.id)

    def _consume_update(self, update, *, allow_policy):
        if not isinstance(update, SpeechUpdate):
            raise TypeError("VAD must return SpeechUpdate")
        for span in update.activity:
            previous = self._activity_ends[-1] if self._activity_ends else 0
            if (not math.isfinite(span.start_s) or not math.isfinite(span.end_s)
                    or span.start_s + _EPS < previous or span.end_s <= span.start_s
                    or span.end_s > self._now + _EPS):
                raise ValueError("invalid or duplicated VAD activity span")
            self._activity.append(span)
            self._activity_ends.append(span.end_s)
            self._event("speech_activity", span.start_s, end_s=span.end_s)
        for event in update.events:
            if (not math.isfinite(event.time_s) or event.time_s + _EPS < self._last_event_time
                    or event.time_s > self._now + _EPS):
                raise ValueError("VAD event is out of order or outside received audio")
            if event.kind == "start" and not self._speech_open:
                self._start_speech(event.time_s, allow_policy=allow_policy)
            elif event.kind == "end" and self._speech_open:
                self._speech_open = False
                self._turn.end_s = event.time_s
                self._segments[-1].end_s = event.time_s
                self._segments[-1].observed_end_s = self._now
            else:
                raise ValueError("VAD events must alternate start and end")
            self._last_event_time = event.time_s
            self._event("speech_" + event.kind, event.time_s, turn_id=self._turn.id,
                        segment_id=self._segments[-1].id)

    def _start_speech(self, at_s, *, allow_policy):
        if self._turn is not None and at_s + _EPS >= self._turn.end_s + self.config.turn_end_silence_s:
            self._classify_overlap(allow_policy=allow_policy)
            self._finish_turn(transcribe=True, allow_policy=allow_policy)
        if allow_policy:
            self._expire_opportunity(at_s)
            if self._retry is not None:
                self._event("late_response", at_s, chain=self._retry[0].chain)
                self._retry = None
        self._response_ready = False
        if not self.utterances:
            self._bridge_pending = True
        self._seen_model = True
        if self._turn is None:
            user = next((u for u in reversed(self.utterances) if u.cursor and
                         u.start_s <= at_s < (u.end_s if u.end_s is not None else u.natural_end_s)), None)
            self._turn_count += 1
            self._turn = _Turn(self._turn_count, at_s, user, kind="normal" if user is None else None)
        self._turn.end_s = None
        self._speech_open = True
        segment = _Segment(len(self._segments) + 1, self._turn.id, at_s,
                           self._turn.user.id if self._turn.user else None, self._now)
        self._segments.append(segment)
        self._turn.segment_ids.append(segment.id)
        if allow_policy and self._opportunity is not None:
            if (self._turn.user is self._opportunity.user
                    and self._turn.kind not in {"normal", "interrupting"}):
                self._opportunity.start_s = self._opportunity.deadline_s = None
                self._event("response_classification_pending", self._now,
                            utterance_id=self._opportunity.user.id)
            else:
                self._resolve_response(at_s)

    def _classify_overlap(self, *, allow_policy=True):
        turn = self._turn
        if turn is None or turn.user is None or turn.kind is not None:
            return
        user = turn.user
        end = min(self._now, turn.end_s if turn.end_s is not None else self._now,
                  user.end_s if user.end_s is not None else user.natural_end_s)
        start = max(turn.start_s, user.start_s)
        total = 0.0
        crossing = None
        for span in self._activity[bisect_right(self._activity_ends, start):]:
            if span.start_s >= end:
                break
            left, right = max(span.start_s, start), min(span.end_s, end)
            if right <= left:
                continue
            if total + right - left + _EPS >= self.config.interrupt_overlap_s:
                crossing = left + self.config.interrupt_overlap_s - total
                break
            total += right - left
        if crossing is not None:
            if user.completion == "natural":
                turn.kind = "normal"
            elif self._active is user and allow_policy:
                turn.kind = "interrupting"
                self._complete_user(user, self._now, "model_interrupt")
                self._event("user_cutoff", self._now, utterance_id=user.id,
                            turn_id=turn.id, threshold_time_s=crossing)
        elif user.completion == "natural" and (turn.end_s is None or turn.end_s + _EPS >= user.end_s):
            turn.kind = "normal"
        if allow_policy and turn.kind in {"normal", "interrupting"}:
            self._resolve_response(user.end_s or self._now)

    def _finish_turn(self, *, transcribe, allow_policy, truncated=False):
        turn = self._turn
        if turn is None:
            return
        end = self._now if turn.end_s is None else turn.end_s
        if turn.kind is None:
            turn.kind = ("normal" if turn.user.end_s is not None and
                         turn.user.end_s <= end + _EPS else "backchannel")
        text = ""
        words = None
        audio_start = round(turn.start_s * self._spec.output_sample_rate)
        audio_end = round(end * self._spec.output_sample_rate)
        if transcribe:
            transcription = _call("ASR", self.asr.transcribe, self._audio_range(turn.start_s, end),
                                  sample_rate=self._spec.output_sample_rate)
            text, words = _transcription(transcription, audio_start, audio_end)
        self.turns.append(ModelTurn(turn.id, turn.start_s, end, text, turn.kind,
                                    turn.user.id if turn.user else None, self._now, truncated,
                                    tuple(turn.segment_ids), audio_start, audio_end, words))
        self._event("model_turn", end, turn_id=turn.id, classification=turn.kind, truncated=truncated)
        if allow_policy:
            if turn.kind == "backchannel":
                if self._opportunity is not None and self._opportunity.user is turn.user:
                    self._opportunity.start_s = max(turn.user.end_s, end)
                    self._opportunity.deadline_s = self._opportunity.start_s + self.config.no_response_s
                    self._event("response_deadline", self._opportunity.start_s,
                                utterance_id=turn.user.id, deadline_s=self._opportunity.deadline_s)
            else:
                self._resolve_response(turn.start_s)
                self._response_ready = True
        self._turn = None

    def _audio_range(self, start_s, end_s):
        rate, chunk = self._spec.output_sample_rate, self._spec.output_chunk_samples
        start = max(0, round(start_s * rate))
        end = min(len(self._output_chunks) * chunk, round(end_s * rate))
        if end <= start:
            raise ValueError("VAD produced an empty turn")
        return np.concatenate([
            self._output_chunks[i][max(0, start - i * chunk):min(chunk, end - i * chunk)]
            for i in range(start // chunk, (end - 1) // chunk + 1)
        ])

    def _resolve_response(self, at_s):
        opportunity = self._opportunity
        if opportunity is None:
            return
        user = opportunity.user
        turn = self._turn
        if turn is None or not turn.segment_ids or turn.kind not in {"normal", "interrupting"}:
            raise RuntimeError("resolved response must refer to a classified speech segment")
        if opportunity.start_s is None:
            opportunity.start_s = user.end_s
            opportunity.deadline_s = user.end_s + self.config.no_response_s
        opportunity.outcome = "response"
        opportunity.response_turn_id = turn.id
        opportunity.response_segment_id = turn.segment_ids[0]
        opportunity.response_classification = turn.kind
        opportunity.resolved_s, opportunity.observed_at_s = at_s, self._now
        self._event("response", at_s, utterance_id=user.id, chain=user.chain, attempt=user.attempt,
                    opportunity_id=user.id, turn_id=turn.id, segment_id=turn.segment_ids[0],
                    classification=turn.kind)
        self._chain_onsets.pop(user.chain, None)
        self._opportunity = None

    def _expire_opportunity(self, at_s):
        opportunity = self._opportunity
        if opportunity is None or opportunity.deadline_s is None or at_s + _EPS < opportunity.deadline_s:
            return
        user, deadline = opportunity.user, opportunity.deadline_s
        opportunity.outcome = "no_response"
        opportunity.resolved_s, opportunity.observed_at_s = deadline, self._now
        self._event("no_response", deadline, utterance_id=user.id, chain=user.chain, attempt=user.attempt)
        self._chain_onsets.setdefault(user.chain, opportunity.start_s)
        self._opportunity = None
        if user.attempt >= self.config.max_no_response_attempts:
            self.termination_reason = "response_collapse"
            self._event("response_collapse", deadline, onset_s=self._chain_onsets[user.chain],
                        chain=user.chain)
        else:
            self._retry = (user, user.attempt + 1)

    def _update_termination(self):
        if self._now + _EPS >= self.config.target_duration_s:
            if not self._target_reached:
                self._event("target_reached", self.config.target_duration_s)
            self._target_reached = True
            if self._retry is not None:
                self._event("retry_suppressed_by_termination", self._now, chain=self._retry[0].chain)
                self._retry = None
        if self.termination_reason is not None or not self._target_reached:
            return
        if self._now + _EPS >= self.config.target_duration_s + self.config.termination_grace_s:
            self.termination_reason = "grace_period_elapsed"
        elif self._active is None and not self._speech_open and self._opportunity is None:
            self.termination_reason = "target_drained"

    @property
    def dialogue(self):
        entries = []
        included_turns = set()
        for user in self.utterances:
            if user.completion is None or not user.cursor:
                continue
            previous_char = 0
            previous_time = user.start_s
            last_user_index = None
            for turn in self.turns:
                if turn.kind != "backchannel" or turn.user_utterance_id != user.id:
                    continue
                cut = user.prefix_end(turn.end_s)
                prefix = user.text[previous_char:cut].strip()
                if prefix:
                    entries.append(DialogueEntry("user", prefix, previous_time, min(turn.end_s, user.end_s),
                                                 user_utterance_id=user.id))
                    last_user_index = len(entries) - 1
                if turn.text:
                    entries.append(DialogueEntry("model", turn.text, turn.start_s, turn.end_s,
                                                 "backchannel", turn.truncated, model_turn_id=turn.id))
                included_turns.add(turn.id)
                previous_char, previous_time = cut, min(turn.end_s, user.end_s)
            tail = user.text[previous_char:user.prefix_end(user.end_s)].strip()
            if tail:
                entries.append(DialogueEntry("user", tail, previous_time, user.end_s,
                                             user_utterance_id=user.id))
                last_user_index = len(entries) - 1
            if last_user_index is not None and user.completion != "natural":
                entries[last_user_index] = replace(entries[last_user_index], truncated=True)
        for turn in self.turns:
            if turn.id not in included_turns and turn.text:
                entries.append(DialogueEntry("model", turn.text, turn.start_s, turn.end_s,
                                             turn.kind, turn.truncated, model_turn_id=turn.id))
        return tuple(sorted(entries, key=lambda entry: (entry.end_s, entry.speaker != "user")))

    def finalize(self, *, successful):
        self._pending = None
        try:
            if successful and self._vad_session is not None:
                self._consume_update(_call("VAD finalize", self._vad_session.finalize), allow_policy=False)
        finally:
            if self._active is not None:
                self._complete_user(self._active, self._now,
                                    "episode_end" if self._active.cursor else "pre_play_cancelled",
                                    allow_policy=False)
        if self._turn is not None:
            self._finish_turn(transcribe=successful, allow_policy=False, truncated=self._speech_open)
            self._speech_open = False

    def close(self):
        if self._vad_session is not None:
            session, self._vad_session = self._vad_session, None
            _call("VAD close", session.close)

    def result(self, episode):
        users = tuple(UserUtterance(u.id, u.text, u.delivered_text, u.start_s,
                                    u.end_s if u.end_s is not None else self._now,
                                    u.completion or "episode_error", u.chain, u.attempt, u.alignment,
                                    u.speech.sample_rate, u.speech.audio.size, u.cursor, u.words, u.goal)
                      for u in self.utterances)
        kinds = {turn.id: turn.kind for turn in self.turns}
        if self._turn is not None:
            kinds[self._turn.id] = self._turn.kind
        segments = tuple(SpeechSegment(
            s.id, s.turn_id, s.start_s, s.end_s if s.end_s is not None else self._now,
            round(s.start_s * self._spec.output_sample_rate),
            round((s.end_s if s.end_s is not None else self._now) * self._spec.output_sample_rate),
            kinds.get(s.turn_id), s.user_utterance_id, s.observed_start_s, s.observed_end_s,
            s.end_s is None,
        ) for s in self._segments)
        return ConversationResult(episode, self.termination_reason if episode.status == "completed" else None,
                                  self.dialogue, users, tuple(self.turns), tuple(self.events), self.config,
                                  segments, tuple(o.snapshot() for o in self._opportunities), self.provenance)


def run_conversation(model: DuplexModel, config: ConversationConfig, *,
                     llm: UtteranceGenerator, tts: SpeechSynthesizer,
                     vad: SpeechDetector, asr: SpeechRecognizer,
                     provenance: EvaluationProvenance | None = None) -> ConversationResult:
    driver = ConversationDriver(config, llm=llm, tts=tts, vad=vad, asr=asr, provenance=provenance)
    try:
        episode = run_episode(model, driver, max_duration_s=(
            Fraction(str(config.target_duration_s)) + Fraction(str(config.termination_grace_s))
        ))
    except BaseException as original:
        try:
            driver.close()
        except Exception as cleanup:
            raise original from cleanup
        raise
    errors = []
    try:
        try:
            driver.finalize(successful=episode.status == "completed")
        except Exception as exc:
            errors.append(EpisodeFailure("conversation_finalize", None, type(exc).__name__, str(exc)))
    finally:
        try:
            driver.close()
        except Exception as exc:
            errors.append(EpisodeFailure("vad_close", None, type(exc).__name__, str(exc)))
    if errors:
        episode = replace(episode, failures=episode.failures + tuple(errors), stop_reason=None)
    return driver.result(episode)
