from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Literal, Protocol

from .audio import Audio, AudioSpec
from .model import DuplexModel, DuplexSession, ModelMeasurements, ModelOutput


@dataclass(frozen=True)
class TickContext:

    index: int
    audio_spec: AudioSpec

    @property
    def time_start(self) -> Fraction:
        return self.index * self.audio_spec.tick_duration

    @property
    def time_end(self) -> Fraction:
        return (self.index + 1) * self.audio_spec.tick_duration


@dataclass(frozen=True)
class TickRecord:

    context: TickContext
    input_audio: Audio
    output: ModelOutput


class InputDriver(Protocol):

    def next_input(self, context: TickContext) -> Audio | None:
        ...

    def observe(self, record: TickRecord) -> None:
        ...


@dataclass(frozen=True)
class EpisodeFailure:
    phase: str
    tick_index: int | None
    error_type: str
    message: str


@dataclass(frozen=True)
class EpisodeResult:

    requested_duration: Fraction
    audio_spec: AudioSpec | None
    planned_tick_count: int
    ticks: tuple[TickRecord, ...]
    model_calls: int
    stop_reason: Literal["duration_reached", "driver_finished"] | None
    failures: tuple[EpisodeFailure, ...]

    @property
    def status(self) -> Literal["completed", "failed"]:
        return "failed" if self.failures else "completed"

    @property
    def logical_duration(self) -> Fraction:
        if self.audio_spec is None:
            return Fraction(0)
        return len(self.ticks) * self.audio_spec.tick_duration


def run_episode(
    model: DuplexModel,
    driver: InputDriver,
    *,
    max_duration_s: int | float | Fraction,
) -> EpisodeResult:
    duration = _duration(max_duration_s)
    session: DuplexSession | None = None
    spec: AudioSpec | None = None
    ticks: list[TickRecord] = []
    failures: list[EpisodeFailure] = []
    planned_ticks = 0
    model_calls = 0
    stop_reason: Literal["duration_reached", "driver_finished"] | None = None
    phase = "open_session"
    tick_index: int | None = None

    try:
        session = model.open_session()
        phase = "audio_spec"
        candidate = session.audio_spec
        if not isinstance(candidate, AudioSpec):
            raise TypeError("session.audio_spec must be an AudioSpec")
        spec = candidate
        planned_ticks = math.ceil(duration / spec.tick_duration)

        for tick_index in range(planned_ticks):
            context = TickContext(tick_index, spec)
            phase = "input"
            input_audio = driver.next_input(context)
            if input_audio is None:
                stop_reason = "driver_finished"
                break
            spec.validate_input(input_audio)
            input_snapshot = _snapshot(input_audio)

            phase = "model"
            model_calls += 1
            output = session.process_tick(input_audio)
            phase = "output"
            if not isinstance(output, ModelOutput):
                raise TypeError("process_tick must return a ModelOutput")
            spec.validate_output(output.audio)
            if output.native_text is not None and not isinstance(output.native_text, str):
                raise TypeError("native_text must be a string or None")
            if output.generation_unit_id is not None and (
                    not isinstance(output.generation_unit_id, str) or not output.generation_unit_id.strip()):
                raise TypeError("generation_unit_id must be a nonempty string or None")
            if output.measurements is not None and not isinstance(output.measurements, ModelMeasurements):
                raise TypeError("measurements must be ModelMeasurements or None")

            record = TickRecord(
                context=context,
                input_audio=input_snapshot,
                output=replace(output, audio=_snapshot(output.audio)),
            )
            ticks.append(record)
            phase = "observe"
            driver.observe(record)
        else:
            stop_reason = "duration_reached"
    except Exception as exc:
        failures.append(_failure(phase, tick_index, exc))
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                failures.append(_failure("close_session", None, exc))

    return EpisodeResult(
        requested_duration=duration,
        audio_spec=spec,
        planned_tick_count=planned_ticks,
        ticks=tuple(ticks),
        model_calls=model_calls,
        stop_reason=None if failures else stop_reason,
        failures=tuple(failures),
    )


def _duration(value: int | float | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("max_duration_s must be a positive finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("max_duration_s must be a positive finite number")
    duration = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    if duration <= 0:
        raise ValueError("max_duration_s must be a positive finite number")
    return duration


def _snapshot(audio: Audio) -> Audio:
    snapshot = audio.copy()
    snapshot.setflags(write=False)
    return snapshot


def _failure(phase: str, tick_index: int | None, exc: Exception) -> EpisodeFailure:
    return EpisodeFailure(phase, tick_index, type(exc).__name__, str(exc))
