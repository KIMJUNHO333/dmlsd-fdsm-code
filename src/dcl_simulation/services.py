from dataclasses import dataclass
from typing import Literal, Protocol

from .audio import Audio
from ._prompts import SYSTEM_PROMPT


@dataclass(frozen=True)
class DialogueEntry:
    speaker: Literal["user", "model"]
    text: str
    start_s: float
    end_s: float
    kind: Literal["normal", "backchannel", "interrupting"] = "normal"
    truncated: bool = False
    user_utterance_id: int | None = None
    model_turn_id: int | None = None


@dataclass(frozen=True)
class UtteranceRequest:
    persona: str
    topic: str
    dialogue: tuple[DialogueEntry, ...]
    goal: str
    attempt: int = 1


def utterance_messages(request: UtteranceRequest) -> list[dict[str, str]]:
    dialogue = []
    for entry in request.dialogue:
        label = "User" if entry.speaker == "user" else "Model"
        marker = f" ({entry.kind})" if entry.kind != "normal" else ""
        text = entry.text.rstrip()
        if entry.truncated and not text.endswith(("—", "-", "–")):
            text += "—"
        dialogue.append(f"{label}{marker}: {text}")
    history = "\n".join(dialogue) or "(empty)"
    action = "normal_response" if request.attempt == 1 else "retry_response"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"<persona>\n{request.persona.strip()}\n</persona>\n"
            f"<conversation_concept>\n{request.topic.strip()}\n</conversation_concept>\n\n"
            f"<dialogue_history>\n{history}\n</dialogue_history>\n\n"
            "<generation_request>\n"
            f"action_type: {action}\ncontent_goal: {request.goal.strip()}\n"
            "</generation_request>"
        )},
    ]


class UtteranceGenerator(Protocol):
    def generate(self, request: UtteranceRequest) -> str:
        ...


@dataclass(frozen=True)
class WordTiming:

    char_start: int
    char_end: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class SynthesizedSpeech:
    audio: Audio
    sample_rate: int
    words: tuple[WordTiming, ...] = ()


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str, *, sample_rate: int) -> SynthesizedSpeech:
        ...


@dataclass(frozen=True)
class SpeechEvent:
    kind: Literal["start", "end"]
    time_s: float


@dataclass(frozen=True)
class SpeechSpan:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class SpeechUpdate:

    events: tuple[SpeechEvent, ...] = ()
    activity: tuple[SpeechSpan, ...] = ()


class SpeechDetectionSession(Protocol):
    def process(self, audio: Audio) -> SpeechUpdate:
        ...

    def finalize(self) -> SpeechUpdate:
        ...

    def close(self) -> None:
        ...


class SpeechDetector(Protocol):
    def open_session(self, *, sample_rate: int) -> SpeechDetectionSession:
        ...


@dataclass(frozen=True)
class RecognizedWord:

    text: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class Transcription:

    text: str
    words: tuple[RecognizedWord, ...] | None = None


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: Audio, *, sample_rate: int) -> Transcription | str:
        ...
