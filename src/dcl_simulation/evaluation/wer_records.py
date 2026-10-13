from __future__ import annotations

from dataclasses import dataclass


MIN_WER_REFERENCE_WORDS = 25


@dataclass(frozen=True)
class WERWord:
    text: str
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True)
class WERSentence:
    sentence_id: str
    block_id: str
    bin_index: int
    reference: str
    hypothesis: str
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    audio_start_time_s: float
    audio_end_time_s: float


@dataclass(frozen=True)
class WERBlock:
    block_id: str
    segment_ids: tuple[int, ...]
    start_time_s: float
    end_time_s: float
    reference: str
    hypothesis: str
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    sentences: tuple[WERSentence, ...]
    original_reference_words: int = 0
    trimmed_native_words: int = 0
    native_speech_coverage: float | None = None
    merged_native_word_boundaries: int = 0


@dataclass(frozen=True)
class WERMinute:
    bin_index: int
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    sentence_count: int
    model_speech_duration_s: float
    status: str
    wer: float | None
