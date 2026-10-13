from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

from .wer_records import MIN_WER_REFERENCE_WORDS, WERBlock, WERMinute, WERSentence, WERWord

_SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+")


_NON_ASCII_APOSTROPHE = r"[\u2018\u2019\u02bc\uff07]"


_CLEAR_CONTRACTION = re.compile(
    rf"\b(?:"
    rf"i{_NON_ASCII_APOSTROPHE}(?:m|ve|ll)"
    rf"|(?:you|we|they){_NON_ASCII_APOSTROPHE}(?:re|ve|ll)"
    rf"|(?:he|she|it|that|what|who){_NON_ASCII_APOSTROPHE}(?:s|ll)"
    rf"|(?:where|here|there|let){_NON_ASCII_APOSTROPHE}s"
    rf"|(?:could|would|should|might|must){_NON_ASCII_APOSTROPHE}ve"
    rf"|[a-z]+n{_NON_ASCII_APOSTROPHE}t"
    rf")\b",
    re.IGNORECASE,
)


_APOSTROPHE_TRANSLATION = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\uff07": "'"}
)


_MIN_TRIMMED_SUFFIX_WORDS = 5


_MIN_TRIMMED_SUFFIX_FRACTION = 0.10


@dataclass
class _BlockSource:
    segment_ids: list[int]
    start_time_s: float
    end_time_s: float
    tick_indices: set[int]


@dataclass(frozen=True)
class _ReferenceSentence:
    text: str
    words: tuple[str, ...]
    native_time_s: float


def _reference_sentences(
    ticks: Sequence[Mapping[str, object]],
    indices: Sequence[int],
    normalizer,
    *,
    fallback_time_s: float,
) -> tuple[_ReferenceSentence, ...]:
    text = ""
    character_times: list[float] = []
    for index in indices:
        tick = ticks[index]
        delta = tick.get("native_text_delta")
        if not isinstance(delta, str) or not delta:
            continue
        start = _optional_number(tick.get("time_start_s"))
        end = _optional_number(tick.get("time_end_s"))
        midpoint = (
            (start + end) / 2
            if start is not None and end is not None
            else fallback_time_s
        )
        text += delta
        character_times.extend([midpoint] * len(delta))
    if not text.strip():
        return ()
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.start() + 1))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    result: list[_ReferenceSentence] = []
    for start, end in spans:
        original = text[start:end].strip()
        normalized = normalizer(original).strip()
        if not normalized:
            continue
        times = character_times[start:end]
        native_time = (
            sum(times) / len(times) if times else fallback_time_s
        )
        result.append(
            _ReferenceSentence(
                text=original,
                words=tuple(normalized.split()),
                native_time_s=native_time,
            )
        )
    return tuple(result)


def _score_block(
    *,
    block_id: str,
    source: _BlockSource,
    reference_sentences: tuple[_ReferenceSentence, ...],
    hypothesis_words: tuple[WERWord, ...],
    normalizer,
    process_words,
    character_distance,
    apply_minicpmo_suffix_correction: bool,
) -> WERBlock:
    normalized_hyp, normalized_hyp_times = _normalize_hypothesis_words(
        hypothesis_words,
        normalizer,
    )
    original_reference_words = sum(
        len(sentence.words) for sentence in reference_sentences
    )
    trimmed_native_words = 0
    if apply_minicpmo_suffix_correction:
        keep_words = _spoken_prefix_word_count(
            [word for sentence in reference_sentences for word in sentence.words],
            normalized_hyp,
            character_distance,
        )
        trimmed_native_words = original_reference_words - keep_words
        if trimmed_native_words:
            reference_sentences = _reference_prefix(
                reference_sentences,
                keep_words=keep_words,
            )
    ref_words = [word for sentence in reference_sentences for word in sentence.words]
    ref_sentence_ids = [
        index
        for index, sentence in enumerate(reference_sentences)
        for _ in sentence.words
    ]
    (
        ref_words,
        ref_sentence_ids,
        normalized_hyp,
        normalized_hyp_times,
        merged_native_word_boundaries,
    ) = _reconcile_exact_word_boundaries(
        ref_words,
        ref_sentence_ids,
        normalized_hyp,
        normalized_hyp_times,
        process_words,
    )
    reference = " ".join(ref_words)
    hypothesis = " ".join(normalized_hyp)
    if not reference_sentences:
        insertions = len(normalized_hyp)
        sentences: tuple[WERSentence, ...] = ()
        if normalized_hyp:
            audio_start = normalized_hyp_times[0][0]
            audio_end = normalized_hyp_times[-1][1]
            sentences = (
                WERSentence(
                    sentence_id=f"{block_id}:sentence:0000",
                    block_id=block_id,
                    bin_index=max(0, int(((audio_start + audio_end) / 2) // 60)),
                    reference="",
                    hypothesis=hypothesis,
                    substitutions=0,
                    deletions=0,
                    insertions=insertions,
                    reference_words=0,
                    audio_start_time_s=audio_start,
                    audio_end_time_s=audio_end,
                ),
            )
        return WERBlock(
            block_id=block_id,
            segment_ids=tuple(source.segment_ids),
            start_time_s=source.start_time_s,
            end_time_s=source.end_time_s,
            reference=reference,
            hypothesis=hypothesis,
            substitutions=0,
            deletions=0,
            insertions=insertions,
            reference_words=0,
            sentences=sentences,
            original_reference_words=original_reference_words,
            trimmed_native_words=trimmed_native_words,
            native_speech_coverage=(
                (original_reference_words - trimmed_native_words)
                / original_reference_words
                if original_reference_words
                else None
            ),
            merged_native_word_boundaries=merged_native_word_boundaries,
        )
    alignment = process_words(reference, hypothesis)
    sentence_words: list[list[str]] = [[] for _ in reference_sentences]
    for word, sentence_id in zip(ref_words, ref_sentence_ids):
        sentence_words[sentence_id].append(word)
    counts = [
        {"s": 0, "d": 0, "i": 0, "hyp": [], "times": []}
        for _ in reference_sentences
    ]
    chunks = alignment.alignments[0]
    for chunk in chunks:
        if chunk.type in {"equal", "substitute"}:
            for ref_index, hyp_index in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                sentence_id = ref_sentence_ids[ref_index]
                if chunk.type == "substitute":
                    counts[sentence_id]["s"] += 1
                counts[sentence_id]["hyp"].append(normalized_hyp[hyp_index])
                counts[sentence_id]["times"].append(
                    normalized_hyp_times[hyp_index]
                )
        elif chunk.type == "delete":
            for ref_index in range(chunk.ref_start_idx, chunk.ref_end_idx):
                counts[ref_sentence_ids[ref_index]]["d"] += 1
        elif chunk.type == "insert":
            for hyp_index, sentence_id in _insertion_assignments(
                ref_sentence_ids,
                ref_index=chunk.ref_start_idx,
                hyp_start=chunk.hyp_start_idx,
                hyp_end=chunk.hyp_end_idx,
                hypothesis_times=normalized_hyp_times,
            ):
                counts[sentence_id]["i"] += 1
                counts[sentence_id]["hyp"].append(normalized_hyp[hyp_index])
                counts[sentence_id]["times"].append(
                    normalized_hyp_times[hyp_index]
                )

    sentence_rows: list[WERSentence] = []
    for index, sentence in enumerate(reference_sentences):
        row = counts[index]
        times = row["times"]
        if times:
            audio_start = min(item[0] for item in times)
            audio_end = max(item[1] for item in times)
        else:
            audio_start = audio_end = sentence.native_time_s
        midpoint = (audio_start + audio_end) / 2
        sentence_rows.append(
            WERSentence(
                sentence_id=f"{block_id}:sentence:{index:04d}",
                block_id=block_id,
                bin_index=max(0, int(midpoint // 60)),
                reference=" ".join(sentence_words[index]),
                hypothesis=" ".join(row["hyp"]),
                substitutions=int(row["s"]),
                deletions=int(row["d"]),
                insertions=int(row["i"]),
                reference_words=len(sentence_words[index]),
                audio_start_time_s=audio_start,
                audio_end_time_s=audio_end,
            )
        )
    if sum(row.substitutions for row in sentence_rows) != alignment.substitutions:
        raise RuntimeError("Sentence substitution totals do not match the block alignment")
    if sum(row.deletions for row in sentence_rows) != alignment.deletions:
        raise RuntimeError("Sentence deletion totals do not match the block alignment")
    if sum(row.insertions for row in sentence_rows) != alignment.insertions:
        raise RuntimeError("Sentence insertion totals do not match the block alignment")
    if sum(row.reference_words for row in sentence_rows) != len(ref_words):
        raise RuntimeError("Sentence reference word totals do not match the block alignment")
    return WERBlock(
        block_id=block_id,
        segment_ids=tuple(source.segment_ids),
        start_time_s=source.start_time_s,
        end_time_s=source.end_time_s,
        reference=reference,
        hypothesis=hypothesis,
        substitutions=alignment.substitutions,
        deletions=alignment.deletions,
        insertions=alignment.insertions,
        reference_words=alignment.hits + alignment.substitutions + alignment.deletions,
        sentences=tuple(sentence_rows),
        original_reference_words=original_reference_words,
        trimmed_native_words=trimmed_native_words,
        native_speech_coverage=(
            (original_reference_words - trimmed_native_words)
            / original_reference_words
            if original_reference_words
            else None
        ),
        merged_native_word_boundaries=merged_native_word_boundaries,
    )


def _spoken_prefix_word_count(
    reference_words: Sequence[str],
    hypothesis_words: Sequence[str],
    character_distance,
) -> int:

    if not reference_words or not hypothesis_words:
        return len(reference_words)
    hypothesis = "".join(hypothesis_words)
    prefix = ""
    best_word_count = len(reference_words)
    best_key: tuple[int, int] | None = None
    for word_count, word in enumerate(reference_words, start=1):
        prefix += word
        key = (character_distance(prefix, hypothesis), -word_count)
        if best_key is None or key < best_key:
            best_key = key
            best_word_count = word_count
    trimmed = len(reference_words) - best_word_count
    if trimmed < _MIN_TRIMMED_SUFFIX_WORDS:
        return len(reference_words)
    if trimmed / len(reference_words) < _MIN_TRIMMED_SUFFIX_FRACTION:
        return len(reference_words)
    return best_word_count


def _reference_prefix(
    sentences: Sequence[_ReferenceSentence],
    *,
    keep_words: int,
) -> tuple[_ReferenceSentence, ...]:
    result: list[_ReferenceSentence] = []
    remaining = keep_words
    for sentence in sentences:
        if remaining <= 0:
            break
        words = sentence.words[:remaining]
        if words:
            result.append(
                _ReferenceSentence(
                    text=" ".join(words),
                    words=words,
                    native_time_s=sentence.native_time_s,
                )
            )
        remaining -= len(words)
    if remaining != 0:
        raise RuntimeError("Native reference prefix exceeds the original word count")
    return tuple(result)


def _reconcile_exact_word_boundaries(
    reference_words: Sequence[str],
    reference_sentence_ids: Sequence[int],
    hypothesis_words: Sequence[str],
    hypothesis_times: Sequence[tuple[float, float]],
    process_words,
) -> tuple[
    list[str],
    list[int],
    list[str],
    list[tuple[float, float]],
    int,
]:

    if not reference_words or not hypothesis_words:
        return (
            list(reference_words),
            list(reference_sentence_ids),
            list(hypothesis_words),
            list(hypothesis_times),
            0,
        )
    alignment = process_words(
        " ".join(reference_words),
        " ".join(hypothesis_words),
    )
    chunks = alignment.alignments[0]
    reconciled_reference: list[str] = []
    reconciled_sentence_ids: list[int] = []
    reconciled_hypothesis: list[str] = []
    reconciled_times: list[tuple[float, float]] = []
    merged_word_boundaries = 0
    chunk_index = 0
    while chunk_index < len(chunks):
        chunk = chunks[chunk_index]
        if chunk.type == "equal":
            reconciled_reference.extend(
                reference_words[chunk.ref_start_idx : chunk.ref_end_idx]
            )
            reconciled_sentence_ids.extend(
                reference_sentence_ids[chunk.ref_start_idx : chunk.ref_end_idx]
            )
            reconciled_hypothesis.extend(
                hypothesis_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
            )
            reconciled_times.extend(
                hypothesis_times[chunk.hyp_start_idx : chunk.hyp_end_idx]
            )
            chunk_index += 1
            continue

        run_end = chunk_index + 1
        while run_end < len(chunks) and chunks[run_end].type != "equal":
            run_end += 1
        run = chunks[chunk_index:run_end]
        ref_start = run[0].ref_start_idx
        ref_end = run[-1].ref_end_idx
        hyp_start = run[0].hyp_start_idx
        hyp_end = run[-1].hyp_end_idx
        ref_slice = list(reference_words[ref_start:ref_end])
        hyp_slice = list(hypothesis_words[hyp_start:hyp_end])
        sentence_slice = list(reference_sentence_ids[ref_start:ref_end])
        time_slice = list(hypothesis_times[hyp_start:hyp_end])
        boundary_only = (
            bool(ref_slice)
            and bool(hyp_slice)
            and len(set(sentence_slice)) == 1
            and "".join(ref_slice) == "".join(hyp_slice)
        )
        if boundary_only:
            token = "".join(ref_slice)
            reconciled_reference.append(token)
            reconciled_sentence_ids.append(sentence_slice[0])
            reconciled_hypothesis.append(token)
            reconciled_times.append(
                (time_slice[0][0], time_slice[-1][1])
            )
            merged_word_boundaries += max(len(ref_slice), len(hyp_slice)) - 1
        else:
            reconciled_reference.extend(ref_slice)
            reconciled_sentence_ids.extend(sentence_slice)
            reconciled_hypothesis.extend(hyp_slice)
            reconciled_times.extend(time_slice)
        chunk_index = run_end
    return (
        reconciled_reference,
        reconciled_sentence_ids,
        reconciled_hypothesis,
        reconciled_times,
        merged_word_boundaries,
    )


def _normalize_hypothesis_words(
    words: tuple[WERWord, ...], normalizer
) -> tuple[list[str], list[tuple[float, float]]]:
    normalized = normalizer(" ".join(word.text for word in words)).split()
    if not normalized or not words:
        return normalized, []
    if len(normalized) == 1:
        return normalized, [(words[0].start_time_s, words[-1].end_time_s)]
    timestamps: list[tuple[float, float]] = []
    for index in range(len(normalized)):
        raw_index = min(
            int(((index + 0.5) / len(normalized)) * len(words)),
            len(words) - 1,
        )
        word = words[raw_index]
        timestamps.append((word.start_time_s, word.end_time_s))
    return normalized, timestamps


def _insertion_assignments(
    sentence_ids: Sequence[int],
    *,
    ref_index: int,
    hyp_start: int,
    hyp_end: int,
    hypothesis_times: Sequence[tuple[float, float]],
) -> list[tuple[int, int]]:
    if not sentence_ids:
        return []
    before = sentence_ids[ref_index - 1] if ref_index > 0 else None
    after = sentence_ids[ref_index] if ref_index < len(sentence_ids) else None
    if before is None:
        assert after is not None
        return [(index, after) for index in range(hyp_start, hyp_end)]
    if after is None or before == after:
        return [(index, before) for index in range(hyp_start, hyp_end)]
    previous_anchor = hyp_start - 1
    next_anchor = hyp_end
    if previous_anchor < 0:
        return [(index, after) for index in range(hyp_start, hyp_end)]
    if next_anchor >= len(hypothesis_times):
        return [(index, before) for index in range(hyp_start, hyp_end)]
    boundary = (
        hypothesis_times[previous_anchor][1]
        + hypothesis_times[next_anchor][0]
    ) / 2
    return [
        (
            index,
            before
            if sum(hypothesis_times[index]) / 2 <= boundary
            else after,
        )
        for index in range(hyp_start, hyp_end)
    ]


def _minute_rows(
    blocks: tuple[WERBlock, ...], *, target_duration_s: float
) -> tuple[WERMinute, ...]:
    minutes: list[WERMinute] = []
    for bin_index in range(math.ceil(target_duration_s / 60)):
        sentences = [
            sentence
            for block in blocks
            for sentence in block.sentences
            if sentence.bin_index == bin_index
        ]
        s = sum(item.substitutions for item in sentences)
        d = sum(item.deletions for item in sentences)
        i = sum(item.insertions for item in sentences)
        n = sum(item.reference_words for item in sentences)
        speech = sum(
            max(
                0.0,
                min(block.end_time_s, (bin_index + 1) * 60)
                - max(block.start_time_s, bin_index * 60),
            )
            for block in blocks
        )
        sufficient = n >= MIN_WER_REFERENCE_WORDS
        minutes.append(
            WERMinute(
                bin_index=bin_index,
                substitutions=s,
                deletions=d,
                insertions=i,
                reference_words=n,
                sentence_count=len(sentences),
                model_speech_duration_s=speech,
                status="ok" if sufficient else "insufficient_speech",
                wer=(s + d + i) / n if sufficient else None,
            )
        )
    return tuple(minutes)


def _normalize_clear_contraction_apostrophes(text: str) -> str:

    return _CLEAR_CONTRACTION.sub(
        lambda match: match.group(0).translate(_APOSTROPHE_TRANSLATION),
        text,
    )


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None
