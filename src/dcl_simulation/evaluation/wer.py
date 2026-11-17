from dataclasses import dataclass
from importlib.metadata import version

from .input import EvaluationEpisode
from .wer_records import WERBlock, WERMinute, WERWord
from ._wer_scoring import (
    _BlockSource, _minute_rows, _normalize_clear_contraction_apostrophes,
    _reference_sentences, _score_block,
)


class WERDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class WERConfig:
    profile: str = 'standard'

    def __post_init__(self):
        if self.profile not in ('standard', 'minicpmo'):
            raise ValueError('WER profile must be standard or minicpmo')


@dataclass(frozen=True)
class EpisodeWER:
    model_id: str
    episode_id: str
    config: WERConfig
    status: str
    reason: str | None
    dependencies: dict[str, str]
    blocks: tuple[WERBlock, ...]
    minutes: tuple[WERMinute, ...]
    unaligned_native_words: int | None
    original_reference_words: int | None
    trimmed_native_words: int | None
    native_speech_coverage: float | None
    merged_native_word_boundaries: int | None


def _dependencies():
    try:
        from jiwer import process_words
        from rapidfuzz.distance.Levenshtein import distance as character_distance
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError as exc:
        raise WERDependencyError('WER dependencies are required: python -m pip install -e ".[wer]"') from exc
    english_normalizer = EnglishTextNormalizer()
    def normalize(text):
        return english_normalizer(_normalize_clear_contraction_apostrophes(text))
    versions = {name: version(name) for name in ('jiwer', 'rapidfuzz', 'whisper-normalizer')}
    return normalize, process_words, character_distance, versions


def _alignment_blocks(episode, config):
    lookback = 3.0 if config.profile == 'minicpmo' else 1.0
    raw = []
    def units(indices):
        return {episode.ticks[i].generation_unit_id for i in indices
                if episode.ticks[i].generation_unit_id is not None}
    for segment in sorted(episode.speech_segments, key=lambda s: s.start_s):
        ticks = {i for i, tick in enumerate(episode.ticks) if tick.native_text
                 and tick.time_end_s >= segment.start_s - lookback
                 and tick.time_start_s <= segment.end_s + 1.0}
        if raw and (segment.start_s - raw[-1].end_time_s < 1.0
                    or units(ticks) & units(raw[-1].tick_indices)):
            raw[-1].segment_ids.append(segment.id)
            raw[-1].end_time_s = max(raw[-1].end_time_s, segment.end_s)
            raw[-1].tick_indices.update(ticks)
        else:
            raw.append(_BlockSource([segment.id], segment.start_s, segment.end_s, ticks))
    merged = []
    for block in raw:
        if merged and merged[-1].tick_indices & block.tick_indices:
            previous = merged[-1]
            previous.segment_ids.extend(block.segment_ids)
            previous.end_time_s = max(previous.end_time_s, block.end_time_s)
            previous.tick_indices.update(block.tick_indices)
        else:
            merged.append(block)
    return merged


def _hypothesis_words(episode, segment_ids):
    wanted = set(segment_ids)
    turns = sorted((turn for turn in episode.model_turns if wanted.intersection(turn.segment_ids)),
                   key=lambda t: t.audio_sample_start)
    words, seen = [], set()
    rate = episode.audio_spec.output_sample_rate
    for turn in turns:
        for word in turn.words:
            key = (word.text, word.start_sample, word.end_sample)
            if not word.text.strip() or key in seen:
                continue
            seen.add(key)
            words.append(WERWord(word.text, word.start_sample / rate, word.end_sample / rate))
    return tuple(sorted(words, key=lambda w: (w.start_time_s, w.end_time_s)))


def evaluate_wer(episode: EvaluationEpisode, *, config: WERConfig = WERConfig()) -> EpisodeWER:
    if not isinstance(config, WERConfig):
        raise TypeError('config must be WERConfig')
    def skipped(status, reason):
        return EpisodeWER(episode.model_id, episode.episode_id, config, status, reason,
                          {}, (), (), None, None, None, None, None)
    if episode.summary.status == 'failed':
        return skipped('excluded', 'simulation_failed')
    if any(tick.native_text is None for tick in episode.ticks):
        return skipped('unavailable', 'native_text_missing')
    if any(turn.words is None for turn in episode.model_turns):
        return skipped('unavailable', 'asr_word_timestamps_missing')
    normalizer, process_words, character_distance, versions = _dependencies()
    blocks, assigned = [], set()
    ticks = tuple({'native_text_delta': tick.native_text, 'time_start_s': tick.time_start_s,
                   'time_end_s': tick.time_end_s} for tick in episode.ticks)
    for index, source in enumerate(_alignment_blocks(episode, config)):
        indices = tuple(sorted(source.tick_indices - assigned))
        assigned.update(indices)
        sentences = _reference_sentences(ticks, indices, normalizer,
                                         fallback_time_s=(source.start_time_s + source.end_time_s) / 2)
        blocks.append(_score_block(
            block_id=f'{episode.episode_id}:wer-block:{index:06d}', source=source,
            reference_sentences=sentences, hypothesis_words=_hypothesis_words(episode, source.segment_ids),
            normalizer=normalizer, process_words=process_words, character_distance=character_distance,
            apply_minicpmo_suffix_correction=config.profile == 'minicpmo',
        ))
    unaligned = sum(len(normalizer(tick.native_text).split()) for i, tick in enumerate(episode.ticks)
                    if i not in assigned and tick.native_text.strip())
    original = sum(b.original_reference_words for b in blocks)
    trimmed = sum(b.trimmed_native_words for b in blocks)
    return EpisodeWER(
        episode.model_id, episode.episode_id, config, 'ok', None, versions, tuple(blocks),
        _minute_rows(tuple(blocks), target_duration_s=episode.config.target_duration_s),
        unaligned, original, trimmed, (original - trimmed) / original if original else None,
        sum(b.merged_native_word_boundaries for b in blocks),
    )
