from dataclasses import dataclass
from importlib.metadata import version
import math
from typing import Protocol

import numpy as np
import soundfile as sf

from ..records import ComponentInfo
from .input import EvaluationEpisode

SPEAKER_SAMPLE_RATE = 16_000


class SpeakerEmbedder(Protocol):
    def embed(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        ...


class SpeakerDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerMinute:
    bin_index: int
    status: str
    similarity: float | None
    valid_speech_duration_s: float
    segment_count: int
    overlap_segment_count: int


@dataclass(frozen=True)
class EpisodeSpeakerIdentity:
    model_id: str
    episode_id: str
    evaluator: ComponentInfo
    status: str
    reason: str | None
    dependencies: dict[str, str]
    enrollment_status: str
    enrollment_speech_duration_s: float | None
    enrollment_segment_count: int | None
    minutes: tuple[SpeakerMinute, ...]


def _normalize(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.all(np.isfinite(vector)):
        raise ValueError('Speaker embedding must be a finite one-dimensional vector')
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid speaker embedding norm')
    return vector / norm


def _chunks(episode):
    result = {}
    for segment in episode.speech_segments:
        start = segment.start_s
        while start < segment.end_s - 1e-9:
            index = int(start // 60)
            end = min(segment.end_s, (index + 1) * 60.0)
            cursor = start
            spans = []
            while end - cursor >= 3.0 - 1e-9:
                spans.append((cursor, cursor + 3.0))
                cursor += 3.0
            if end - cursor >= 2.0 - 1e-9:
                spans.append((cursor, end))
            for left, right in spans:
                overlap = any(min(right, u.end_s) > max(left, u.start_s)
                              for u in episode.user_utterances if u.delivered_sample_count)
                result.setdefault(index, []).append((left, right, overlap))
            start = end
    return result


def evaluate_speaker(episode: EvaluationEpisode, embedder: SpeakerEmbedder, *,
                     evaluator: ComponentInfo) -> EpisodeSpeakerIdentity:
    if not isinstance(evaluator, ComponentInfo):
        raise TypeError('evaluator must be ComponentInfo')
    try:
        return _evaluate_speaker(episode, embedder, evaluator=evaluator)
    except SpeakerDependencyError:
        raise
    except Exception as exc:
        return EpisodeSpeakerIdentity(episode.model_id, episode.episode_id, evaluator,
            'evaluator_error', f'{type(exc).__name__}: {exc}', {}, 'not_evaluated', None, None, ())


def _evaluate_speaker(episode, embedder, *, evaluator):
    if episode.summary.status == 'failed':
        return EpisodeSpeakerIdentity(episode.model_id, episode.episode_id, evaluator,
            'excluded', 'simulation_failed', {}, 'not_evaluated', None, None, ())
    chunks = _chunks(episode)
    enrollment = chunks.get(0, ())
    duration = sum(end - start for start, end, _ in enrollment)
    dependencies = {'numpy': version('numpy'), 'soundfile': version('soundfile')}
    count = math.ceil(episode.config.target_duration_s / 60)
    centroid, embeddings = None, {}
    if duration >= 10.0:
        audio, rate = sf.read(episode.directory / 'model.wav', dtype='float32', always_2d=False)
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise ValueError('Speaker audio must be a finite mono array')
        if rate != SPEAKER_SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise SpeakerDependencyError('Resampling dependencies are required: python -m pip install -e ".[speaker]"') from exc
            dependencies['scipy'] = version('scipy')
            divisor = math.gcd(rate, SPEAKER_SAMPLE_RATE)
            audio = resample_poly(audio, SPEAKER_SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
        dimension = None
        for index in range(count):
            pieces = chunks.get(index, ())
            if index > 0 and sum(end - start for start, end, _ in pieces) < 5.0:
                continue
            vectors = []
            for start, end, _ in pieces:
                piece = np.ascontiguousarray(audio[max(0, round(start * SPEAKER_SAMPLE_RATE)):
                                                  min(audio.size, round(end * SPEAKER_SAMPLE_RATE))])
                if not piece.size:
                    raise ValueError('Speaker segment audio is empty')
                vector = _normalize(embedder.embed(piece, sample_rate=SPEAKER_SAMPLE_RATE))
                dimension = vector.size if dimension is None else dimension
                if vector.size != dimension:
                    raise ValueError('Speaker embedding dimensions differ between calls')
                vectors.append(vector)
            embeddings[index] = vectors
        centroid = _normalize(np.mean(embeddings[0], axis=0))
    minutes = []
    for index in range(1, count):
        pieces = chunks.get(index, ())
        seconds = sum(end - start for start, end, _ in pieces)
        score = None
        if centroid is None:
            status = 'insufficient_reference_audio'
        elif seconds < 5.0:
            status = 'insufficient_speaker_audio'
        else:
            status = 'ok'
            score = float(np.mean([float(np.dot(centroid, _normalize(v))) for v in embeddings[index]]))
        minutes.append(SpeakerMinute(index, status, score, seconds, len(pieces),
                                     sum(overlap for _, _, overlap in pieces)))
    return EpisodeSpeakerIdentity(episode.model_id, episode.episode_id, evaluator, 'ok', None,
        dependencies, 'ok' if centroid is not None else 'insufficient_reference_audio',
        duration, len(enrollment), tuple(minutes))
