from dataclasses import asdict
import json
from pathlib import Path

from .aggregate import DatasetEvaluation
from .interaction import EpisodeEvaluation
from .judge import EpisodeJudge
from .quality_aggregate import QualityDatasetEvaluation
from .speaker import EpisodeSpeakerIdentity
from .wer import EpisodeWER
from .wer_aggregate import WERDatasetEvaluation


def _serialize_evaluation(result):
    formats = {EpisodeEvaluation: 'interaction', DatasetEvaluation: 'interaction-comparison',
               EpisodeWER: 'wer', WERDatasetEvaluation: 'wer-comparison',
               EpisodeSpeakerIdentity: 'speaker', EpisodeJudge: 'judge'}
    if isinstance(result, QualityDatasetEvaluation):
        name = 'speaker-comparison' if result.metric == 'speaker_similarity' else 'judge-comparison'
    else:
        try:
            name = formats[type(result)]
        except KeyError as exc:
            raise TypeError('Unsupported evaluation result format') from exc
    return json.dumps({'format': f'dcl-simulation/{name}', 'schema_version': 1, **asdict(result)},
                      ensure_ascii=False, allow_nan=False, indent=2) + '\n'


def write_evaluation(path: str | Path, result) -> Path:
    text = _serialize_evaluation(result)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(text)
    return destination
