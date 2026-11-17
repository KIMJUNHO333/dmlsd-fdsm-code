from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .conversation import ConversationResult


SCHEMA_VERSION = 2
FORMAT_NAME = "dcl-simulation/conversation"


def write_conversation_result(
    destination: str | Path,
    result: ConversationResult,
    *,
    model_id: str,
    episode_id: str,
) -> Path:
    for name, value in (("model_id", model_id), ("episode_id", episode_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    import soundfile as sf

    directory = Path(destination).expanduser().absolute()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    try:
        episode = result.episode
        spec = episode.audio_spec
        audio = {}
        if spec is not None:
            for channel, rate, chunk_samples in (
                ("user", spec.input_sample_rate, spec.input_chunk_samples),
                ("model", spec.output_sample_rate, spec.output_chunk_samples),
            ):
                filename = f"{channel}.wav"
                with sf.SoundFile(directory / filename, mode="w", samplerate=rate,
                                  channels=1, format="WAV", subtype="FLOAT") as stream:
                    for tick in episode.ticks:
                        stream.write(tick.input_audio if channel == "user" else tick.output.audio)
                audio[channel] = {
                    "path": filename, "sample_rate": rate, "channels": 1,
                    "sample_count": len(episode.ticks) * chunk_samples,
                    "format": "WAV", "subtype": "FLOAT",
                }

        files = {
            "ticks": "ticks.jsonl", "events": "events.jsonl",
            "dialogue": "dialogue.jsonl", "user_utterances": "user_utterances.jsonl",
            "model_turns": "model_turns.jsonl",
            "speech_segments": "speech_segments.jsonl",
            "response_opportunities": "response_opportunities.jsonl",
        }
        _write_jsonl(directory / files["ticks"], (
            {
                "tick_index": tick.context.index,
                "time_start_s": float(tick.context.time_start),
                "time_end_s": float(tick.context.time_end),
                "input_sample_start": tick.context.index * spec.input_chunk_samples,
                "input_sample_end": (tick.context.index + 1) * spec.input_chunk_samples,
                "output_sample_start": tick.context.index * spec.output_chunk_samples,
                "output_sample_end": (tick.context.index + 1) * spec.output_chunk_samples,
                "native_text": tick.output.native_text,
                "generation_unit_id": tick.output.generation_unit_id,
                "measurements": (asdict(tick.output.measurements)
                                 if tick.output.measurements is not None else None),
            }
            for tick in episode.ticks
        ))
        for name, records in (
            ("events", result.events), ("dialogue", result.dialogue),
            ("user_utterances", result.utterances), ("model_turns", result.model_turns),
            ("speech_segments", result.speech_segments),
            ("response_opportunities", result.response_opportunities),
        ):
            _write_jsonl(directory / files[name], (asdict(record) for record in records))

        collapse = next((event for event in reversed(result.events)
                         if event.kind == "response_collapse"), None)
        manifest = {
            "format": FORMAT_NAME, "schema_version": SCHEMA_VERSION,
            "model_id": model_id, "episode_id": episode_id,
            "config": asdict(result.config),
            "provenance": asdict(result.provenance) if result.provenance is not None else None,
            "audio_spec": asdict(spec) if spec is not None else None,
            "summary": {
                "status": episode.status,
                "termination_reason": result.termination_reason,
                "stop_reason": episode.stop_reason,
                "requested_duration_s": float(episode.requested_duration),
                "logical_duration_s": float(episode.logical_duration),
                "planned_tick_count": episode.planned_tick_count,
                "recorded_tick_count": len(episode.ticks),
                "model_calls": episode.model_calls,
                "no_response_count": sum(event.kind == "no_response" for event in result.events),
                "collapse_onset_time_s": collapse.details["onset_s"] if collapse else None,
                "collapse_decision_time_s": collapse.time_s if collapse else None,
                "failures": [asdict(failure) for failure in episode.failures],
            },
            "audio": audio, "files": files,
        }
        temporary_manifest = directory / ".manifest.json.tmp"
        temporary_manifest.write_text(_json(manifest, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.rename(directory / "manifest.json")
    except BaseException as original:
        try:
            shutil.rmtree(directory)
        except OSError as cleanup:
            raise original from cleanup
        raise
    return directory


def _json(value, **kwargs):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs)


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(_json(record) + "\n")
