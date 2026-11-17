import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from dcl_simulation import (
    ConversationConfig, EvaluationProvenance, run_conversation, write_conversation_result,
)
from dcl_simulation.adapters import (
    ChatCompletionLLM, LLMConfig, KokoroConfig, KokoroTTS,
    SileroConfig, SileroVAD, ParakeetConfig, ParakeetASR,
)
from dcl_simulation.duplex_client import DuplexClient, DuplexClientConfig
from dcl_simulation._paths import ensure_new_output_path


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplex-client-config", type=Path, default=CONFIGS / "duplex_client.json")
    parser.add_argument("--config", type=Path, default=CONFIGS / "conversation.json")
    for role in ("llm", "tts", "vad", "asr"):
        parser.add_argument(f"--{role}-config", type=Path, default=CONFIGS / "services" / f"{role}.json")
    parser.add_argument("--output", type=Path, default=Path("results/episode-001"))
    parser.add_argument("--episode-id", default="episode-001")
    args = parser.parse_args()
    try:
        args.output = ensure_new_output_path(args.output)
    except OSError as exc:
        parser.error(str(exc))
    config = ConversationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    with ExitStack() as stack:
        llm = stack.enter_context(ChatCompletionLLM(LLMConfig.from_file(args.llm_config)))
        llm.connect()
        tts = KokoroTTS(KokoroConfig.from_file(args.tts_config))
        stack.callback(tts.close)
        asr = ParakeetASR(ParakeetConfig.from_file(args.asr_config))
        stack.callback(asr.close)
        vad = SileroVAD(SileroConfig.from_file(args.vad_config))
        model = stack.enter_context(DuplexClient(DuplexClientConfig.from_file(args.duplex_client_config)))
        provenance = EvaluationProvenance(
            model=model.provenance, llm=llm.provenance, tts=tts.provenance,
            vad=vad.provenance, asr=asr.provenance,
            vad_timestamp_resolution_s=vad.timestamp_resolution_s,
        )
        result = run_conversation(model, config, llm=llm, tts=tts, vad=vad, asr=asr,
                                  provenance=provenance)
    path = write_conversation_result(
        args.output, result, model_id=provenance.model.name, episode_id=args.episode_id,
    )
    print(path)
    if result.episode.status != "completed":
        raise RuntimeError(result.episode.failures)


if __name__ == "__main__":
    main()
