import argparse
import os
from pathlib import Path
import tarfile

from dcl_simulation.adapters import (
    KokoroConfig, ParakeetConfig, ECAPAConfig, MoshiConfig,
    PersonaPlexConfig, MiniCPMOConfig, RaonConfig,
)
from dcl_simulation.adapters._common import model_path


MODELS = {
    "moshi": (MoshiConfig, "kyutai/moshiko-pytorch-bf16", [
        "model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model",
    ]),
    "personaplex": (PersonaPlexConfig, "nvidia/personaplex-7b-v1", [
        "model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tokenizer_spm_32k_3.model", "voices.tgz",
    ]),
    "minicpmo": (MiniCPMOConfig, "openbmb/MiniCPM-o-4_5", None),
    "raon": (RaonConfig, "KRAFTON/Raon-SpeechChat-9B", None),
    "tts": (KokoroConfig, "hexgrad/Kokoro-82M", ["config.json", "kokoro-v1_0.pth", "voices/*.pt"]),
    "asr": (ParakeetConfig, "nvidia/parakeet-unified-en-0.6b", ["parakeet-unified-en-0.6b.nemo"]),
    "speaker": (ECAPAConfig, "speechbrain/spkrec-ecapa-voxceleb", ["*.yaml", "*.ckpt", "*.txt"]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=tuple(MODELS), required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cls, repo_id, patterns = MODELS[args.service]
    group = "duplex" if args.service in ("moshi", "personaplex", "minicpmo", "raon") else "evaluation" if args.service == "speaker" else "services"
    source = args.config or Path(__file__).resolve().parents[1] / "configs" / group / f"{args.service}.json"
    config = cls.from_file(source)
    if not config.revision:
        parser.error("Set the revision to download in the config")
    target = model_path(config.model_path)
    if target.suffix == ".nemo":
        parser.error("Set model_path to a directory for downloads")
    root = Path(os.environ.get("MODEL_ROOT", "~/model")).expanduser()
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, revision=config.revision, local_dir=str(target), allow_patterns=patterns)
    if args.service == "personaplex":
        voice = model_path(config.voice_prompt_path)
        if not voice.exists():
            with tarfile.open(target / "voices.tgz") as archive:
                member = archive.getmember(f"voices/{voice.name}")
                if not member.isfile():
                    raise ValueError("voice prompt archive member must be a file")
                with archive.extractfile(member) as stream:
                    content = stream.read()
            voice.parent.mkdir(parents=True, exist_ok=True)
            with voice.open("xb") as stream:
                stream.write(content)
    print(target)


if __name__ == "__main__":
    main()
