import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/services/llm.json")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if not devices or devices == ["-1"]:
        parser.error("Select GPUs with CUDA_VISIBLE_DEVICES")
    if args.model_path is not None and not args.model_path.expanduser().is_dir():
        parser.error("--model-path is not an existing directory")
    command = [
        "vllm", "serve", str(args.model_path.expanduser()) if args.model_path else config["model"],
        "--served-model-name", config["model"],
        "--tensor-parallel-size", str(len(devices)),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--reasoning-parser", "qwen3",
        "--language-model-only",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.model_path is None and config.get("revision"):
        command += ["--revision", config["revision"]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
