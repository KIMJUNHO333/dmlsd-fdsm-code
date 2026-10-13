# LLM setup

The default model in [configs/services/llm.json](../configs/services/llm.json) is `Qwen/Qwen3.8-27B`.
For an existing chat-completions server, start with [Client configuration](#client-configuration).

## Server setup

Create the server environment at the project root.

```bash
python3.12 -m venv .venv-llm --prompt dcl-llm
source .venv-llm/bin/activate
python -m pip install vllm==0.25.1 transformers==5.14.1 huggingface-hub==1.24.0 accelerate==1.14.0
```

Set `CUDA_VISIBLE_DEVICES` in [serve_llm.sh](../scripts/serve_llm.sh) using [GPU assignment](setup.md#gpu-assignment).
The default is `0,1`. Weights are downloaded on first use to `~/model/huggingface`.
Set `MODEL_ROOT` or `HF_HOME` to use another storage location.

```bash
bash scripts/serve_llm.sh
```

For existing weights, specify the snapshot directory.

```bash
bash scripts/serve_llm.sh --model-path /path/to/model
```

Tensor parallelism uses the number of selected GPUs. Adjust context length and memory usage with
`--max-model-len` and `--gpu-memory-utilization`.
When changing ports, update both `--port` and the client config's `endpoint`.

## Client configuration

Activate the [simulation environment](setup.md#environment) in another terminal.

```bash
source .venv/bin/activate
```

Set `endpoint`, `model`, `revision`, and `decoding` in [configs/services/llm.json](../configs/services/llm.json).
`model` is the name served by the API; `revision` identifies the model version.
For other servers, adapt `top_k`, `repetition_penalty`, and `chat_template_kwargs` to the supported API options.
For authentication, add `"api_key_env": "LLM_API_KEY"` to the config and set that environment variable.

Check the connection by generating one user utterance.

```bash
python examples/llm/connect.py
```

Use `--config` to select another config file.

Next: [Prepare speech models](setup.md#2-prepare-speech-models).
