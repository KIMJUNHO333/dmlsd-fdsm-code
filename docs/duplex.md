# Duplex models

Create a separate environment for each model at the project root and run its server there.
The simulation and speech services use a [separate environment](setup.md#environment).

## Model paths

`MODEL_ROOT` defaults to `~/model`. To use another directory, set it in each terminal
used for downloads, servers, and simulation.

```bash
export MODEL_ROOT=/path/to/models
```

Download commands use the configured `revision`.
For existing weights, set `model_path` to their directory instead of downloading again.
Relative paths are resolved from the config file directory; `${MODEL_ROOT}` and `~` are supported.
Update `revision` when using a different model version.

## Moshi

```bash
python3.12 -m venv .venv-moshi --prompt dcl-moshi
source .venv-moshi/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[moshi,models]"
python scripts/prepare_models.py --service moshi
bash scripts/serve_duplex.sh --config configs/duplex/moshi.json
```

Set `device`, `dtype`, and audio/text temperature and top-k in [moshi.json](../configs/duplex/moshi.json).
The adapter uses the original Moshiko 7B checkpoint with 24 kHz input/output and 80 ms ticks.
Required files for existing weights:

```text
${MODEL_ROOT}/duplex/moshiko/repo/
├── model.safetensors
├── tokenizer-e351c8d8-checkpoint125.safetensors
└── tokenizer_spm_32k_3.model
```

Next: [Conversation setup and execution](setup.md).

## MiniCPM-o

```bash
python3.12 -m venv .venv-minicpmo --prompt dcl-minicpmo
source .venv-minicpmo/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[minicpmo,models]"
python scripts/prepare_models.py --service minicpmo
bash scripts/serve_duplex.sh --config configs/duplex/minicpmo.json
```

Set `prefix_system_prompt` and generation options in [minicpmo.json](../configs/duplex/minicpmo.json).
The default reference voice is [minicpmo_ref_af_heart.wav](../assets/audio/minicpmo_ref_af_heart.wav).
See [reference audio](../assets/audio/README.md) for its source.
Set `reference_audio` to a WAV path to use another voice.
Input is 16 kHz, output is 24 kHz, and each tick is one second.
Use the `minicpmo` profile for WER evaluation.

Next: [Conversation setup and execution](setup.md).

## RAON

```bash
python3.12 -m venv .venv-raon --prompt dcl-raon
source .venv-raon/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[raon,models]"
python scripts/prepare_models.py --service raon
bash scripts/serve_duplex.sh --config configs/duplex/raon.json
```

Set `system_prompt`, `speak_first`, and generation options in [raon.json](../configs/duplex/raon.json).
For a reference speaker, set `speaker_audio` to a WAV path.
The SDK downloads ECAPA weights on first use of a reference speaker.
`serve_duplex.sh` stores them under `${MODEL_ROOT}/duplex/raon-speechchat-9b/speechbrain` by default.
To use another cache location, set it in the server terminal.

```bash
export SPEECHBRAIN_ECAPA_SAVEDIR=/path/to/speechbrain-cache
```

Audio specifications are read from the model processor.
The default `max_sequence_length` is 23612, covering a 10-minute conversation and a 15-second termination grace period.
For longer conversations, round the total duration up to a full tick, then to a whole second.
Set the capacity to at least `ceil(37.5 × total_seconds) + 512`.
A 20-minute conversation with a 15-second grace period requires 46112.

Next: [Conversation setup and execution](setup.md).

## PersonaPlex

Run PersonaPlex in a dedicated environment. Install the SDK before switching to the CUDA 12.8 PyTorch build.

Obtain access to the weights on the [model page](https://huggingface.co/nvidia/personaplex-7b-v1).

```bash
python3.12 -m venv .venv-personaplex --prompt dcl-personaplex
source .venv-personaplex/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchaudio==2.4.1
python -m pip install -e ".[personaplex,models]"
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

The final command reports a conflict with the SDK's declared `torch<2.5` dependency.

If needed, authenticate with `huggingface-cli login`, then download the weights.

```bash
python scripts/prepare_models.py --service personaplex
bash scripts/serve_duplex.sh --config configs/duplex/personaplex.json
```

Set `text_prompt` and `voice_prompt_path` in [personaplex.json](../configs/duplex/personaplex.json).
The download command saves the selected `.pt` voice to that path.
The default voice is `NATF2.pt`, with 24 kHz input/output and 80 ms ticks.

Next: [Conversation setup and execution](setup.md).

## Server connection

Select model GPUs with `CUDA_VISIBLE_DEVICES` in [serve_duplex.sh](../scripts/serve_duplex.sh).
`cuda:0` in the model config refers to the first selected GPU.
The default address is `127.0.0.1:8765`. Start the simulation in another terminal once the server is ready.

To change ports, set both the server's `--port` and `port` in [duplex_client.json](../configs/duplex_client.json).

```bash
bash scripts/serve_duplex.sh --config configs/duplex/moshi.json --port 8766
```

Each server handles one client at a time. A connected client can run episodes sequentially;
session state is cleared between episodes while model weights remain loaded.
Disconnects and communication errors close the active session. Model errors close the connection
and are returned to the client. Stop the server with `Ctrl+C`.

`timeout_s` in `duplex_client.json` controls the response timeout.
On timeout, the client closes its connection. An ongoing inference is allowed to return before
session cleanup; restart the model server if inference itself is stuck.
The server's `--idle-timeout-s` limits time between client requests and defaults to 600 seconds.
Increase it if LLM, TTS, or ASR processing requires more time.
Use trusted local connections; the server provides no authentication or encryption.

Connect another model: [Custom adapters](extensions.md#duplex).
