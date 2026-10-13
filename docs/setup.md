# Simulation setup

Run all commands from the project root. Use separate environments and terminals
for the simulation, duplex model, and LLM server.

## GPU assignment

Before starting servers or the simulation, select physical GPU IDs with each script's `CUDA_VISIBLE_DEVICES`.

| Process | Script | Default GPUs |
| --- | --- | --- |
| LLM server | [serve_llm.sh](../scripts/serve_llm.sh) | `0,1` |
| Duplex server | [serve_duplex.sh](../scripts/serve_duplex.sh) | `0` |
| Simulation ASR | [simulate.sh](../scripts/simulate.sh) | `0` |

You can also override the value in each terminal.

```bash
export CUDA_VISIBLE_DEVICES=0
```

`cuda:0` in a config refers to the first selected GPU. LLM tensor parallelism uses the number of selected GPUs.
When sharing a GPU, account for the combined model memory. Default TTS and VAD run on CPU.

## Environment

```bash
python3.12 -m venv .venv --prompt dcl-simulation
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[llm,tts,vad,asr,ecapa,judge,wer,models]"
python -m spacy download en_core_web_sm
```

## 1. Connect an LLM

Start a server using [LLM setup](llm.md). For an existing chat-completions server,
follow [Client configuration](llm.md#client-configuration).
Once the server is ready, activate `.venv` in the simulation terminal.

## 2. Prepare speech models

```bash
python scripts/prepare_models.py --service tts
python scripts/prepare_models.py --service asr
```

| Role | Default model | Configuration |
| --- | --- | --- |
| LLM | Qwen3.8-27B · chat-completions API | [llm.json](../configs/services/llm.json) |
| TTS | Kokoro-82M | [tts.json](../configs/services/tts.json) |
| VAD | Silero | [vad.json](../configs/services/vad.json) |
| ASR | Parakeet Unified En 0.6B | [asr.json](../configs/services/asr.json) |

Silero weights are included in the package. See [Model paths](duplex.md#model-paths) for storage settings.
For existing weights, set `model_path` instead of downloading again.
The TTS directory must contain `config.json`, `kokoro-v1_0.pth`, and `voices/<voice>.pt`.
ASR accepts a `.nemo` file or its directory.
Kokoro supports American English voices (`a` prefix) and British English voices (`b` prefix).

## 3. Configure the conversation

Edit the persona and topic in [conversation.json](../configs/conversation.json).
The default duration is 600 seconds. To change it, add `"target_duration_s": 120`.
See [ConversationConfig](../src/dcl_simulation/conversation.py) for all options and defaults.

## 4. Run

Once the [duplex server](duplex.md) and LLM server are ready, run from the simulation terminal.

```bash
source .venv/bin/activate
bash scripts/simulate.sh
```

| Setting | Configuration or argument |
| --- | --- |
| Server address and response timeout | [duplex_client.json](../configs/duplex_client.json) · `--duplex-client-config` |
| Conversation policy | [conversation.json](../configs/conversation.json) · `--config` |
| LLM and speech services | `--llm-config`, `--tts-config`, `--vad-config`, `--asr-config` |
| Output path and episode ID | `--output`, `--episode-id` |

Use a new output directory. Next: [Evaluation](evaluation.md) · [Custom adapters](extensions.md).
