# Custom adapters

Set addresses, weights, voices, and generation options in `configs/`.
To connect another SDK, implement the interfaces below.

## Duplex

Copy [adapter.py](../examples/duplex/adapter.py) and [config.json](../examples/duplex/config.json).
Implement SDK and weight loading in `load_model`, and one inference tick in `infer_tick`.
Use `new_state` and `close_state` to create and release session state.

```bash
cp -r examples/duplex examples/my_duplex
```

Install the SDK and this package in the model environment, then start the server.

```bash
bash scripts/serve_duplex.sh --adapter-dir examples/my_duplex --config examples/my_duplex/config.json
```

The config's `adapter` is `module:factory`; the skeleton uses `adapter:create_model`.
`--adapter-dir` is the module directory; omit it for installed Python packages.
The factory receives the config without `adapter`, plus `config_dir`.
The simulation connects using [duplex_client.json](../configs/duplex_client.json).

| Component | Interface and examples |
| --- | --- |
| Model loading, session creation, and shutdown | [DuplexModel](../src/dcl_simulation/model.py) · `open_session()`, `close()`, `provenance` |
| Tick inference and session shutdown | [DuplexSession](../src/dcl_simulation/model.py) · `process_tick()`, `close()`, `audio_spec` |
| Audio specification | [AudioSpec](../src/dcl_simulation/audio.py) · mono `float32`, matching input/output tick durations |
| Model implementations | [duplex adapters](../src/dcl_simulation/adapters/duplex) |

## LLM, TTS, VAD, and ASR

For an existing chat-completions server, update [LLM client configuration](llm.md#client-configuration).
For other SDKs, implement the interfaces in [services.py](../src/dcl_simulation/services.py)
and connect them in the component setup in [simulate.py](../scripts/simulate.py).
The execution and result saving calls can be reused.

| Role | Method | Example |
| --- | --- | --- |
| LLM | `generate(request) -> str` | [llm.py](../src/dcl_simulation/adapters/services/llm.py) |
| TTS | `synthesize(text, *, sample_rate) -> SynthesizedSpeech` | [tts.py](../src/dcl_simulation/adapters/services/tts.py) |
| VAD | `open_session(*, sample_rate)` · session `process`, `finalize`, `close` | [vad.py](../src/dcl_simulation/adapters/services/vad.py) |
| ASR | `transcribe(audio, *, sample_rate) -> Transcription` | [asr.py](../src/dcl_simulation/adapters/services/asr.py) |

Each component provides `provenance: ComponentInfo`. Register `close` with `ExitStack` for components that need cleanup.
The default LLM uses a context manager and `connect()`; adapt its initialization and cleanup when replacing it.
VAD provides `timestamp_resolution_s`.

| Return value | Requirements |
| --- | --- |
| TTS | Mono `float32` at the requested sample rate; optional `words` give character spans and times in seconds |
| VAD | Times in seconds from episode start; `events` mark start/end, `activity` marks actual speech spans |
| ASR | Transcript and word spans in input audio samples; WER requires word spans |

## Evaluation models

[SpeakerEmbedder](../src/dcl_simulation/evaluation/speaker.py) returns a one-dimensional embedding from 16 kHz audio.
[ConversationJudge](../src/dcl_simulation/evaluation/judge.py) returns a JSON string matching the `JudgeTask` schema.

See [evaluation adapters](../src/dcl_simulation/adapters/evaluation) for existing implementations.
Connect a new implementation in the CLI's [_backend](../src/dcl_simulation/evaluation/__main__.py),
providing `provenance: ComponentInfo` and `close()`. Comparisons also require evaluator version information.
