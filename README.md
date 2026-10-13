# DCL Simulation

Long-session simulation and evaluation for full-duplex speech models.
Supports interaction metrics, WER, speaker identity, and conversation quality judging.

[Project page](https://kimjunho333.github.io/dmlsd-fdsm/)

## Quick start

Run all commands from the project root. The default setup uses Linux, NVIDIA GPUs,
Python 3.12, and English conversations.

1. Set the [GPU assignment](docs/setup.md#gpu-assignment), then prepare the [simulation environment](docs/setup.md#environment), LLM, TTS, VAD, and ASR.
2. Choose a duplex model below and start its server in a separate environment.
3. Run the conversation from the simulation terminal.

| Duplex model | Installation and server | Configuration |
| --- | --- | --- |
| Moshi · Moshiko 7B | [Moshi](docs/duplex.md#moshi) | [moshi.json](configs/duplex/moshi.json) |
| MiniCPM-o 4.5 | [MiniCPM-o](docs/duplex.md#minicpm-o) | [minicpmo.json](configs/duplex/minicpmo.json) |
| RAON SpeechChat 9B | [RAON](docs/duplex.md#raon) | [raon.json](configs/duplex/raon.json) |
| PersonaPlex 7B | [PersonaPlex](docs/duplex.md#personaplex) | [personaplex.json](configs/duplex/personaplex.json) |

```bash
source .venv/bin/activate
bash scripts/simulate.sh
```

Set the persona and topic in [conversation.json](configs/conversation.json), and the duplex
server address in [duplex_client.json](configs/duplex_client.json).
`MODEL_ROOT` sets the model storage directory; its default is `~/model`.

Results are saved to `results/episode-001`. Use a new output path for each run.

```bash
bash scripts/simulate.sh --output results/episode-002 --episode-id episode-002
```

## Evaluation

```bash
bash scripts/evaluate.sh results/episode-001 --metric interaction --output results/interaction.json
bash scripts/evaluate.sh results/episode-001 --metric wer --output results/wer.json
```

[Evaluation guide](docs/evaluation.md): speaker and judge setup, episode comparisons, and result formats.

## Code layout

```text
src/dcl_simulation/   simulation, interfaces, adapters, and evaluation
scripts/             model preparation, servers, simulation, and evaluation commands
configs/
  duplex/            model configurations
  services/          LLM, TTS, VAD, and ASR configurations
  evaluation/        evaluator configurations and episode comparisons
examples/            custom adapter skeleton and LLM connection example
docs/                setup, usage, and extension guides
```

Start with [simulate.py](scripts/simulate.py) for component setup and
[conversation.py](src/dcl_simulation/conversation.py) for the conversation policy.
[episode.py](src/dcl_simulation/episode.py) runs ticks;
[model.py](src/dcl_simulation/model.py) and [services.py](src/dcl_simulation/services.py)
define the duplex and speech service interfaces.
Model implementations are in [adapters](src/dcl_simulation/adapters), and metrics are in
[evaluation](src/dcl_simulation/evaluation).

Connect another model: [Custom adapters](docs/extensions.md).

## License

The simulation code is licensed under the [MIT License](LICENSE).
External models and SDKs retain their respective licenses.
