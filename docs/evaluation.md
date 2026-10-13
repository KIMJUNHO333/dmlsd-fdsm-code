# Evaluation

Activate the simulation environment from the project root. Use a new output file for each evaluation.

```bash
source .venv/bin/activate
```

## Single episode

```bash
bash scripts/evaluate.sh results/episode-001 --metric interaction --output results/interaction.json
bash scripts/evaluate.sh results/episode-001 --metric wer --output results/wer.json
bash scripts/evaluate.sh results/episode-001 --metric speaker --output results/speaker.json
bash scripts/evaluate.sh results/episode-001 --metric judge --output results/judge.json
```

Speaker and judge evaluation require the models below.
WER requires native text for each tick and ASR word timestamps; add `--wer-profile minicpmo` for MiniCPM-o.

## Prepare evaluation models

| Metric | Default model | Configuration | Preparation |
| --- | --- | --- | --- |
| Speaker | ECAPA-TDNN · CPU | [speaker.json](../configs/evaluation/speaker.json) | Download command below |
| Judge | Gemini 3.8 Flash | [judge.json](../configs/evaluation/judge.json) | [API key](https://aistudio.google.com/apikey) |

```bash
python scripts/prepare_models.py --service speaker
export GEMINI_API_KEY="your-api-key"
```

For existing ECAPA weights, set `model_path` to the directory containing `hyperparams.yaml`, `*.ckpt`, and `label_encoder.txt`.
Use `--backend-config` to select another evaluator config.

## Compare episodes

Set `episodes` in the comparison config to your result paths. Relative paths are resolved from the config directory.

| Metric | Default comparison config |
| --- | --- |
| Interaction | [interaction.json](../configs/evaluation/interaction.json) |
| WER | [wer.json](../configs/evaluation/wer.json) · `wer.profiles` keys are model IDs from the results |
| Speaker and Judge | [quality.json](../configs/evaluation/quality.json) |

```bash
bash scripts/evaluate.sh --metric interaction --output results/interaction-comparison.json
bash scripts/evaluate.sh --metric wer --output results/wer-comparison.json
bash scripts/evaluate.sh --metric speaker --output results/speaker-comparison.json
bash scripts/evaluate.sh --metric judge --output results/judge-comparison.json
```

Omitting the episode path uses the default comparison configs above. Use `--comparison` to select another config.
Compared models must share episode IDs, scenarios, conversation policies, and LLM/TTS/VAD/ASR settings.
Include failed episodes in the lists. Results record execution settings in `provenance`.

## Metrics and aggregation

| Metric | Calculation |
| --- | --- |
| Interaction | Speech duration, FTO, backchannels and interruptions, no-response rate, cumulative collapse |
| WER | ASR word error rate against native text; at least 25 reference words per minute |
| Speaker identity | Mean cosine similarity to the first minute's speaker centroid; at least 10 seconds of enrollment audio and 5 seconds of evaluation audio |
| Judge | Content quality scored from 1 to 5 per minute using the full dialogue; at least 30 seconds observed and some model speech |

WER, speaker, and judge comparisons require at least 10 episodes by default; FTO requires 20 normal responses.
95% confidence intervals use episode bootstrap.
Scores are `null` when samples or inputs are insufficient. Speaker and judge inference failures are recorded
as `status: evaluator_error` with a `reason`; comparison results include `coverage`.
Configuration, input, model setup, and missing dependency errors are reported as command failures.

## Read results

Each episode directory contains `manifest.json`, user/model WAV files, and JSONL files for dialogue, ticks, events, and speech segments.
Evaluation JSON contains metrics, statuses, and comparison conditions.

```python
from dcl_simulation.evaluation import load_episode, evaluate_interaction, write_evaluation

episode = load_episode("results/episode-001")
write_evaluation("results/metrics.json", evaluate_interaction(episode))
```

Connect another evaluator: [Custom adapters](extensions.md#evaluation-models).
