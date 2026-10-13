import argparse
from contextlib import ExitStack
from pathlib import Path

from .._paths import ensure_new_output_path
from . import (
    BootstrapConfig, EvaluationInputError, evaluate_dataset, evaluate_interaction,
    evaluate_judge, evaluate_judge_dataset, evaluate_speaker, evaluate_speaker_dataset,
    load_episode, write_evaluation,
)
from .input import _json
from .output import _serialize_evaluation
from .wer import WERConfig, evaluate_wer
from .wer_aggregate import evaluate_wer_dataset


def _comparison(path, metric="interaction", backend=None, *, backend_factory=None):
    config_path = Path(path).expanduser().resolve()
    try:
        config = _json(config_path.read_text(encoding="utf-8"))
        fields = {"wer"} if metric == "wer" else {"minimum_fto_events"} if metric == "interaction" else {"minimum_episodes"}
        allowed = {"episodes", "bootstrap", "condition_id"} | fields
        if not isinstance(config, dict) or not set(config) <= allowed:
            raise ValueError("Unsupported comparison config fields")
        paths = config["episodes"]
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p.strip() for p in paths):
            raise ValueError("episodes must be a nonempty list of paths")
        options = dict(bootstrap=BootstrapConfig(**config.get("bootstrap", {})),
                       condition_id=config.get("condition_id", "long_session"))
        episodes = []
        for value in paths:
            source = Path(value).expanduser()
            episodes.append(load_episode(source if source.is_absolute() else config_path.parent / source))
        if metric == "wer":
            wer = config.get("wer", {})
            if not isinstance(wer, dict) or not set(wer) <= {"profiles", "minimum_episodes"}:
                raise ValueError("Unsupported WER comparison config")
            return evaluate_wer_dataset(episodes, profiles=wer.get("profiles"),
                                        minimum_episodes=wer.get("minimum_episodes", 10), **options)
        if metric == "interaction":
            return evaluate_dataset(episodes, minimum_fto_events=config.get("minimum_fto_events", 20), **options)
        if backend_factory is not None:
            backend = backend_factory()
        evaluate = evaluate_speaker_dataset if metric == "speaker" else evaluate_judge_dataset
        return evaluate(episodes, backend, evaluator=backend.provenance,
                        minimum_episodes=config.get("minimum_episodes", 10), **options)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{config_path}: {exc}") from exc


def _backend(metric, path):
    from ..adapters import ECAPAConfig, ECAPAEmbedder, GeminiConfig, GeminiJudge
    if metric == "speaker":
        return ECAPAEmbedder(ECAPAConfig.from_file(path))
    return GeminiJudge(GeminiConfig.from_file(path))


def main(argv=None, *, config_dir=None):
    parser = argparse.ArgumentParser(description="Evaluate interaction, WER, speaker identity, or judge quality for saved episodes.")
    parser.add_argument("episode", nargs="?", help="Directory containing one simulation episode")
    parser.add_argument("--comparison", "--config", dest="comparison", type=Path, help="Comparison config JSON listing episodes")
    parser.add_argument("--output", type=Path, help="New JSON output path; omit to write to stdout")
    parser.add_argument("--metric", choices=("interaction", "wer", "speaker", "judge"), default="interaction")
    parser.add_argument("--wer-profile", choices=("standard", "minicpmo"))
    parser.add_argument("--backend-config", type=Path, help="Speaker or judge backend config JSON")
    args = parser.parse_args(argv)
    quality = args.metric in ("speaker", "judge")
    if config_dir is not None:
        config_dir = Path(config_dir)
        if args.episode is None and args.comparison is None:
            args.comparison = config_dir / ("quality.json" if quality else f"{args.metric}.json")
        if quality and args.backend_config is None:
            args.backend_config = config_dir / f"{args.metric}.json"
    if (args.episode is None) == (args.comparison is None):
        parser.error("Specify an episode path or --comparison, not both")
    if args.comparison and args.wer_profile is not None:
        parser.error("Set per-model comparison profiles in wer.profiles in the config JSON")
    if args.wer_profile is not None and args.metric != "wer":
        parser.error("--wer-profile requires --metric wer")
    if quality and args.backend_config is None:
        parser.error("Speaker and judge evaluation require --backend-config")
    if not quality and args.backend_config is not None:
        parser.error("--backend-config requires --metric speaker or --metric judge")
    try:
        if args.output is not None:
            args.output = ensure_new_output_path(args.output)
        with ExitStack() as stack:
            def open_backend():
                backend = _backend(args.metric, args.backend_config)
                stack.callback(backend.close)
                return backend

            if args.comparison:
                result = _comparison(args.comparison, args.metric,
                                     backend_factory=open_backend if quality else None)
            else:
                episode = load_episode(args.episode)
                if args.metric == "interaction":
                    result = evaluate_interaction(episode)
                elif args.metric == "wer":
                    result = evaluate_wer(episode, config=WERConfig(args.wer_profile or "standard"))
                else:
                    backend = open_backend()
                    evaluate = evaluate_speaker if args.metric == "speaker" else evaluate_judge
                    result = evaluate(episode, backend, evaluator=backend.provenance)
        if args.output:
            print(write_evaluation(args.output, result))
        else:
            print(_serialize_evaluation(result), end="")
    except (EvaluationInputError, OSError, ValueError) as exc:
        parser.exit(2, f"Evaluation input error: {exc}\n")
    except RuntimeError as exc:
        parser.exit(1, f"Evaluation runtime error: {exc}\n")


if __name__ == "__main__":
    main()
