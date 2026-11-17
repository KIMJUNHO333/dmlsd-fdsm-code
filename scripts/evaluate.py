from pathlib import Path

from dcl_simulation.evaluation.__main__ import main


if __name__ == "__main__":
    main(config_dir=Path(__file__).resolve().parents[1] / "configs" / "evaluation")
