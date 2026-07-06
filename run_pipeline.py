"""Run the measured MiniGrid world-model/VLM experiment.

Examples
--------
Full run:
    python run_pipeline.py

Cheap sanity check without downloading CLIP:
    python run_pipeline.py --fast --methods random wm

Cheap end-to-end check with the VLM path enabled:
    python run_pipeline.py --fast --methods random wm wm_vlm
"""

from __future__ import annotations

import argparse
import copy

from src.build_report import build
from src.collect_data import collect
from src.evaluate import evaluate
from src.train_world_model import train
from src.utils import load_config


def apply_fast_overrides(config: dict) -> dict:
    """Make the pipeline cheap enough for a local smoke test."""
    cfg = copy.deepcopy(config)
    cfg["data"]["episodes"] = 12
    cfg["train"]["epochs"] = 1
    cfg["train"]["batch_size"] = min(8, int(cfg["train"]["batch_size"]))
    cfg["planner"]["horizon"] = 4
    cfg["planner"]["candidates"] = 4
    cfg["planner"]["vlm_stride"] = 2
    cfg["eval"]["seeds"] = [0]
    cfg["eval"]["episodes_per_seed"] = 1
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_empty.yaml")
    parser.add_argument("--fast", action="store_true", help="Use tiny settings for smoke testing.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["random", "wm", "wm_vlm"],
        default=["random", "wm", "wm_vlm"],
        help="Evaluation methods. Use '--methods random wm' to avoid loading CLIP.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fast:
        config = apply_fast_overrides(config)

    collect(config)
    train(config)
    evaluate(config, args.methods)
    build(config)


if __name__ == "__main__":
    main()
