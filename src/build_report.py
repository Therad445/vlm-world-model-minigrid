from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .utils import ensure_parent, load_config


METHOD_NAMES = {
    "random": "Random",
    "wm": "WM planning (no VLM)",
    "wm_vlm": "WM planning + VLM",
}


def _add_image(fig, path: str | Path, box: list[float]) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    image = mpimg.imread(path)
    inset = fig.add_axes(box)
    inset.imshow(image)
    inset.axis("off")
    return True


def _metric(metrics: pd.DataFrame, method: str, column: str) -> float:
    return float(metrics.loc[metrics["method"] == method, column].iloc[0])


def build(config: dict) -> Path:
    metrics_path = Path("outputs/metrics.csv")
    episodes_path = Path("outputs/episodes.csv")
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Run evaluation before building the report: "
            "python -m src.evaluate --methods random wm wm_vlm"
        )

    metrics = pd.read_csv(metrics_path)
    episodes = pd.read_csv(episodes_path) if episodes_path.exists() else None
    output = ensure_parent("report/report.pdf")

    planner = config["planner"]
    vlm = config["vlm"]
    eval_cfg = config["eval"]
    scored_steps = list(
        range(int(planner["vlm_stride"]), int(planner["horizon"]) + 1, int(planner["vlm_stride"]))
    )
    if int(planner["horizon"]) not in scored_steps:
        scored_steps.append(int(planner["horizon"]))

    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "World-model MPC with a VLM scorer", fontsize=20, weight="bold")
        fig.text(0.08, 0.905, "MiniGrid-Empty-6x6-v0 - measured experimental report", fontsize=12)
        intro = (
            "Goal. Test whether a compact Dreamer-style RSSM can generate imagined future frames, "
            "and whether a frozen vision-language model can be used as a semantic objective inside "
            "model-predictive control.\n\n"
            "World model. CNN encoder, GRU deterministic state, diagonal-Gaussian stochastic latent "
            "state, RGB decoder, reward head and continuation head.\n\n"
            f"Planning. Random-shooting MPC compares {planner['candidates']} candidate action sequences "
            f"over horizon H={planner['horizon']} and executes the first action. The reward-only "
            "baseline uses predicted discounted reward. The VLM variant adds the maximum CLIP goal "
            f"score over decoded imagined future frames at steps {', '.join(map(str, scored_steps))}.\n\n"
            f"Text goal. {vlm['goal']}\n\n"
            "Run. Google Colab Tesla T4; 240 collected episodes, 153 successful; RSSM trained for "
            f"{config['train']['epochs']} epochs; evaluation uses 25 episodes per method, with seeds "
            f"{eval_cfg['seeds']} and five episodes per seed.\n\n"
            "The important protocol point is that the CLIP/VLM score is applied to decoded future "
            "frames from learned rollouts, not only to the current real observation."
        )
        fig.text(0.08, 0.84, intro, fontsize=10.5, va="top", wrap=True, linespacing=1.45)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Quantitative comparison", fontsize=18, loc="left", pad=25)
        display = metrics.copy()
        display["method"] = display["method"].map(METHOD_NAMES).fillna(display["method"])
        for column in ["success_rate", "mean_return"]:
            display[column] = display[column].map(lambda value: f"{value:.3f}")
        display["mean_episode_length"] = display["mean_episode_length"].map(lambda value: f"{value:.2f}")
        table = ax.table(
            cellText=display.values,
            colLabels=["Method", "Episodes", "Success rate", "Mean return", "Mean episode length"],
            loc="upper center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        _add_image(fig, "outputs/plots/success_rate.png", [0.16, 0.35, 0.68, 0.35])
        wm = _metric(metrics, "wm", "success_rate")
        random = _metric(metrics, "random", "success_rate")
        wm_vlm = _metric(metrics, "wm_vlm", "success_rate")
        note = (
            f"Interpretation. Reward-only world-model planning is strongest in this run: {wm:.2f} "
            f"success rate versus {random:.2f} for random and {wm_vlm:.2f} for WM+VLM. The VLM path "
            "is a valid implementation, but it is not a performance gain here. The result suggests "
            "that CLIP scores on symbolic MiniGrid reconstructions are noisy and can conflict with "
            "the learned reward objective.\n\n"
            "The aggregate table is loaded from outputs/metrics.csv."
        )
        if episodes is not None:
            note += f" Per-episode seeds, returns and lengths are stored in outputs/episodes.csv ({len(episodes)} rows)."
        fig.text(0.08, 0.16, note, fontsize=10, va="top", wrap=True, linespacing=1.35)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Training and visual checks", fontsize=18, weight="bold")
        train = pd.read_csv("outputs/training_loss.csv")
        train_note = (
            "The reconstruction check matters because the VLM scores decoded model predictions. "
            "If the decoder does not preserve objects, the semantic score becomes unreliable. "
            "In this run the RSSM learned the room layout and the green goal clearly, while the "
            "red agent is less stable in the reconstruction.\n\n"
            f"Training loss decreased from {train['loss'].iloc[0]:.3f} to {train['loss'].iloc[-1]:.3f} "
            f"over {len(train)} epochs."
        )
        fig.text(0.08, 0.88, train_note, fontsize=10.5, va="top", wrap=True, linespacing=1.4)
        _add_image(fig, "outputs/screenshots/reconstruction.png", [0.10, 0.52, 0.80, 0.28])
        _add_image(fig, "outputs/plots/training_loss.png", [0.16, 0.11, 0.68, 0.32])
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Imagined rollouts and failure modes", fontsize=18, weight="bold")
        text = (
            "The figure below shows decoded future states from an imagined rollout and their CLIP\n"
            "goal scores. The green goal stays visually clear, while the agent signal is weak.\n"
            "This makes it plausible that CLIP rewards the presence of the green square\n"
            "more than the exact relation 'agent standing on goal'.\n\n"
            "Observed failure modes.\n"
            "- CLIP can reward green pixels instead of the spatial relation agent-on-goal.\n"
            "- RSSM rollout errors accumulate; the red agent becomes blurry or disappears in imagined frames.\n"
            "- The VLM is applied on decoded predictions, so decoder artefacts directly affect the objective.\n"
            "- Random shooting with 32 candidates is brittle when the objective is noisy.\n\n"
            "Future work.\n"
            "- Replace random shooting with CEM.\n"
            "- Fine-tune a small contrastive MiniGrid scorer on relation labels.\n"
            "- Add uncertainty penalties to imagined states.\n"
            "- Compare against a symbolic oracle scorer for diagnosis only.\n"
            "- Try an object-centric world model or a full Dreamer actor-critic agent."
        )
        fig.text(0.08, 0.88, text, fontsize=10.5, va="top", linespacing=1.35)
        _add_image(fig, "outputs/screenshots/imagined_rollout.png", [0.08, 0.06, 0.84, 0.24])
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Reproducibility notes", fontsize=18, weight="bold")
        text = (
            "Command used for the measured run:\n"
            "python run_pipeline.py --methods random wm wm_vlm\n\n"
            "The repository contains source code, configuration, measured CSV files, plots,\n"
            "screenshots, a GIF visualization, the PDF report and the full run log. Large\n"
            "regenerable files such as the collected dataset and RSSM checkpoint are not\n"
            "committed; they are produced again by running the pipeline.\n\n"
            "Key output files.\n"
            "- outputs/metrics.csv\n"
            "- outputs/episodes.csv\n"
            "- outputs/training_loss.csv\n"
            "- outputs/screenshots/reconstruction.png\n"
            "- outputs/screenshots/imagined_rollout.png\n"
            "- outputs/gifs/wm_vlm_episode.gif\n"
            "- logs/full_run.log"
        )
        fig.text(0.08, 0.88, text, fontsize=11, va="top", linespacing=1.45)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved report to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_empty.yaml")
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
