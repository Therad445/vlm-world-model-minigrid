from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .envs import make_env
from .planner import RandomShootingPlanner
from .rssm import model_from_config
from .utils import ensure_parent, load_config, obs_to_tensor, resolve_device, seed_everything
from .vlm_scorer import CLIPGoalScorer


def load_model(config: dict, device: torch.device):
    checkpoint = torch.load(config["train"]["checkpoint"], map_location=device, weights_only=False)
    model = model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def save_rollout_sheet(debug: dict) -> None:
    frames = debug["frames"]
    scores = debug["vlm_scores"]
    frame_steps = debug["frame_steps"]
    if frames is None:
        return
    indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
    figure, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
    axes = np.atleast_1d(axes)
    for axis, index in zip(axes, indices):
        axis.imshow(frames[index].permute(1, 2, 0).numpy())
        axis.set_title(f"t+{frame_steps[index]}\nCLIP={float(scores[index]):.3f}")
        axis.axis("off")
    figure.suptitle("Decoded imagined future frames scored by the VLM")
    figure.tight_layout()
    figure.savefig(ensure_parent("outputs/screenshots/imagined_rollout.png"), dpi=180)
    plt.close(figure)


def evaluate(config: dict, methods: list[str], episodes_override: int | None = None) -> pd.DataFrame:
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    need_model = any(method != "random" for method in methods)
    model = load_model(config, device) if need_model else None
    scorer = None
    if "wm_vlm" in methods:
        vlm = config["vlm"]
        scorer = CLIPGoalScorer(vlm["model_name"], vlm["goal"], vlm["negatives"], device)
    planners = {}
    if "wm" in methods:
        planners["wm"] = RandomShootingPlanner(model, config["planner"])
    if "wm_vlm" in methods:
        planners["wm_vlm"] = RandomShootingPlanner(model, config["planner"], scorer)

    seeds = list(map(int, config["eval"]["seeds"]))
    repetitions = int(config["eval"]["episodes_per_seed"])
    episode_seeds = [seed * 10_000 + repeat for seed in seeds for repeat in range(repetitions)]
    if episodes_override is not None:
        episode_seeds = episode_seeds[:episodes_override]
    rows, gif_frames, saved_debug = [], [], False
    rng = np.random.default_rng(int(config["seed"]))
    for method in methods:
        env = make_env(config)
        for episode_index, episode_seed in enumerate(tqdm(episode_seeds, desc=method)):
            obs, _ = env.reset(seed=episode_seed)
            total_reward, success = 0.0, False
            frames_this_episode = [obs]
            for step in range(int(config["eval"]["max_steps"])):
                debug = None
                if method == "random":
                    action = int(rng.integers(0, int(config["env"]["action_dim"])))
                else:
                    action, debug = planners[method].plan(
                        obs_to_tensor(obs, device), use_vlm=(method == "wm_vlm"),
                        return_debug=(method == "wm_vlm" and not saved_debug),
                    )
                if debug is not None and not saved_debug:
                    save_rollout_sheet(debug); saved_debug = True
                obs, reward, terminated, truncated, _ = env.step(action)
                frames_this_episode.append(obs)
                total_reward += float(reward)
                success = success or reward > 0
                if terminated or truncated:
                    break
            if method == "wm_vlm" and (success or not gif_frames):
                gif_frames = frames_this_episode
            rows.append({
                "method": method, "episode": episode_index, "seed": episode_seed,
                "success": int(success), "return": total_reward, "length": step + 1,
            })
        env.close()

    details = pd.DataFrame(rows)
    ensure_parent("outputs/episodes.csv")
    details.to_csv("outputs/episodes.csv", index=False)
    summary = details.groupby("method", as_index=False).agg(
        episodes=("episode", "count"), success_rate=("success", "mean"),
        mean_return=("return", "mean"), mean_episode_length=("length", "mean"),
    )
    summary.to_csv("outputs/metrics.csv", index=False)
    plot_metrics(summary)
    if gif_frames:
        gif_path = ensure_parent("outputs/gifs/wm_vlm_episode.gif")
        imageio.mimsave(gif_path, gif_frames, duration=0.16, loop=0)
    print(summary.to_string(index=False))
    return summary


def plot_metrics(summary: pd.DataFrame) -> None:
    order = [method for method in ("random", "wm", "wm_vlm") if method in set(summary.method)]
    values = summary.set_index("method").loc[order, "success_rate"]
    labels = {"random": "Random", "wm": "WM only", "wm_vlm": "WM + VLM"}
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([labels[value] for value in values.index], values.values, color=["#888888", "#4C78A8", "#59A14F"][:len(values)])
    axis.set_ylim(0, 1); axis.set_ylabel("Success rate"); axis.set_title("MiniGrid evaluation")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(ensure_parent("outputs/plots/success_rate.png"), dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_empty.yaml")
    parser.add_argument("--methods", nargs="+", choices=["random", "wm", "wm_vlm"], default=["random", "wm", "wm_vlm"])
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.methods, args.episodes)


if __name__ == "__main__":
    main()
