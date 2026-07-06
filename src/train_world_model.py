from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import trange

from .rssm import gaussian_kl, model_from_config
from .utils import ensure_parent, load_config, resolve_device, seed_everything


class SequenceSampler:
    def __init__(self, data: dict[str, np.ndarray], sequence_length: int, rng: np.random.Generator):
        self.data = data
        self.sequence_length = sequence_length
        self.rng = rng
        self.eligible = np.flatnonzero(data["lengths"] >= sequence_length)
        if len(self.eligible) == 0:
            raise ValueError("No episode is long enough for train.sequence_length")

    def batch(self, batch_size: int, device: torch.device):
        episodes = self.rng.choice(self.eligible, batch_size, replace=True)
        starts = [self.rng.integers(0, int(self.data["lengths"][ep]) - self.sequence_length + 1) for ep in episodes]
        obs, actions, rewards, continues = [], [], [], []
        for episode, start in zip(episodes, starts):
            end = start + self.sequence_length
            obs.append(self.data["observations"][episode, start : end + 1])
            actions.append(self.data["actions"][episode, start:end])
            rewards.append(self.data["rewards"][episode, start:end])
            continues.append(self.data["continues"][episode, start:end])
        obs_t = torch.from_numpy(np.stack(obs)).permute(0, 1, 4, 2, 3).float().to(device) / 255.0
        return (
            obs_t,
            torch.from_numpy(np.stack(actions)).long().to(device),
            torch.from_numpy(np.stack(rewards)).float().to(device),
            torch.from_numpy(np.stack(continues)).float().to(device),
        )


def train(config: dict, epochs_override: int | None = None) -> Path:
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    raw = np.load(config["data"]["path"])
    data = {key: raw[key] for key in raw.files}
    settings = config["train"]
    batch_size = int(settings["batch_size"])
    sequence_length = int(settings["sequence_length"])
    epochs = epochs_override or int(settings["epochs"])
    steps_per_epoch = max(25, len(data["lengths"]) // batch_size)
    sampler = SequenceSampler(data, sequence_length, np.random.default_rng(int(config["seed"])))
    model = model_from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        progress = trange(steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for _ in progress:
            obs, actions, rewards, continues = sampler.batch(batch_size, device)
            state = model.infer(obs[:, 0], sample=True)
            recon_loss = F.binary_cross_entropy_with_logits(model.decode_logits(state), obs[:, 0])
            kl_loss = torch.zeros((), device=device)
            reward_loss = torch.zeros((), device=device)
            continue_loss = torch.zeros((), device=device)
            for step in range(sequence_length):
                state, prior = model.observe_step(state, actions[:, step], obs[:, step + 1], sample=True)
                recon_loss = recon_loss + F.binary_cross_entropy_with_logits(model.decode_logits(state), obs[:, step + 1])
                kl_loss = kl_loss + gaussian_kl(state, prior).clamp_min(float(settings["free_nats"])).mean()
                reward_loss = reward_loss + F.mse_loss(model.predict_reward(state), rewards[:, step])
                continue_loss = continue_loss + F.binary_cross_entropy_with_logits(model.predict_continue_logits(state), continues[:, step])
            denom = float(sequence_length)
            loss = (
                recon_loss / (sequence_length + 1)
                + float(settings["beta_kl"]) * kl_loss / denom
                + float(settings["reward_scale"]) * reward_loss / denom
                + float(settings["continue_scale"]) * continue_loss / denom
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip"]))
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            progress.set_postfix(loss=f"{epoch_losses[-1]:.3f}")
        history.append(np.mean(epoch_losses))
        print(f"epoch={epoch + 1} loss={history[-1]:.4f}")

    save_training_history(history)
    checkpoint = ensure_parent(settings["checkpoint"])
    torch.save({"model": model.state_dict(), "config": config, "loss": history}, checkpoint)
    save_reconstruction(model, data["observations"][0, 0], device)
    return checkpoint


def save_training_history(history: list[float]) -> None:
    frame = pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "loss": history})
    frame.to_csv(ensure_parent("outputs/training_loss.csv"), index=False)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(frame["epoch"], frame["loss"], marker="o")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_title("RSSM training curve")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ensure_parent("outputs/plots/training_loss.png"), dpi=160)
    plt.close(figure)


@torch.no_grad()
def save_reconstruction(model, frame: np.ndarray, device: torch.device) -> None:
    model.eval()
    target = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    reconstruction = model.decode(model.infer(target))[0].permute(1, 2, 0).cpu().numpy()
    output = ensure_parent("outputs/screenshots/reconstruction.png")
    figure, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(frame); axes[0].set_title("Real observation")
    axes[1].imshow(reconstruction); axes[1].set_title("RSSM reconstruction")
    for axis in axes: axis.axis("off")
    figure.tight_layout(); figure.savefig(output, dpi=160); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_empty.yaml")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    train(load_config(args.config), args.epochs)


if __name__ == "__main__":
    main()

