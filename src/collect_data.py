from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

from .envs import make_env, shortest_path_action
from .utils import ensure_parent, load_config, seed_everything


def collect(config: dict, episodes: int | None = None) -> Path:
    seed = int(config["seed"])
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    episodes = episodes or int(config["data"]["episodes"])
    max_steps = int(config["env"]["max_steps"])
    size = int(config["env"]["image_size"])
    heuristic_fraction = float(config["data"]["heuristic_fraction"])

    observations = np.zeros((episodes, max_steps + 1, size, size, 3), dtype=np.uint8)
    actions = np.zeros((episodes, max_steps), dtype=np.int64)
    rewards = np.zeros((episodes, max_steps), dtype=np.float32)
    continues = np.zeros((episodes, max_steps), dtype=np.float32)
    valid = np.zeros((episodes, max_steps), dtype=np.float32)
    lengths = np.zeros(episodes, dtype=np.int64)

    env = make_env(config)
    successes = 0
    for episode in trange(episodes, desc="Collecting episodes"):
        obs, _ = env.reset(seed=seed + episode)
        observations[episode, 0] = obs
        use_oracle = rng.random() < heuristic_fraction
        for step in range(max_steps):
            action = shortest_path_action(env) if use_oracle else int(rng.integers(0, 3))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            observations[episode, step + 1] = next_obs
            actions[episode, step] = action
            rewards[episode, step] = reward
            continues[episode, step] = float(not (terminated or truncated))
            valid[episode, step] = 1.0
            lengths[episode] = step + 1
            obs = next_obs
            if terminated or truncated:
                successes += int(reward > 0)
                break
    env.close()

    output = ensure_parent(config["data"]["path"])
    np.savez_compressed(
        output,
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        valid=valid,
        lengths=lengths,
    )
    print(f"Saved {episodes} episodes to {output}; successful={successes}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_empty.yaml")
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    collect(load_config(args.config), args.episodes)


if __name__ == "__main__":
    main()

