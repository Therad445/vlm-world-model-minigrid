from __future__ import annotations

import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper

from .utils import resize_frame


class FullRGBEnv:
    """Return a compact, fully observable RGB frame while retaining MiniGrid state."""

    def __init__(self, env_id: str, image_size: int, max_steps: int):
        self.env = RGBImgObsWrapper(
            gym.make(env_id, render_mode="rgb_array", max_steps=max_steps),
            tile_size=16,
        )
        self.image_size = image_size

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _frame(self, obs: dict) -> np.ndarray:
        return resize_frame(obs["image"], self.image_size)

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return self._frame(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self._frame(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()


def make_env(config: dict) -> FullRGBEnv:
    return FullRGBEnv(
        config["env"]["id"],
        config["env"]["image_size"],
        config["env"]["max_steps"],
    )


def shortest_path_action(env: FullRGBEnv) -> int:
    """Oracle data-collection policy for EmptyEnv; actions 0/1/2 only."""
    base = env.unwrapped
    ax, ay = map(int, base.agent_pos)
    # EmptyEnv places its goal in the lower-right interior cell but does not
    # expose a stable public goal_pos attribute across MiniGrid versions.
    gx, gy = int(base.width - 2), int(base.height - 2)
    dx, dy = gx - ax, gy - ay
    if abs(dx) >= abs(dy) and dx != 0:
        target_dir = 0 if dx > 0 else 2
    elif dy != 0:
        target_dir = 1 if dy > 0 else 3
    else:
        return 2
    current = int(base.agent_dir)
    delta = (target_dir - current) % 4
    if delta == 0:
        return 2
    return 1 if delta in (1, 2) else 0
