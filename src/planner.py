from __future__ import annotations

import torch

from .rssm import RSSMState, RSSMWorldModel


class RandomShootingPlanner:
    def __init__(self, model: RSSMWorldModel, config: dict, vlm_scorer=None):
        self.model = model
        self.vlm = vlm_scorer
        self.horizon = int(config["horizon"])
        self.candidates = int(config["candidates"])
        self.action_dim = int(config["action_dim"])
        self.gamma = float(config["gamma"])
        self.vlm_weight = float(config["vlm_weight"])
        self.vlm_stride = int(config.get("vlm_stride", 1))

    @staticmethod
    def _repeat(state: RSSMState, count: int) -> RSSMState:
        return RSSMState(*(value.repeat(count, 1) for value in (state.deter, state.stoch, state.mean, state.std)))

    @torch.inference_mode()
    def plan(self, observation: torch.Tensor, use_vlm: bool, return_debug: bool = False):
        state = self._repeat(self.model.infer(observation, sample=False), self.candidates)
        device = observation.device
        sequences = torch.randint(self.action_dim, (self.candidates, self.horizon), device=device)
        rewards, frames, frame_steps = [], [], []
        for step in range(self.horizon):
            state = self.model.imagine_step(state, sequences[:, step], sample=False)
            rewards.append(self.model.predict_reward(state))
            if use_vlm and ((step + 1) % self.vlm_stride == 0 or step == self.horizon - 1):
                frames.append(self.model.decode(state))
                frame_steps.append(step + 1)
        reward_matrix = torch.stack(rewards, dim=1)
        discounts = self.gamma ** torch.arange(self.horizon, device=device)
        objective = (reward_matrix * discounts).sum(dim=1)
        vlm_matrix = None
        if use_vlm:
            if self.vlm is None:
                raise RuntimeError("use_vlm=True requires a VLM scorer")
            frame_tensor = torch.stack(frames, dim=1)
            flat_scores = self.vlm.score(frame_tensor.flatten(0, 1))
            scored_steps = len(frame_steps)
            vlm_matrix = flat_scores.view(self.candidates, scored_steps)
            objective = objective + self.vlm_weight * vlm_matrix.max(dim=1).values
        best = int(objective.argmax())
        debug = None
        if return_debug:
            debug = {
                "actions": sequences[best].detach().cpu(),
                "predicted_rewards": reward_matrix[best].detach().cpu(),
                "frames": torch.stack(frames, dim=1)[best].detach().cpu() if frames else None,
                "vlm_scores": vlm_matrix[best].detach().cpu() if vlm_matrix is not None else None,
                "frame_steps": frame_steps,
                "objective": float(objective[best]),
            }
        return int(sequences[best, 0]), debug
