from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(*(value.detach() for value in (self.deter, self.stoch, self.mean, self.std)))


class Encoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(),
            nn.Flatten(), nn.Linear(256 * 4 * 4, embed_dim), nn.ELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class Decoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 256 * 4 * 4), nn.ELU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(feature).view(-1, 256, 4, 4))


class RSSMWorldModel(nn.Module):
    def __init__(self, action_dim: int, deter_dim: int, stoch_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.encoder = Encoder(embed_dim)
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior = nn.Sequential(nn.Linear(deter_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * stoch_dim))
        self.posterior = nn.Sequential(nn.Linear(deter_dim + embed_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * stoch_dim))
        feature_dim = deter_dim + stoch_dim
        self.decoder = Decoder(feature_dim, hidden_dim)
        self.reward_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.continue_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        zeros_d = torch.zeros(batch_size, self.deter_dim, device=device)
        zeros_s = torch.zeros(batch_size, self.stoch_dim, device=device)
        return RSSMState(zeros_d, zeros_s, zeros_s, torch.ones_like(zeros_s))

    @staticmethod
    def _stats(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = raw.chunk(2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return mean, std

    @staticmethod
    def _sample(mean: torch.Tensor, std: torch.Tensor, sample: bool) -> torch.Tensor:
        return mean + std * torch.randn_like(mean) if sample else mean

    def infer(self, image: torch.Tensor, sample: bool = False) -> RSSMState:
        initial = self.initial(image.shape[0], image.device)
        mean, std = self._stats(self.posterior(torch.cat([initial.deter, self.encoder(image)], -1)))
        return RSSMState(initial.deter, self._sample(mean, std, sample), mean, std)

    def observe_step(self, previous: RSSMState, action: torch.Tensor, image: torch.Tensor, sample: bool = True):
        one_hot = F.one_hot(action.long(), self.action_dim).float()
        deter = self.gru(torch.cat([previous.stoch, one_hot], -1), previous.deter)
        prior_mean, prior_std = self._stats(self.prior(deter))
        post_mean, post_std = self._stats(self.posterior(torch.cat([deter, self.encoder(image)], -1)))
        posterior = RSSMState(deter, self._sample(post_mean, post_std, sample), post_mean, post_std)
        prior = RSSMState(deter, self._sample(prior_mean, prior_std, sample), prior_mean, prior_std)
        return posterior, prior

    def imagine_step(self, previous: RSSMState, action: torch.Tensor, sample: bool = False) -> RSSMState:
        one_hot = F.one_hot(action.long(), self.action_dim).float()
        deter = self.gru(torch.cat([previous.stoch, one_hot], -1), previous.deter)
        mean, std = self._stats(self.prior(deter))
        return RSSMState(deter, self._sample(mean, std, sample), mean, std)

    @staticmethod
    def feature(state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], -1)

    def decode_logits(self, state: RSSMState) -> torch.Tensor:
        return self.decoder(self.feature(state))

    def decode(self, state: RSSMState) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(state))

    def predict_reward(self, state: RSSMState) -> torch.Tensor:
        return self.reward_head(self.feature(state)).squeeze(-1)

    def predict_continue_logits(self, state: RSSMState) -> torch.Tensor:
        return self.continue_head(self.feature(state)).squeeze(-1)


def gaussian_kl(post: RSSMState, prior: RSSMState) -> torch.Tensor:
    variance_ratio = (post.std / prior.std).pow(2)
    mean_term = ((post.mean - prior.mean) / prior.std).pow(2)
    return 0.5 * (variance_ratio + mean_term - 1.0 - variance_ratio.log()).sum(-1)


def model_from_config(config: dict) -> RSSMWorldModel:
    params = config["model"]
    return RSSMWorldModel(
        action_dim=int(config["env"]["action_dim"]),
        deter_dim=int(params["deter_dim"]),
        stoch_dim=int(params["stoch_dim"]),
        embed_dim=int(params["embed_dim"]),
        hidden_dim=int(params["hidden_dim"]),
    )

