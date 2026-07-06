from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _feature_tensor(output):
    """Handle both Transformers 4.x and 5.x CLIP outputs.

    Transformers 4.x get_*_features usually returns a Tensor directly.
    Some newer builds can return a BaseModelOutputWithPooling-like object.
    For CLIP scoring we need the pooled embedding tensor.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, (tuple, list)):
        # CLIP-like outputs often keep the pooled vector at index 1.
        if len(output) > 1 and isinstance(output[1], torch.Tensor):
            return output[1]
        if len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")


class CLIPGoalScorer:
    """Frozen CLIP scorer for decoded imagined frames.

    The planner passes float BCHW tensors in [0, 1]. I convert them to ordinary
    uint8 PIL images before CLIP preprocessing. This avoids the common pitfall
    where CLIPProcessor treats float tensors as images that still need rescaling,
    which can silently change the brightness distribution of RSSM-decoded frames.
    """

    def __init__(self, model_name: str, goal: str, negatives: list[str], device: torch.device):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        texts = [goal, *negatives]
        tokens = self.processor(text=texts, return_tensors="pt", padding=True)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            text = _feature_tensor(self.model.get_text_features(**tokens))
            self.text_features = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _to_pil_images(frames: torch.Tensor) -> list[Image.Image]:
        if frames.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape={tuple(frames.shape)}")
        array = (
            frames.detach()
            .cpu()
            .clamp(0.0, 1.0)
            .permute(0, 2, 3, 1)
            .numpy()
        )
        return [Image.fromarray((frame * 255.0).astype(np.uint8)) for frame in array]

    @torch.inference_mode()
    def score(self, frames: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        """Return P(goal | frame, goal+negative prompts) for each frame."""
        scores = []
        for chunk in frames.split(batch_size):
            pil_images = self._to_pil_images(chunk)
            inputs = self.processor(images=pil_images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            image = _feature_tensor(self.model.get_image_features(pixel_values=pixels))
            image = image / image.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            logits = self.model.logit_scale.exp() * image @ self.text_features.T
            scores.append(logits.softmax(dim=-1)[:, 0].detach().cpu())
        return torch.cat(scores).to(frames.device)
