"""Augmentations for the contrastive loss"""
import math
import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def _rand(a, b):
    return a + (b - a) * random.random()


class SimCLRAugment(nn.Module):
    def __init__(
        self,
        size: int = 32,
        strength: float = 1.0,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        p_grayscale: float = 0.2,
        p_flip: float = 0.5,
        use_hue: bool = True,
    ):
        super().__init__()
        self.size = size
        self.s = strength
        self.scale = scale
        self.ratio = ratio
        self.p_grayscale = p_grayscale
        self.p_flip = p_flip
        self.use_hue = use_hue

    def _crop_params(self, h, w):
        area = h * w
        for _ in range(10):
            target_area = area * _rand(*self.scale)
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect = math.exp(_rand(*log_ratio))
            cw = int(round(math.sqrt(target_area * aspect)))
            ch = int(round(math.sqrt(target_area / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                i = random.randint(0, h - ch)
                j = random.randint(0, w - cw)
                return i, j, ch, cw
        return 0, 0, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one augmented view of a (B, 3, H, W) batch in [0, 1]."""
        b, c, h, w = x.shape

        i, j, ch, cw = self._crop_params(h, w)
        out = TF.resized_crop(x, i, j, ch, cw, [self.size, self.size], antialias=True)

        if random.random() < self.p_flip:
            out = TF.hflip(out)

        ops = [
            ("brightness", 0.8 * self.s),
            ("contrast", 0.8 * self.s),
            ("saturation", 0.8 * self.s),
        ]
        random.shuffle(ops)
        for name, mag in ops:
            factor = _rand(max(0.0, 1.0 - mag), 1.0 + mag)
            if name == "brightness":
                out = TF.adjust_brightness(out, factor)
            elif name == "contrast":
                out = TF.adjust_contrast(out, factor)
            else:
                out = TF.adjust_saturation(out, factor)
        if self.use_hue:
            out = TF.adjust_hue(out, _rand(-0.2 * self.s, 0.2 * self.s))

        if random.random() < self.p_grayscale:
            out = TF.rgb_to_grayscale(out, num_output_channels=3)

        return out.clamp(0.0, 1.0)


def make_views(x: torch.Tensor, aug: SimCLRAugment, n_views: int = 4) -> torch.Tensor:
    """Stack n_views augmented copies of x into one (n_views*B, 3, H, W) batch."""
    return torch.cat([aug(x) for _ in range(n_views)], dim=0)
