"""Convolutional autoencoder for anomaly detection.

Architecture overview
---------------------
Encoder:
    For each of *num_blocks* stages the spatial resolution is halved with
    ``MaxPool2d(2)`` and the channel count is doubled (starting from
    *base_channels*).  After the final block an ``AdaptiveAvgPool2d(4)``
    collapses spatial dims to 4x4, which is then flattened and projected
    to *latent_dim* via a linear layer.

Decoder:
    A linear layer maps *latent_dim* back to the 4x4 feature map.  Each
    stage applies ``Upsample(scale_factor=2, mode='bilinear')`` followed by
    a ``ConvBlock``, progressively halving the channel count until we reach
    *base_channels*.  A final 1x1 convolution + Sigmoid produces the
    output in [0, 1] range -- but note that our preprocessing normalises
    with ImageNet statistics, so the network actually regresses
    normalised pixel values.

    After the learned upsampling stages, the spatial size may be smaller
    than the original *image_size* (because ``AdaptiveAvgPool2d`` discards
    fine-grained spatial info).  An ``nn.Upsample`` is appended to
    guarantee the output exactly matches ``(image_size, image_size)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
# Building blocks
# ======================================================================

class ConvBlock(nn.Module):
    """Two successive Conv-BN-LeakyReLU layers with an optional residual shortcut.

    When *residual=True* and the channel counts differ, a 1x1 convolution is
    used to match dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.residual = residual
        if residual:
            self.shortcut: Optional[nn.Module]
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.residual:
            out = out + self.shortcut(identity)

        out = self.act2(out)
        return out


# ======================================================================
# Encoder
# ======================================================================

class Encoder(nn.Module):
    """Contracting path: ConvBlock + MaxPool repeated *num_blocks* times,
    then AdaptiveAvgPool(4) and a linear projection to *latent_dim*.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_blocks: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        ch_in = in_channels
        ch_out = base_channels

        for _ in range(num_blocks):
            blocks.append(ConvBlock(ch_in, ch_out, residual=True))
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch_in = ch_out
            ch_out = min(ch_out * 2, 512)  # cap to avoid explosion

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(4)

        # After pool: (ch_in, 4, 4)  -- ch_in is the *last* output channel count
        self.flatten_dim = ch_in * 4 * 4
        self.fc = nn.Linear(self.flatten_dim, latent_dim)
        self.final_channels = ch_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ======================================================================
# Decoder
# ======================================================================

class Decoder(nn.Module):
    """Expanding path that mirrors the encoder."""

    def __init__(
        self,
        latent_dim: int,
        base_channels: int,
        num_blocks: int,
        out_channels: int,
        image_size: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size

        # Compute the channel progression (must mirror encoder exactly)
        channel_list: list[int] = []
        ch = base_channels
        for _ in range(num_blocks):
            channel_list.append(ch)
            ch = min(ch * 2, 512)
        # After the loop, *ch* has been doubled one extra time; the encoder's
        # last output channels equal channel_list[-1], so the decoder starts
        # from that value.
        start_channels = channel_list[-1]

        self.start_channels = start_channels
        self.fc = nn.Linear(latent_dim, start_channels * 4 * 4)

        blocks: list[nn.Module] = []
        reversed_channels = list(reversed(channel_list))
        ch_in = start_channels

        for i in range(num_blocks):
            ch_out = reversed_channels[i + 1] if i + 1 < num_blocks else base_channels
            blocks.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            blocks.append(ConvBlock(ch_in, ch_out, residual=True))
            ch_in = ch_out

        self.blocks = nn.Sequential(*blocks)

        # Final projection to image channels
        self.head = nn.Sequential(
            nn.Conv2d(ch_in, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Guarantee final spatial size matches input image
        self.final_resize = nn.Upsample(
            size=(image_size, image_size), mode="bilinear", align_corners=False
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.start_channels, 4, 4)
        x = self.blocks(x)
        x = self.head(x)
        x = self.final_resize(x)
        return x


# ======================================================================
# Full autoencoder
# ======================================================================

class AnomalyAutoencoder(nn.Module):
    """Combines :class:`Encoder` and :class:`Decoder` into a single model.

    Parameters
    ----------
    in_channels:
        1 for grayscale, 3 for RGB.
    latent_dim:
        Dimensionality of the bottleneck vector.
    base_channels:
        Number of channels after the first encoder block.
    num_blocks:
        Number of encoder (and decoder) stages.
    image_size:
        Spatial resolution the network is designed for (assumes square input).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 32,
        num_blocks: int = 4,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.image_size = image_size

        self.encoder = Encoder(in_channels, base_channels, num_blocks, latent_dim)
        self.decoder = Decoder(latent_dim, base_channels, num_blocks, in_channels, image_size)

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward / latent
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Reconstruct *x* through the bottleneck."""
        z = self.encoder(x)
        return self.decoder(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent-space vector for *x*."""
        return self.encoder(x)

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
