"""DINOv3 ViT-L/16 encoder construction for pretraining on the Hydro water dataset.

Both variants are initialized from the already-pretrained DINOv3-SAT (`sat493m`) weights
rather than trained from scratch, and differ only in patch_embed.in_chans:

- RGB (3ch): patch_embed is used as-is.
- MSI (12ch, Sentinel-2 L2A band order [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]):
  patch_embed.proj is inflated from 3->12 input channels. The 3 channels matching the
  pretrained R/G/B bands (tif indices 3/2/1) keep their original filters exactly; the 9 new
  bands are initialized from the mean RGB filter (scaled to preserve activation magnitude)
  so training starts close to the pretrained solution and only has to learn the new bands.

The encoder's `dynamic_img_size=True` + RoPE positional encoding means multi-crop views at
different resolutions (global 256px, local 96px) flow through unchanged, with no
positional-embedding interpolation needed.
"""

import timm
import torch
from torch import nn

# tif band order: [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12] (0-indexed)
# RGB = [B04, B03, B02] -> tif indices [3, 2, 1], matching src/datasets/hydro.py
RGB_BAND_INDICES = (3, 2, 1)


def inflate_patch_embed(proj: nn.Conv2d, in_chans: int, rgb_indices=RGB_BAND_INDICES) -> nn.Conv2d:
    """Expand a pretrained RGB patch_embed conv to `in_chans`, preserving the RGB filters."""
    new_proj = nn.Conv2d(
        in_chans,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        bias=proj.bias is not None,
    )
    # Same recipe as Meta's dinov3/utils/expand_patch_embed.py (FINO): new channels get the
    # mean RGB filter and the whole weight is scaled by C_old / C_new so the per-patch activation
    # magnitude matches the original. Only difference: RGB filters go to the target's actual
    # R/G/B band positions instead of channels 0-2.
    with torch.no_grad():
        mean_w = proj.weight.mean(dim=1, keepdim=True)
        new_proj.weight.copy_(mean_w.repeat(1, in_chans, 1, 1))
        for new_ch, old_ch in zip(rgb_indices, range(proj.in_channels)):
            new_proj.weight[:, new_ch] = proj.weight[:, old_ch]
        new_proj.weight.mul_(proj.in_channels / in_chans)
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)
    return new_proj


def build_encoder(
    variant: str = "sat493m",
    in_chans: int = 3,
    img_size: int = 256,
    rgb_indices=RGB_BAND_INDICES,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Load a pretrained DINOv3 ViT-L/16 (timm) and optionally inflate it to `in_chans`.

    `rgb_indices` are the positions of R, G, B within the *selected* band subset, so the
    pretrained RGB filters land on the right channels for any preset.

    Called with num_classes=0 so `encoder(x)` returns the pooled image embedding
    (B, embed_dim) that the DINO projection head consumes.
    """
    name = f"vit_large_patch16_dinov3.{variant}"
    encoder = timm.create_model(
        name, pretrained=True, num_classes=0, img_size=img_size, drop_path_rate=drop_path_rate
    )
    if in_chans != encoder.patch_embed.proj.in_channels:
        encoder.patch_embed.proj = inflate_patch_embed(encoder.patch_embed.proj, in_chans, rgb_indices)
        encoder.in_chans = in_chans
    return encoder
