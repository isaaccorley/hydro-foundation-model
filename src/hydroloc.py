"""HydroLoc: a ReSIREN location encoder distilled from DINOv3 + OlmoEarth embeddings.

Dependency-light standalone: needs only `torch` (and `huggingface_hub` + `safetensors`
for `from_pretrained`). Maps a (lat, lon) coordinate to a 512-d embedding whose nested
Matryoshka prefixes (64/128/256/512) are each usable as a compact location embedding.

Example
-------
    import torch
    from src.hydroloc import HydroLoc

    model = HydroLoc.from_pretrained("isaaccorley/hydroloc").eval()
    latlon = torch.tensor([[37.77, -122.42], [-8.70, 45.00]])  # (lat, lon) degrees
    emb = model(latlon)          # [2, 512]
    emb64 = model(latlon)[:, :64]  # compact 64-d Matryoshka prefix
"""

import json
import math

import torch
from torch import Tensor, nn

# Equal-Earth projection constants (Savric, Patterson & Jenny 2018), as used by GeoCLIP.
_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796
_SF = 66.50336
_SQRT3 = math.sqrt(3.0)


def equal_earth_projection(latlon: Tensor) -> Tensor:
    """Project (lat, lon) degrees -> 2D Equal-Earth coordinates, scaled to ~[-1, 1]."""
    lat = torch.deg2rad(latlon[..., 0])
    lon = torch.deg2rad(latlon[..., 1])
    theta = torch.asin((_SQRT3 / 2.0) * torch.sin(lat))
    denom = 3.0 * (9.0 * _A4 * theta**8 + 7.0 * _A3 * theta**6 + 3.0 * _A2 * theta**2 + _A1)
    x = (2.0 * _SQRT3 * lon * torch.cos(theta)) / denom
    y = _A1 * theta + _A2 * theta**3 + _A3 * theta**7 + _A4 * theta**9
    return torch.stack((x, y), dim=-1) * _SF / 180.0


class SIRENLayer(nn.Module):
    """sin(w0 * (Wx + b)), with SIREN-style initialization."""

    def __init__(self, in_f: int, out_f: int, w0: float = 1.0, is_first: bool = False) -> None:
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        bound = 1.0 / in_f if is_first else math.sqrt(6.0 / in_f) / w0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * self.linear(x))


class HydroLoc(nn.Module):
    """Residual-SIREN location encoder over Equal-Earth-projected (lat, lon)."""

    def __init__(
        self, embed_dim: int = 512, depth: int = 4, w0_first: float = 30.0, w0: float = 1.0
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.first = SIRENLayer(2, embed_dim, w0=w0_first, is_first=True)
        self.blocks = nn.ModuleList(
            SIRENLayer(embed_dim, embed_dim, w0=w0) for _ in range(depth)
        )

    def forward(self, latlon: Tensor) -> Tensor:
        """(B, 2) tensor of (lat, lon) degrees -> (B, embed_dim) embedding."""
        h = self.first(equal_earth_projection(latlon))
        for blk in self.blocks:
            h = h + blk(h)
        return h

    @classmethod
    def from_config(cls, config: dict) -> "HydroLoc":
        return cls(
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            w0_first=config.get("w0_first", 30.0),
        )

    @classmethod
    def from_pretrained(cls, repo_id: str = "isaaccorley/hydroloc") -> "HydroLoc":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        with open(hf_hub_download(repo_id, "config.json")) as f:
            config = json.load(f)
        model = cls.from_config(config)
        state = load_file(hf_hub_download(repo_id, "model.safetensors"))
        model.load_state_dict(state)
        return model.eval()
