"""DINO self-distillation components (Caron et al. 2021; Oquab et al. 2023).

Student sees all crops (global + local), teacher (EMA of student, no grad) sees only the
global crops. Loss is cross-entropy between the teacher's centered + sharpened softmax and
the student's softmax, summed over every (teacher view, student view) pair except the
identical view.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DINOHead(nn.Module):
    """3-layer MLP -> L2-normalized bottleneck -> weight-normalized prototype layer."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int = 2048, bottleneck_dim: int = 256
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # Fix the weight_norm magnitude g at 1 so prototypes live on the unit sphere.
        g = self.last_layer.parametrizations.weight.original0
        g.data.fill_(1.0)
        g.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiCropWrapper(nn.Module):
    """Encoder + head over a list of per-view batches, one forward per distinct resolution."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, crops: list[Tensor]) -> Tensor:
        sizes = [c.shape[-1] for c in crops]
        outputs, start = [], 0
        for i in range(1, len(sizes) + 1):
            if i == len(sizes) or sizes[i] != sizes[start]:
                outputs.append(self.encoder(torch.cat(crops[start:i], dim=0)))
                start = i
        return self.head(torch.cat(outputs, dim=0))


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: Tensor,
        teacher_output: Tensor,
        n_student_views: int,
        n_teacher_views: int,
        teacher_temp: float,
    ) -> Tensor:
        student_out = (student_output / self.student_temp).chunk(n_student_views)
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(n_teacher_views)

        total, n_terms = 0.0, 0
        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                if iv == iq:
                    continue
                total += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
                n_terms += 1
        self.update_center(teacher_output)
        return total / n_terms

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.detach().data, alpha=1 - momentum)


def cosine_schedule(
    base: float, final: float, total_steps: int, warmup_steps: int = 0, start_warmup: float = 0.0
) -> list[float]:
    """Linear warmup then cosine decay; used for lr and for teacher EMA momentum."""
    warmup = [start_warmup + (base - start_warmup) * i / max(warmup_steps, 1) for i in range(warmup_steps)]
    n_cos = total_steps - warmup_steps
    cos = [final + 0.5 * (base - final) * (1 + math.cos(math.pi * i / max(n_cos, 1))) for i in range(n_cos)]
    return warmup + cos


def teacher_temp_schedule(
    warmup_temp: float, temp: float, warmup_epochs: int, total_epochs: int, steps_per_epoch: int
) -> list[float]:
    warmup_steps = min(warmup_epochs, total_epochs) * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    warmup = [warmup_temp + (temp - warmup_temp) * i / max(warmup_steps, 1) for i in range(warmup_steps)]
    return warmup + [temp] * (total_steps - warmup_steps)
