"""DINO pretraining of DINOv3 ViT-L/16 on the Hydro water dataset, initialized from DINOv3-SAT.

Two runs, same recipe, different `--bands`:

    python -m src.pretrain.train --bands rgb --root ~/github/hydroloc/data_jpg
    python -m src.pretrain.train --bands all --root ~/github/hydroloc/data

Recipe (Caron et al. 2021 / Oquab et al. 2023, scaled down for ~106k images):
  student/teacher pair, teacher = EMA of student (cosine momentum -> 1.0), teacher sees the 2
  global crops, student sees global + local crops, cross-entropy against the teacher's
  centered + sharpened softmax, teacher-temp warmup, lr warmup + cosine decay, grad clipping,
  and the prototype layer frozen for the first epoch.
"""

import argparse
import copy
import math
import os
import time

import timm
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.pretrain.augment import MultiCropTransform, multicrop_collate, rgb_photometric
from src.pretrain.data import MSIHydroDataset, RGBHydroDataset
from src.pretrain.dino import (
    DINOHead,
    DINOLoss,
    MultiCropWrapper,
    cosine_schedule,
    teacher_temp_schedule,
    update_teacher,
)
from src.pretrain.model import build_encoder

IMG_SIZE = 256

# tif band order: [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]
# preset -> (tif band indices to read, positions of R/G/B within that subset)
BAND_PRESETS = {
    "all": (tuple(range(12)), (3, 2, 1)),
    # Landsat OLI-equivalent: B02 blue, B03 green, B04 red, B08 NIR, B11 SWIR1, B12 SWIR2.
    # Keeps the encoder transferable to Landsat/HLS without a channel mismatch.
    "landsat6": ((1, 2, 3, 7, 10, 11), (2, 1, 0)),
}


def build_dataset(args):
    crop_kwargs = dict(
        n_global=args.n_global, n_local=args.n_local,
        global_size=IMG_SIZE, local_size=args.local_size,
    )
    if args.bands == "rgb":
        probe = timm.create_model(f"vit_large_patch16_dinov3.{args.variant}", pretrained=False, num_classes=0)
        cfg = timm.data.resolve_model_data_config(probe)
        finalize = T.Compose([T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
        transform = MultiCropTransform(**crop_kwargs, photometric=rgb_photometric(), finalize=finalize)
        return RGBHydroDataset(args.root, transform)
    band_indices, _ = BAND_PRESETS[args.bands]
    clamp = None if any(math.isnan(v) for v in args.clamp) else tuple(args.clamp)
    return MSIHydroDataset(
        args.root, MultiCropTransform(**crop_kwargs), band_indices=band_indices, norm=args.norm, clamp=clamp
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=("rgb", *BAND_PRESETS), required=True)
    ap.add_argument("--root", required=True, help="data_jpg dir for rgb, data dir (tifs) otherwise")
    ap.add_argument("--variant", default="sat493m", choices=("sat493m", "lvd1689m"))
    ap.add_argument("--norm", default="robust", choices=("robust", "plain"),
                    help="MSI z-score stats: median/MAD (robust) or mean/std (plain); see src/pretrain/stats.py")
    ap.add_argument("--clamp", type=float, nargs=2, default=(-5.0, 15.0), metavar=("LO", "HI"),
                    help="clamp normalized MSI values; pass 'nan nan' to disable")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5, help="peak lr after warmup")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--clip-grad", type=float, default=3.0)
    ap.add_argument("--out-dim", type=int, default=8192, help="DINO prototypes")
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=6)
    ap.add_argument("--local-size", type=int, default=96)
    ap.add_argument("--teacher-temp", type=float, default=0.07)
    ap.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    ap.add_argument("--warmup-teacher-temp-epochs", type=int, default=2)
    ap.add_argument("--momentum-teacher", type=float, default=0.996)
    ap.add_argument("--freeze-last-layer-epochs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output-dir", default="checkpoints/dino-hydro")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="cap dataset size for a smoke test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.bands == "rgb":
        in_chans, rgb_indices = 3, (0, 1, 2)
    else:
        band_indices, rgb_indices = BAND_PRESETS[args.bands]
        in_chans = len(band_indices)

    dataset = build_dataset(args)
    if args.limit:
        dataset.paths = dataset.paths[: args.limit]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=multicrop_collate,
    )

    encoder = build_encoder(
        variant=args.variant, in_chans=in_chans, img_size=IMG_SIZE, rgb_indices=rgb_indices
    )
    student = MultiCropWrapper(encoder, DINOHead(encoder.embed_dim, args.out_dim)).to(args.device)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False

    loss_fn = DINOLoss(args.out_dim).to(args.device)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    lr_sched = cosine_schedule(args.lr, 1e-6, total_steps, warmup_steps=args.warmup_epochs * steps_per_epoch)
    mom_sched = cosine_schedule(args.momentum_teacher, 1.0, total_steps)
    temp_sched = teacher_temp_schedule(
        args.warmup_teacher_temp, args.teacher_temp, args.warmup_teacher_temp_epochs,
        args.epochs, steps_per_epoch,
    )

    print(f"bands={args.bands} variant={args.variant} n={len(dataset)} steps/epoch={steps_per_epoch}")
    step = 0
    for epoch in range(args.epochs):
        student.train()
        t0 = time.time()
        for crops in loader:
            crops = [c.to(args.device, non_blocking=True) for c in crops]
            for g in optimizer.param_groups:
                g["lr"] = lr_sched[step]

            with torch.no_grad():
                teacher_output = teacher(crops[: args.n_global])
            student_output = student(crops)
            loss = loss_fn(student_output, teacher_output, len(crops), args.n_global, temp_sched[step])

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            if epoch < args.freeze_last_layer_epochs:
                for p in student.head.last_layer.parameters():
                    p.grad = None
            optimizer.step()
            update_teacher(student, teacher, mom_sched[step])

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.4f} "
                    f"lr={lr_sched[step]:.2e} ttemp={temp_sched[step]:.3f} mom={mom_sched[step]:.4f}"
                )
            step += 1

        ckpt_path = os.path.join(args.output_dir, f"{args.bands}_epoch{epoch}.pt")
        torch.save(
            {"teacher_encoder": teacher.encoder.state_dict(), "student": student.state_dict(), "args": vars(args)},
            ckpt_path,
        )
        print(f"epoch={epoch} done in {time.time() - t0:.1f}s, saved {ckpt_path}")


if __name__ == "__main__":
    main()
