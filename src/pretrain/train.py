"""DINO pretraining of DINOv3 ViT-L/16 on the Hydro water dataset, initialized from DINOv3-SAT.

Three runs, same recipe, different `--bands`:

    python -m src.pretrain.train --bands rgb      --root ~/github/hydroloc/data_jpg
    python -m src.pretrain.train --bands landsat6 --root ~/github/hydroloc/data
    python -m src.pretrain.train --bands all      --root ~/github/hydroloc/data

Recipe follows Meta's DINOv3 -> satellite adaptation reference (FINO, `vitl16_fmow_guided.yaml`),
minus iBOT and the metadata guides: student and teacher both start from the pretrained encoder;
stage 1 keeps the backbone frozen while the freshly initialized DINO head trains; stage 2
unfreezes the backbone with a linear LR ramp. Constant LR after warmup, layerwise LR decay,
reduced patch-embed LR, frozen cls/register tokens, EMA teacher, teacher-temp warmup, grad clip,
no Gram anchoring (DINOv3 only uses it as a late refinement of the 7B; distilled models skip it).
"""

import argparse
import copy
import glob
import math
import os
import time

import timm
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.pretrain.augment import MultiCropTransform, multicrop_collate, rgb_photometric
from src.pretrain.data import MSIHydroDataset, RGBHydroDataset
from src.pretrain.dino import DINOHead, DINOLoss, MultiCropWrapper, cosine_schedule, update_teacher
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
        norm_info = {"input": "8-bit RGB quicklook (B04,B03,B02) in [0,1]", "mean": list(cfg["mean"]), "std": list(cfg["std"])}
        return RGBHydroDataset(args.root, transform), norm_info
    band_indices, _ = BAND_PRESETS[args.bands]
    clamp = None if any(math.isnan(v) for v in args.clamp) else tuple(args.clamp)
    ds = MSIHydroDataset(
        args.root, MultiCropTransform(**crop_kwargs), band_indices=band_indices, norm=args.norm, clamp=clamp
    )
    norm_info = {
        "input": "Sentinel-2 L2A DN; reflectance = (DN - per-patch BOA offset) / 10000; zero DN = nodata -> 0 after z-score",
        "band_indices": list(band_indices),
        "norm": args.norm,
        "mean": ds.mean.flatten().tolist(),
        "std": ds.std.flatten().tolist(),
        "clamp": clamp,
    }
    return ds, norm_info


def param_groups(student: MultiCropWrapper, args) -> list[dict]:
    """Head vs backbone groups; backbone gets layerwise LR decay and a patch-embed multiplier.

    1-D params and biases get no weight decay (DINO convention). cls/register tokens are frozen.
    """
    enc = student.encoder
    n_blocks = len(enc.blocks)
    groups = []

    def add(params, lr_scale, backbone):
        reg = [p for n, p in params if p.ndim > 1 and not n.endswith(".bias")]
        noreg = [p for n, p in params if not (p.ndim > 1 and not n.endswith(".bias"))]
        if reg:
            groups.append({"params": reg, "lr_scale": lr_scale, "weight_decay": args.weight_decay, "backbone": backbone})
        if noreg:
            groups.append({"params": noreg, "lr_scale": lr_scale, "weight_decay": 0.0, "backbone": backbone})

    add(list(student.head.named_parameters()), 1.0, backbone=False)

    for name in ("cls_token", "reg_token"):
        tok = getattr(enc, name, None)
        if tok is not None:
            tok.requires_grad_(False)

    decay = args.layerwise_decay
    add(list(enc.patch_embed.named_parameters()), decay ** (n_blocks + 1) * args.patch_embed_lr_mult, backbone=True)
    for i, blk in enumerate(enc.blocks):
        add(list(blk.named_parameters()), decay ** (n_blocks - i), backbone=True)
    rest = [
        (n, p) for n, p in enc.named_parameters()
        if not n.startswith(("patch_embed.", "blocks.", "cls_token", "reg_token"))
    ]
    add(rest, 1.0, backbone=True)
    return groups


def set_backbone_trainable(student: MultiCropWrapper, trainable: bool) -> None:
    for n, p in student.encoder.named_parameters():
        if n.startswith(("cls_token", "reg_token")):
            continue
        p.requires_grad_(trainable)


def save_checkpoint(path, teacher, student, optimizer, loss_fn, args, norm_info, in_chans, step):
    state = {
        "teacher_encoder": teacher.encoder.state_dict(),
        "teacher_head": teacher.head.state_dict(),
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_center": loss_fn.center,
        "args": vars(args),
        "step": step,
        "norm": norm_info,
        "encoder": {"timm_model": f"vit_large_patch16_dinov3.{args.variant}", "in_chans": in_chans, "img_size": IMG_SIZE},
    }
    torch.save(state, path)
    torch.save(state, os.path.join(os.path.dirname(path), "last.pt"))


def load_checkpoint(path, teacher, student, optimizer, loss_fn, device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["student"])
    teacher.encoder.load_state_dict(ckpt["teacher_encoder"])
    if "teacher_head" in ckpt:
        teacher.head.load_state_dict(ckpt["teacher_head"])
    else:  # older checkpoints: EMA is ~1 by then, so the student head is the closest stand-in
        teacher.head.load_state_dict(student.head.state_dict())
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        print("resume: checkpoint has no optimizer state, AdamW moments start fresh", flush=True)
    if "loss_center" in ckpt:
        loss_fn.center.copy_(ckpt["loss_center"].to(device))
    print(f"resumed from {path} at step {ckpt['step']}", flush=True)
    return int(ckpt["step"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=("rgb", *BAND_PRESETS), required=True)
    ap.add_argument("--root", required=True, help="data_jpg dir for rgb, data dir (tifs) otherwise")
    ap.add_argument("--variant", default="sat493m", choices=("sat493m", "lvd1689m"))
    ap.add_argument("--norm", default="robust", choices=("robust", "plain"),
                    help="MSI z-score stats: median/MAD (robust) or mean/std (plain); see src/pretrain/stats.py")
    ap.add_argument("--clamp", type=float, nargs=2, default=(-5.0, 15.0), metavar=("LO", "HI"),
                    help="clamp normalized MSI values; pass 'nan nan' to disable")
    # schedule (optimizer steps)
    ap.add_argument("--steps", type=int, default=15000, help="total optimizer steps")
    ap.add_argument("--batch-size", type=int, default=32, help="per forward pass")
    ap.add_argument("--accum-steps", type=int, default=8, help="effective batch = batch-size * accum-steps")
    ap.add_argument("--amp", default="bf16", choices=("bf16", "none"), help="autocast dtype on cuda")
    ap.add_argument("--lr", type=float, default=5e-5, help="peak lr (heads and full-scale backbone groups)")
    ap.add_argument("--lr-schedule", default="constant", choices=("constant", "cosine"))
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--freeze-backbone-steps", type=int, default=6000, help="stage 1: heads train, backbone frozen")
    ap.add_argument("--backbone-warmup-steps", type=int, default=1000, help="stage 2: linear LR ramp after unfreeze")
    ap.add_argument("--freeze-last-layer-steps", type=int, default=450, help="freeze DINO prototype layer")
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--clip-grad", type=float, default=3.0)
    ap.add_argument("--layerwise-decay", type=float, default=0.9)
    ap.add_argument("--patch-embed-lr-mult", type=float, default=0.2)
    ap.add_argument("--drop-path", type=float, default=0.1)
    # dino
    ap.add_argument("--out-dim", type=int, default=8192, help="DINO prototypes")
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=8)
    ap.add_argument("--local-size", type=int, default=96)
    ap.add_argument("--teacher-temp", type=float, default=0.07)
    ap.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    ap.add_argument("--warmup-teacher-temp-steps", type=int, default=3000)
    ap.add_argument("--momentum-teacher", type=float, default=0.994)
    # io
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output-dir", default="output/dino")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--resume", default=None,
                    help="checkpoint path, or 'auto' to continue from <output-dir>/last.pt if it exists")
    ap.add_argument("--limit", type=int, default=0, help="cap dataset size for a smoke test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.bands == "rgb":
        in_chans, rgb_indices = 3, (0, 1, 2)
    else:
        band_indices, rgb_indices = BAND_PRESETS[args.bands]
        in_chans = len(band_indices)

    dataset, norm_info = build_dataset(args)
    if args.limit:
        dataset.paths = dataset.paths[: args.limit]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=multicrop_collate,
        persistent_workers=args.workers > 0,
    )

    encoder = build_encoder(
        variant=args.variant, in_chans=in_chans, img_size=IMG_SIZE, rgb_indices=rgb_indices,
        drop_path_rate=args.drop_path,
    )
    student = MultiCropWrapper(encoder, DINOHead(encoder.embed_dim, args.out_dim)).to(args.device)
    teacher = copy.deepcopy(student).eval()  # eval: no drop-path noise in targets
    for p in teacher.parameters():
        p.requires_grad_(False)

    groups = param_groups(student, args)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DINOLoss(args.out_dim).to(args.device)
    trainable = [p for g in groups for p in g["params"]]

    total_steps = args.steps
    lr_final = args.lr if args.lr_schedule == "constant" else 1e-6
    lr_sched = cosine_schedule(args.lr, lr_final, total_steps, warmup_steps=args.warmup_steps)
    mom_sched = cosine_schedule(args.momentum_teacher, 1.0, total_steps)
    temp_sched = [
        args.warmup_teacher_temp + (args.teacher_temp - args.warmup_teacher_temp) * min(1.0, s / max(args.warmup_teacher_temp_steps, 1))
        for s in range(total_steps)
    ]

    def backbone_ramp(step: int) -> float:
        if step < args.freeze_backbone_steps:
            return 0.0
        return min(1.0, (step - args.freeze_backbone_steps + 1) / max(args.backbone_warmup_steps, 1))

    use_amp = args.amp == "bf16" and args.device.startswith("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    steps_per_epoch = len(loader) // args.accum_steps
    print(
        f"bands={args.bands} variant={args.variant} n={len(dataset)} in_chans={in_chans} "
        f"effective_batch={args.batch_size * args.accum_steps} steps={total_steps} "
        f"(~{total_steps / max(steps_per_epoch, 1):.1f} epochs) freeze_backbone={args.freeze_backbone_steps} amp={use_amp}",
        flush=True,
    )

    step = 0
    resume_path = args.resume
    if args.resume == "auto":
        last = os.path.join(args.output_dir, "last.pt")
        steps = sorted(
            glob.glob(os.path.join(args.output_dir, f"{args.bands}_step*.pt")),
            key=lambda p: int(p.rsplit("step", 1)[1][:-3]),
        )
        resume_path = last if os.path.exists(last) else (steps[-1] if steps else None)
    if resume_path and os.path.exists(resume_path):
        step = load_checkpoint(resume_path, teacher, student, optimizer, loss_fn, args.device)
    elif args.resume:
        print(f"resume: {resume_path} not found, starting from scratch", flush=True)

    set_backbone_trainable(student, step >= args.freeze_backbone_steps)
    unfrozen = step >= args.freeze_backbone_steps
    micro, t0, step0 = 0, time.time(), step
    optimizer.zero_grad(set_to_none=True)
    student.train()
    while step < total_steps:
        for crops in loader:
            if step >= total_steps:
                break
            if not unfrozen and step >= args.freeze_backbone_steps and micro % args.accum_steps == 0:
                set_backbone_trainable(student, True)
                unfrozen = True
                print(f"step={step} backbone unfrozen", flush=True)
            crops = [c.to(args.device, non_blocking=True) for c in crops]

            with autocast:
                with torch.no_grad():
                    teacher_output = teacher(crops[: args.n_global])
                student_output = student(crops)
                loss = loss_fn(
                    student_output.float(), teacher_output.float(), len(crops), args.n_global, temp_sched[step]
                )
            (loss / args.accum_steps).backward()
            micro += 1
            if micro % args.accum_steps != 0:
                continue

            ramp = backbone_ramp(step)
            for g in optimizer.param_groups:
                g["lr"] = lr_sched[step] * g["lr_scale"] * (ramp if g["backbone"] else 1.0)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            if step < args.freeze_last_layer_steps:
                for p in student.head.last_layer.parameters():
                    p.grad = None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_teacher(student, teacher, mom_sched[step])

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"step={step} loss={loss.item():.4f} lr={lr_sched[step]:.2e} ramp={ramp:.2f} "
                    f"ttemp={temp_sched[step]:.3f} mom={mom_sched[step]:.4f} "
                    f"{elapsed / max(step - step0, 1):.2f}s/step",
                    flush=True,
                )
            step += 1
            if step % args.save_every == 0 or step == total_steps:
                path = os.path.join(args.output_dir, f"{args.bands}_step{step}.pt")
                save_checkpoint(path, teacher, student, optimizer, loss_fn, args, norm_info, in_chans, step)
                print(f"saved {path}", flush=True)

    final = os.path.join(args.output_dir, f"{args.bands}_final.pt")
    save_checkpoint(final, teacher, student, optimizer, loss_fn, args, norm_info, in_chans, step)
    print(f"done in {(time.time() - t0) / 3600:.2f}h, saved {final}", flush=True)


if __name__ == "__main__":
    main()
