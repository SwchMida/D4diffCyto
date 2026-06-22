"""
AnoDDPM inference for simple folder datasets.

Expected dataset layout (relative to repo root by default):
  data/
    train/normal/*.jpg
    test/normal/*.jpg
    test/abnormal/*.jpg

This script:
- loads the trained EMA model checkpoint from: model/diff-params-ARGS=<ARGNUM>/params-final.pt
- runs partial diffusion reconstruction at timestep distance (lambda) = --detect_t (default: args["sample_distance"])
- writes per-image anomaly scores to: metrics/ARGS=<ARGNUM>/cells_scores.csv
- optionally saves example reconstructions + mse heatmaps.
- optionally saves unconditional diffusion samples (start from noise, run reverse to x_0).

Run examples:
  python cells_detection.py args90.json --detect_t 250
  python cells_detection.py args90_d4_equiv_noattn.json --detect_t 250 --save_examples
  python cells_detection.py args90_d4_equiv_noattn.json --save_samples --num_samples 64 --sample_batch 8
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel


def resolve_args_path(spec: str) -> Path:
    spec = str(spec).strip()

    if spec.isnumeric():
        return Path("test_args") / f"args{spec}.json"

    if spec.startswith("args") and spec.endswith(".json"):
        return Path("test_args") / spec

    if spec.startswith("args") and not spec.endswith(".json"):
        return Path("test_args") / f"{spec}.json"

    p = Path("test_args") / spec
    if p.exists():
        return p

    p2 = Path("test_args") / f"args{spec}.json"
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Could not resolve args file from: {spec}")


def argnum_from_argsfile(p: Path) -> str:
    name = p.name
    if name.startswith("args") and name.endswith(".json"):
        return name[4:-5]
    return p.stem


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor in [-1, 1] to uint8 image for saving/plotting.
    """
    img = (img_t.clamp(-1, 1) + 1.0) * 127.5
    img = img.to(torch.uint8)
    if img.ndim == 3:
        return img.permute(1, 2, 0).cpu().numpy()
    return img.cpu().numpy()


def save_tensor_image(path: Path, x: torch.Tensor, in_channels: int):
    """
    Save a single BCHW or CHW tensor image (in [-1,1]) to PNG.
    """
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    img = to_uint8(x)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    if in_channels == 3:
        plt.imshow(img)
    else:
        plt.imshow(img.squeeze(), cmap="gray")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def run_unconditional_sampling(
    diffusion: GaussianDiffusionModel,
    unet_ema: torch.nn.Module,
    args: dict,
    device: torch.device,
    in_channels: int,
    out_dir: Path,
    denoise_mode: str,
    num_samples: int,
    sample_batch: int,
    sample_t: int,
):
    """
    Generate unconditional samples:
      - start from noise at timestep sample_t (default: T)
      - run reverse diffusion to t=0
      - save final samples as PNG
    """
    ensure_dir(out_dir)

    H, W = int(args["img_size"][0]), int(args["img_size"][1])
    T = int(args["T"])
    if sample_t is None:
        sample_t = T
    sample_t = int(sample_t)
    if sample_t <= 0 or sample_t > T:
        raise ValueError(f"--sample_t must be in [1, T={T}]")

    remaining = int(num_samples)
    idx = 0

    with torch.no_grad():
        while remaining > 0:
            b = min(int(sample_batch), remaining)

            # Start from noise as x_{sample_t}
            # Use gaussian noise for unconditional generation by default.
            # If you REALLY want simplex, set denoise_mode=noise_fn and we still start from torch.randn_like
            # (simplex is time-dependent and this repo's sampling is usually fine with gaussian init).
            x = torch.randn((b, in_channels, H, W), device=device)

            # Now run reverse from t=sample_t-1 down to 0.
            # We mimic GaussianDiffusionModel.forward_backward reverse loop, but starting from x.
            for t in range(sample_t - 1, -1, -1):
                t_batch = torch.tensor([t], device=device).repeat(b)
                out = diffusion.sample_p(unet_ema, x, t_batch, denoise_fn=denoise_mode)
                x = out["sample"]

            # Save images
            for i in range(b):
                save_tensor_image(out_dir / f"sample_{idx:05d}.png", x[i:i+1], in_channels)
                idx += 1

            remaining -= b

    print(f"Saved unconditional samples to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", help="Args spec: e.g. 90, args90.json, args90, args90_d4_equiv_noattn.json")

    # detection / scoring
    ap.add_argument("--detect_t", type=int, default=None,
                    help="Partial diffusion distance (lambda). Default: args['sample_distance']")
    ap.add_argument("--batch", type=int, default=1, help="Inference batch size (keep small).")
    ap.add_argument("--save_examples", action="store_true", help="Save example reconstructions + heatmaps.")
    ap.add_argument("--num_examples", type=int, default=16, help="How many examples to save.")

    # sampling
    ap.add_argument("--save_samples", action="store_true",
                    help="Save unconditional DDPM samples (start from noise, reverse to x_0).")
    ap.add_argument("--num_samples", type=int, default=64, help="Total unconditional samples to save.")
    ap.add_argument("--sample_batch", type=int, default=8, help="Batch size for unconditional sampling.")
    ap.add_argument("--sample_t", type=int, default=None,
                    help="Start timestep for unconditional sampling. Default: args['T'] (full).")
    ap.add_argument("--sample_dir", type=str, default=None,
                    help="Optional subfolder name under diffusion-samples/ARGS=<argnum>/")

    # denoising noise choice for reverse steps
    ap.add_argument("--denoise_mode", type=str, default="noise_fn",
                    help="Noise used in reverse steps: 'noise_fn' (recommended), 'gauss', etc.")

    args_cli = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load args ----------
    args_path = resolve_args_path(args_cli.args)
    with open(args_path, "r") as f:
        args = json.load(f)
    arg_num = argnum_from_argsfile(args_path)

    # ---------- Group config ----------
    group_equiv = args.get("group", "none")
    group_fa = bool(args.get("group_fa", False))
    group_en = bool(args.get("group_en", False))
    group_fa_train = bool(args.get("group_fa_train", False))
    use_attention = args.get("use_attention", None)

    print(
        f"[CONFIG] ARGNUM={arg_num} | group={group_equiv} | "
        f"group_fa={group_fa} | group_en={group_en} | use_attention={use_attention}",
        flush=True
    )

    # ---------- Load checkpoint ----------
    ckpt_path = Path("model") / f"diff-params-ARGS={arg_num}" / "params-final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first with: python diffusion_training.py {args_path.name}"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ---------- Build model ----------
    in_channels = args.get("channels", 1)
    if in_channels == "" or in_channels is None:
        in_channels = 1
    in_channels = int(in_channels)

    attention_resolutions = args.get("attention_resolutions", "32,16,8")

    unet_ema = UNetModel(
        args["img_size"][0],
        args["base_channels"],
        channel_mults=args["channel_mults"],
        dropout=args.get("dropout", 0),
        n_heads=args.get("num_heads", 1),
        n_head_channels=args.get("num_head_channels", -1),
        attention_resolutions=attention_resolutions,
        in_channels=in_channels,
        group=group_equiv,
        use_attention=use_attention
    ).to(device)

    if "ema" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} missing key 'ema'. Keys: {list(ckpt.keys())}")
    unet_ema.load_state_dict(ckpt["ema"], strict=True)
    unet_ema.eval()

    # ---------- Diffusion wrapper ----------
    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    diffusion = GaussianDiffusionModel(
        args["img_size"],
        betas,
        img_channels=in_channels,
        loss_type=args.get("loss-type", "l2"),
        loss_weight=args.get("loss_weight", "none"),
        noise=args.get("noise_fn", "gauss"),
        group=group_equiv,
        group_fa=group_fa,
        group_fa_train=group_fa_train,
        group_en=group_en
    )

    # ---------- Optional unconditional sampling ----------
    denoise_mode = str(args_cli.denoise_mode).strip()
    if args_cli.save_samples:
        base = Path("diffusion-samples") / f"ARGS={arg_num}"
        if args_cli.sample_dir is not None and str(args_cli.sample_dir).strip() != "":
            samp_dir = base / str(args_cli.sample_dir).strip()
        else:
            # default folder includes t and denoise mode for clarity
            st = args_cli.sample_t if args_cli.sample_t is not None else int(args["T"])
            samp_dir = base / f"samples_t{st}_{denoise_mode}"
        run_unconditional_sampling(
            diffusion=diffusion,
            unet_ema=unet_ema,
            args=args,
            device=device,
            in_channels=in_channels,
            out_dir=samp_dir,
            denoise_mode=denoise_mode,
            num_samples=int(args_cli.num_samples),
            sample_batch=int(args_cli.sample_batch),
            sample_t=args_cli.sample_t if args_cli.sample_t is not None else int(args["T"]),
        )

    # ---------- Data for scoring ----------
    data_root = args.get("data_root", "./data")
    img_size = tuple(args["img_size"])

    test_normal_dir = os.path.join(data_root, "test", "normal")
    test_abnormal_dir = os.path.join(data_root, "test", "abnormal")

    if not os.path.isdir(test_normal_dir):
        raise FileNotFoundError(f"Missing folder: {test_normal_dir}")
    if not os.path.isdir(test_abnormal_dir):
        raise FileNotFoundError(f"Missing folder: {test_abnormal_dir}")

    ds_normal = dataset.SimpleImageFolder(
        root_dir=test_normal_dir,
        img_size=img_size,
        channels=in_channels,
        augment=False,
        label=0,
    )
    ds_abnormal = dataset.SimpleImageFolder(
        root_dir=test_abnormal_dir,
        img_size=img_size,
        channels=in_channels,
        augment=False,
        label=1,
    )

    full_ds = torch.utils.data.ConcatDataset([ds_normal, ds_abnormal])
    loader = torch.utils.data.DataLoader(full_ds, batch_size=args_cli.batch, shuffle=False, num_workers=0)

    detect_t = int(args_cli.detect_t) if args_cli.detect_t is not None else int(args.get("sample_distance", args["T"]))

    # ---------- Outputs ----------
    out_dir = Path("metrics") / f"ARGS={arg_num}"
    ensure_dir(out_dir)
    csv_path = out_dir / "cells_scores.csv"

    ex_dir = Path("diffusion-training-images") / f"ARGS={arg_num}" / "cells_examples"
    if args_cli.save_examples:
        ensure_dir(ex_dir)

    rows = []
    saved = 0

    # ---------- Inference (reconstruction for anomaly score) ----------
    for batch in loader:
        x = batch["image"].to(device)                 # [B,C,H,W] in [-1,1]
        y = batch["label"].cpu().numpy().astype(int)  # [B]
        fns = batch["filenames"]                      # list[str]

        with torch.no_grad():
            x_recon = diffusion.forward_backward(
                unet_ema,
                x,
                see_whole_sequence=None,
                t_distance=detect_t,
                denoise_fn=denoise_mode,
            )

        mse_map = (x - x_recon).pow(2)
        scores = mse_map.mean(dim=(1, 2, 3)).detach().cpu().numpy()

        for i in range(x.shape[0]):
            rows.append((str(fns[i]), int(y[i]), float(scores[i])))

            if args_cli.save_examples and saved < args_cli.num_examples:
                xin = to_uint8(x[i])
                xrc = to_uint8(x_recon[i])
                hm = mse_map[i].mean(dim=0).detach().cpu().numpy()

                plt.figure(figsize=(10, 3))
                plt.subplot(1, 3, 1)
                plt.title("input")
                plt.axis("off")
                plt.imshow(xin if in_channels == 3 else xin.squeeze(), cmap=None if in_channels == 3 else "gray")

                plt.subplot(1, 3, 2)
                plt.title("recon")
                plt.axis("off")
                plt.imshow(xrc if in_channels == 3 else xrc.squeeze(), cmap=None if in_channels == 3 else "gray")

                plt.subplot(1, 3, 3)
                plt.title("mse")
                plt.axis("off")
                plt.imshow(hm)

                plt.tight_layout()
                plt.savefig(ex_dir / f"example_{saved:04d}_label={int(y[i])}_score={scores[i]:.6f}.png", dpi=200)
                plt.close()
                saved += 1

    # ---------- Save CSV ----------
    with open(csv_path, "w") as f:
        f.write("filename,true_label,score\n")
        for fn, yy, ss in rows:
            f.write(f"{fn},{yy},{ss}\n")

    # optional AUC print
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.array([r[1] for r in rows])
        y_score = np.array([r[2] for r in rows])
        print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
    except Exception:
        pass

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()