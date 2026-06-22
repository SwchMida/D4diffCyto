# cells_detection_phase2.py
"""
Phase-2 inference only (VERY DETAILED PROGRESS + SAFE CHECKPOINTING)

EN CONSISTENCY GUARANTEE (IMPORTANT):
- q-step noise EN: applied when creating x_t (forward noising) in BOTH paths:
    (1) normal path via diffusion.forward_backward() (handled inside GaussianDiffusion.py)
    (2) logging path via reconstruct_with_logging() (FIXED here to call diffusion._maybe_align_noise)
- p-step noise EN: applied inside diffusion.sample_p() (handled inside GaussianDiffusion.py)
So: with --group_en 1, EN is consistent across q-noise and p-noise, and across logging vs non-logging paths.
"""

import os
import json
import time
import random
import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

import dataset
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from UNet import UNetModel


# ----------------------------
# Utilities
# ----------------------------
def now() -> float:
    return time.time()


def fmt_sec(s: float) -> str:
    s = max(float(s), 0.0)
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = int(s % 60)
    if hh > 0:
        return f"{hh:d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_device_str() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    return f"cuda:{i} ({name}, {total:.1f} GB)"


def cuda_mem_str() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / (1024**3)
    reserv = torch.cuda.memory_reserved(dev) / (1024**3)
    max_alloc = torch.cuda.max_memory_allocated(dev) / (1024**3)
    max_res = torch.cuda.max_memory_reserved(dev) / (1024**3)
    return f"alloc={alloc:.2f}G reserv={reserv:.2f}G | max_alloc={max_alloc:.2f}G max_res={max_res:.2f}G"


def to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """Tensor in [-1,1] CHW -> uint8 HWC (or HW)."""
    img = (img_t.clamp(-1, 1) + 1.0) * 127.5
    img = img.to(torch.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        return img.permute(1, 2, 0).cpu().numpy()
    return img.cpu().numpy()


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


def wrangle_int(v: Any, default: int) -> int:
    if v is None or v == "":
        return default
    return int(v)


def wrangle_float(v: Any, default: float) -> float:
    if v is None or v == "":
        return default
    return float(v)


def wrangle_str(v: Any, default: str) -> str:
    if v is None:
        return default
    return str(v)


def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by fn (by signature)."""
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def write_scores_csv(path: Path, rows: List[Tuple[str, int, float]]):
    with open(path, "w") as f:
        f.write("filename,true_label,score\n")
        for fn, yy, ss in rows:
            f.write(f"{fn},{yy},{ss}\n")


def load_partial_rows(path: Path) -> List[Tuple[str, int, float]]:
    """
    Load partial CSV into rows. Keeps LAST occurrence of each filename.
    """
    if not path.exists():
        return []
    rows = []
    seen = {}
    with open(path, "r") as f:
        _ = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            fn = ",".join(parts[:-2])
            yy = int(parts[-2])
            ss = float(parts[-1])
            seen[fn] = (fn, yy, ss)

    with open(path, "r") as f:
        _ = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            fn = ",".join(parts[:-2])
            if fn in seen:
                rows.append(seen.pop(fn))
    for v in seen.values():
        rows.append(v)
    return rows


# ----------------------------
# UNet strict loader (robust)
# ----------------------------
def build_unet_that_loads_strict(
    ckpt_ema_state: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    img_size0: int,
    base_channels: int,
    channel_mults: Any,
    dropout: float,
    n_heads: int,
    n_head_channels: int,
    in_channels: int,
    attention_resolutions_candidates: List[str],
    extra_unet_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Search over common UNet configs and require load_state_dict(strict=True) to succeed.
    """
    extra_unet_kwargs = extra_unet_kwargs or {}

    num_res_blocks_cands = [2, 3, 1, 4]
    biggan_updown_cands = [True, False]
    conv_resample_cands = [True, False]

    last_err: Optional[str] = None

    for attn_res in attention_resolutions_candidates:
        attn_res = str(attn_res).strip()
        if attn_res == "":
            continue

        for num_res_blocks in num_res_blocks_cands:
            for biggan_updown in biggan_updown_cands:
                for conv_resample in conv_resample_cands:
                    try:
                        unet_kwargs = dict(
                            img_size=img_size0,
                            base_channels=base_channels,
                            conv_resample=conv_resample,
                            n_heads=n_heads,
                            n_head_channels=n_head_channels,
                            channel_mults=channel_mults,
                            num_res_blocks=num_res_blocks,
                            dropout=dropout,
                            attention_resolutions=attn_res,
                            biggan_updown=biggan_updown,
                            in_channels=in_channels,
                        )
                        unet_kwargs.update(extra_unet_kwargs)
                        unet_kwargs = filter_kwargs_for_callable(UNetModel.__init__, unet_kwargs)

                        m = UNetModel(**unet_kwargs).to(device)
                        m.load_state_dict(ckpt_ema_state, strict=True)
                        m.eval()

                        if verbose:
                            print(
                                "[PHASE2] UNet match FOUND:",
                                f"attn_res='{attn_res}' num_res_blocks={num_res_blocks} "
                                f"biggan_updown={biggan_updown} conv_resample={conv_resample}",
                                flush=True,
                            )
                        return m, dict(
                            attention_resolutions=attn_res,
                            num_res_blocks=num_res_blocks,
                            biggan_updown=biggan_updown,
                            conv_resample=conv_resample,
                            **{k: extra_unet_kwargs[k] for k in extra_unet_kwargs.keys()},
                        )

                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"
                        continue

    print("[PHASE2][ERROR] Could not find a UNet config that loads EMA strictly.", flush=True)
    if last_err:
        print("[PHASE2][ERROR] Last error:", flush=True)
        print(last_err, flush=True)
    raise RuntimeError("UNet strict load failed for all tried configurations.")


# ----------------------------
# EN-consistent q-noise helper for logging path
# ----------------------------
@torch.no_grad()
def make_xt_with_en_consistent_qnoise(
    diffusion: GaussianDiffusionModel,
    x0: torch.Tensor,
    t_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Create x_t = q(x0, t, noise) using diffusion.noise_fn, and if EN is enabled in diffusion,
    align q-noise via diffusion._maybe_align_noise(noise, x0).

    This matches the behavior inside GaussianDiffusion.forward_backward() when group_en is enabled.
    """
    n = diffusion.noise_fn(x0, t_tensor).float()
    # EN alignment for q-noise (only if available/enabled inside diffusion)
    if hasattr(diffusion, "_maybe_align_noise"):
        try:
            n = diffusion._maybe_align_noise(n, x0)
        except Exception:
            # if diffusion doesn't support alignment for some reason, fall back safely
            pass
    return diffusion.sample_q(x0, t_tensor, n)


# ----------------------------
# Per-t logging reconstruction (EN-consistent)
# ----------------------------
@torch.no_grad()
def reconstruct_with_logging(
    diffusion: GaussianDiffusionModel,
    model: torch.nn.Module,
    x0: torch.Tensor,
    detect_t: int,
    denoise_mode: str,
    *,
    log_t_every: int = 0,
    log_t_all_batches: bool = False,
    batch_idx: int = 0,
    show_mem: bool = False,
    sync_timing: bool = False,
) -> torch.Tensor:
    device = x0.device
    B = x0.shape[0]
    if detect_t <= 0:
        return x0.detach()

    # Create x_{t} by forward noising (q-step) WITH EN alignment (if enabled)
    t_tensor = torch.tensor([detect_t - 1], device=device).repeat(B)
    x = make_xt_with_en_consistent_qnoise(diffusion, x0, t_tensor)

    if sync_timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = now()

    # Reverse steps: EN for p-step noise is handled INSIDE diffusion.sample_p() (GaussianDiffusion.py)
    for t in range(int(detect_t) - 1, -1, -1):
        t_batch = torch.tensor([t], device=device).repeat(B)
        out = diffusion.sample_p(model, x, t_batch, denoise_fn=denoise_mode)
        x = out["sample"]

        if log_t_every and (t % log_t_every == 0) and (log_t_all_batches or batch_idx == 0):
            if sync_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = now() - t0
            msg = f"[t-step] batch={batch_idx} t={t:4d}/{detect_t-1} | dt={dt:.2f}s"
            if show_mem:
                msg += f" | {cuda_mem_str()}"
            print(msg, flush=True)

    return x.detach()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("args", help="args spec: 90 | args90.json | args90 | args90_phase2_fa_en.json")
    ap.add_argument("--ckpt_argnum", type=str, default=None, help="Override checkpoint ARG number.")

    # core inference
    ap.add_argument("--detect_t", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--pin_memory", action="store_true")

    # group knobs
    ap.add_argument("--group", type=str, default=None)
    ap.add_argument("--group_fa", type=int, default=None)  # 0/1
    ap.add_argument("--group_en", type=int, default=None)  # 0/1

    ap.add_argument("--denoise_mode", type=str, default=None)
    ap.add_argument("--seed", type=int, default=7)

    # logging
    ap.add_argument("--log_batch_every", type=int, default=None)
    ap.add_argument("--log_t_every", type=int, default=None)
    ap.add_argument("--log_t_all_batches", action="store_true")
    ap.add_argument("--show_mem", action="store_true")
    ap.add_argument("--sync_timing", action="store_true")

    # checkpointing
    ap.add_argument("--write_every_batches", type=int, default=5,
                    help="Write partial CSV/progress every N batches (default: 5). Set 0 to disable.")
    ap.add_argument("--resume", action="store_true",
                    help="If partial CSV exists, load it and skip already processed filenames.")

    # output
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--out_csv_name", type=str, default=None)

    # examples
    ap.add_argument("--save_examples", action="store_true")
    ap.add_argument("--num_examples", type=int, default=16)

    # data
    ap.add_argument("--data_root", type=str, default=None)

    args_cli = ap.parse_args()
    seed_everything(args_cli.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load JSON ----
    args_path = resolve_args_path(args_cli.args)
    with open(args_path, "r") as f:
        args_json: Dict[str, Any] = json.load(f)

    argsfile_argnum = argnum_from_argsfile(args_path)
    ckpt_argnum = args_cli.ckpt_argnum if args_cli.ckpt_argnum is not None else args_json.get("ckpt_argnum", argsfile_argnum)

    print("=" * 100, flush=True)
    print(f"[PHASE2] repo_root={Path('.').resolve()}", flush=True)
    print(f"[PHASE2] device={cuda_device_str()}", flush=True)
    print(f"[PHASE2] torch={torch.__version__} | numpy={np.__version__}", flush=True)
    print(f"[PHASE2] argsfile={args_path} | argsfile_argnum={argsfile_argnum} | ckpt_argnum={ckpt_argnum}", flush=True)
    print("=" * 100, flush=True)

    # ---- Group knobs ----
    group = args_cli.group if args_cli.group is not None else args_json.get("group", "none")
    group_fa = bool(int(args_cli.group_fa)) if args_cli.group_fa is not None else bool(args_json.get("group_fa", False))
    group_en = bool(int(args_cli.group_en)) if args_cli.group_en is not None else bool(args_json.get("group_en", False))

    detect_t = int(args_cli.detect_t) if args_cli.detect_t is not None else int(
        args_json.get("detect_t", args_json.get("sample_distance", args_json.get("T", 1000)))
    )
    batch = int(args_cli.batch) if args_cli.batch is not None else int(args_json.get("batch", 16))
    num_workers = int(args_cli.num_workers) if args_cli.num_workers is not None else int(args_json.get("num_workers", 0))
    pin_memory = bool(args_cli.pin_memory) or bool(args_json.get("pin_memory", False))
    denoise_mode = args_cli.denoise_mode if args_cli.denoise_mode is not None else args_json.get("denoise_mode", "noise_fn")

    log_t_every = int(args_cli.log_t_every) if args_cli.log_t_every is not None else int(args_json.get("log_t_every", 0))
    log_batch_every = int(args_cli.log_batch_every) if args_cli.log_batch_every is not None else int(args_json.get("log_batch_every", 1))

    print(f"[PHASE2] group={group} | FA={group_fa} | EN={group_en}", flush=True)
    print(f"[PHASE2] detect_t={detect_t} | denoise_mode={denoise_mode}", flush=True)
    print(f"[PHASE2] seed={args_cli.seed} | batch={batch} | num_workers={num_workers} | pin_memory={pin_memory}", flush=True)
    print(f"[PHASE2] log_t_every={log_t_every} | log_batch_every={log_batch_every} | show_mem={args_cli.show_mem} | sync_timing={args_cli.sync_timing}", flush=True)
    print(f"[PHASE2] write_every_batches={args_cli.write_every_batches} | resume={args_cli.resume}", flush=True)

    # ---- Load checkpoint ----
    ckpt_path = Path("model") / f"diff-params-ARGS={ckpt_argnum}" / "params-final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    if "ema" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'ema'. Keys present: {list(ckpt.keys())}")

    # ---- UNet knobs (prefer checkpoint) ----
    img_size = ckpt_args.get("img_size", args_json.get("img_size", [256, 256]))
    if isinstance(img_size, (tuple, list)):
        img_size0 = int(img_size[0])
    else:
        img_size0 = int(img_size)

    base_channels = wrangle_int(ckpt_args.get("base_channels", args_json.get("base_channels", 64)), 64)
    channel_mults = ckpt_args.get("channel_mults", args_json.get("channel_mults", ""))
    dropout = wrangle_float(ckpt_args.get("dropout", args_json.get("dropout", 0.0)), 0.0)
    n_heads = wrangle_int(ckpt_args.get("num_heads", args_json.get("num_heads", 1)), 1)
    n_head_channels = wrangle_int(ckpt_args.get("num_head_channels", args_json.get("num_head_channels", -1)), -1)

    in_channels = ckpt_args.get("channels", args_json.get("channels", 1))
    in_channels = 1 if (in_channels is None or in_channels == "") else int(in_channels)

    extra_unet_kwargs = {}
    for k in ["group", "group_equiv", "use_attention", "use_attn", "equiv_attn", "attn_equiv"]:
        if k in ckpt_args:
            extra_unet_kwargs[k] = ckpt_args[k]
        elif k in args_json:
            extra_unet_kwargs[k] = args_json[k]

    attn_from_ckpt = wrangle_str(ckpt_args.get("attention_resolutions", ""), "")
    attn_from_json = wrangle_str(args_json.get("attention_resolutions", ""), "")
    attention_candidates: List[str] = []
    for s in [attn_from_ckpt, attn_from_json, "32,16,8", "16,8", "32,16", "8"]:
        s = str(s).strip()
        if s and s not in attention_candidates:
            attention_candidates.append(s)

    print(
        f"[PHASE2] UNet search base: img={img_size0} base={base_channels} ch_mults={channel_mults} "
        f"heads={n_heads} head_ch={n_head_channels} dropout={dropout} in_ch={in_channels}",
        flush=True,
    )
    print(f"[PHASE2] attention_resolutions candidates: {attention_candidates}", flush=True)
    if extra_unet_kwargs:
        print(f"[PHASE2] extra UNet kwargs (filtered by signature): {extra_unet_kwargs}", flush=True)

    unet_ema, matched_cfg = build_unet_that_loads_strict(
        ckpt["ema"],
        device=device,
        img_size0=img_size0,
        base_channels=base_channels,
        channel_mults=channel_mults,
        dropout=dropout,
        n_heads=n_heads,
        n_head_channels=n_head_channels,
        in_channels=in_channels,
        attention_resolutions_candidates=attention_candidates,
        extra_unet_kwargs=extra_unet_kwargs,
        verbose=True,
    )

    # ---- Diffusion wrapper ----
    T = wrangle_int(ckpt_args.get("T", args_json.get("T", 1000)), 1000)
    beta_schedule = wrangle_str(ckpt_args.get("beta_schedule", args_json.get("beta_schedule", "linear")), "linear")
    loss_type = wrangle_str(ckpt_args.get("loss-type", args_json.get("loss-type", "l2")), "l2")
    loss_weight = wrangle_str(ckpt_args.get("loss_weight", args_json.get("loss_weight", "none")), "none")
    noise_fn = wrangle_str(ckpt_args.get("noise_fn", args_json.get("noise_fn", "simplex")), "simplex")

    betas = get_beta_schedule(T, beta_schedule)

    diffusion_kwargs = dict(
        img_size=img_size if isinstance(img_size, (list, tuple)) else [img_size0, img_size0],
        betas=betas,
        img_channels=in_channels,
        loss_type=loss_type,
        loss_weight=loss_weight,
        noise=noise_fn,
        group=group,
        group_fa=group_fa,
        group_fa_train=False,
        group_en=group_en,
    )
    diffusion_kwargs = filter_kwargs_for_callable(GaussianDiffusionModel.__init__, diffusion_kwargs)
    diffusion = GaussianDiffusionModel(**diffusion_kwargs)

    print(f"[PHASE2] matched UNet cfg: {matched_cfg}", flush=True)
    print(f"[PHASE2] diffusion: T={T} beta_schedule={beta_schedule} noise_fn={noise_fn} loss={loss_type} loss_weight={loss_weight}", flush=True)

    # ---- Dataset ----
    data_root = args_cli.data_root if args_cli.data_root is not None else args_json.get("data_root", ckpt_args.get("data_root", "./data"))
    test_normal_dir = os.path.join(data_root, "test", "normal")
    test_abnormal_dir = os.path.join(data_root, "test", "abnormal")

    if not os.path.isdir(test_normal_dir):
        raise FileNotFoundError(f"Missing folder: {test_normal_dir}")
    if not os.path.isdir(test_abnormal_dir):
        raise FileNotFoundError(f"Missing folder: {test_abnormal_dir}")

    ds_normal = dataset.SimpleImageFolder(test_normal_dir, tuple(img_size), channels=in_channels, augment=False, label=0)
    ds_abnormal = dataset.SimpleImageFolder(test_abnormal_dir, tuple(img_size), channels=in_channels, augment=False, label=1)
    full_ds = torch.utils.data.ConcatDataset([ds_normal, ds_abnormal])

    loader = torch.utils.data.DataLoader(
        full_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    n_imgs = len(full_ds)
    n_batches = len(loader)
    print(f"[PHASE2] data_root={data_root}", flush=True)
    print(f"[PHASE2] test/normal={len(ds_normal)} | test/abnormal={len(ds_abnormal)} | total={n_imgs} | batches={n_batches}", flush=True)

    # ---- Outputs ----
    out_dir = Path(args_cli.out_dir) if args_cli.out_dir else (Path("metrics") / f"ARGS={ckpt_argnum}")
    ensure_dir(out_dir)

    tag = f"group={group}_fa={int(group_fa)}_en={int(group_en)}_t={detect_t}_den={denoise_mode}"
    if args_cli.tag:
        tag = f"{tag}__{args_cli.tag}"

    final_csv = out_dir / (args_cli.out_csv_name if args_cli.out_csv_name else f"cells_scores__{tag}.csv")
    partial_csv = final_csv.with_suffix(".partial.csv")
    progress_json = final_csv.with_suffix(".progress.json")

    ex_dir = Path("diffusion-training-images") / f"ARGS={ckpt_argnum}" / "cells_examples_phase2"
    if args_cli.save_examples:
        ensure_dir(ex_dir)

    print(f"[PHASE2] final_csv    -> {final_csv}", flush=True)
    print(f"[PHASE2] partial_csv  -> {partial_csv}", flush=True)
    print(f"[PHASE2] progress_json-> {progress_json}", flush=True)

    # ---- Resume (optional) ----
    rows: List[Tuple[str, int, float]] = []
    done_filenames = set()

    if args_cli.resume and partial_csv.exists():
        print(f"[PHASE2] RESUME: loading partial CSV: {partial_csv}", flush=True)
        rows = load_partial_rows(partial_csv)
        done_filenames = {r[0] for r in rows}
        print(f"[PHASE2] RESUME: loaded {len(rows)} rows, will skip already-scored filenames.", flush=True)

    # ---- Reset CUDA peak stats ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    saved = 0
    start_all = now()
    processed = 0
    skipped = 0

    # ---- Inference loop ----
    for bidx, batch_data in enumerate(loader):
        b_start = now()

        # batch parsing
        if isinstance(batch_data, dict):
            x = batch_data["image"]
            y = np.array(batch_data["label"], dtype=int)
            fns = list(batch_data["filenames"])
        else:
            x, y, fns = batch_data
            y = y.cpu().numpy().astype(int)
            fns = list(fns)

        # resume skipping at filename granularity
        keep_idx = [i for i, fn in enumerate(fns) if str(fn) not in done_filenames]
        if len(keep_idx) == 0:
            skipped += len(fns)
            if log_batch_every > 0 and ((bidx % log_batch_every) == 0 or (bidx == n_batches - 1)):
                elapsed = now() - start_all
                msg = f"[batch] {bidx+1:4d}/{n_batches} | SKIP(all) {len(fns)} | elapsed {fmt_sec(elapsed)}"
                if args_cli.show_mem:
                    msg += f" | {cuda_mem_str()}"
                print(msg, flush=True)
            continue

        if len(keep_idx) != len(fns):
            x = x[keep_idx]
            y = y[keep_idx]
            fns = [fns[i] for i in keep_idx]

        if pin_memory and torch.cuda.is_available() and hasattr(x, "pin_memory"):
            x = x.pin_memory()
        x = x.to(device, non_blocking=True)

        # reconstruction timing
        if args_cli.sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        recon_start = now()

        if log_t_every and (args_cli.log_t_all_batches or bidx == 0):
            # logging path: q-noise EN is handled by reconstruct_with_logging() (fixed)
            x_recon = reconstruct_with_logging(
                diffusion,
                unet_ema,
                x,
                detect_t=detect_t,
                denoise_mode=denoise_mode,
                log_t_every=log_t_every,
                log_t_all_batches=args_cli.log_t_all_batches,
                batch_idx=bidx,
                show_mem=args_cli.show_mem,
                sync_timing=args_cli.sync_timing,
            )
        else:
            # non-logging path: q-noise EN + p-noise EN are handled inside GaussianDiffusion.forward_backward/sample_p
            x_recon = diffusion.forward_backward(
                unet_ema,
                x,
                see_whole_sequence=None,
                t_distance=detect_t,
                denoise_fn=denoise_mode,
            )

        if args_cli.sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        recon_s = now() - recon_start

        mse_map = (x - x_recon).pow(2)
        scores = mse_map.mean(dim=(1, 2, 3)).detach().cpu().numpy()

        # store
        for i in range(x.shape[0]):
            fn = str(fns[i])
            rows.append((fn, int(y[i]), float(scores[i])))
            done_filenames.add(fn)

            if args_cli.save_examples and saved < int(args_cli.num_examples):
                xin = to_uint8(x[i].detach().cpu())
                xrc = to_uint8(x_recon[i].detach().cpu())
                hm = mse_map[i].mean(dim=0).detach().cpu().numpy()

                plt.figure(figsize=(10, 3))
                plt.subplot(1, 3, 1); plt.title("input"); plt.axis("off")
                plt.imshow(xin if in_channels == 3 else xin.squeeze(), cmap=None if in_channels == 3 else "gray")
                plt.subplot(1, 3, 2); plt.title("recon"); plt.axis("off")
                plt.imshow(xrc if in_channels == 3 else xrc.squeeze(), cmap=None if in_channels == 3 else "gray")
                plt.subplot(1, 3, 3); plt.title("mse"); plt.axis("off")
                plt.imshow(hm)
                plt.tight_layout()
                plt.savefig(ex_dir / f"ex_{saved:04d}__{tag}__label={int(y[i])}__score={scores[i]:.6f}.png", dpi=200)
                plt.close()
                saved += 1

        processed += int(x.shape[0])

        if args_cli.sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_total_s = now() - b_start

        if log_batch_every > 0 and ((bidx % log_batch_every) == 0 or (bidx == n_batches - 1)):
            elapsed = now() - start_all
            it_per_sec = processed / max(elapsed, 1e-9)
            remaining = n_imgs - processed
            eta = remaining / max(it_per_sec, 1e-9)

            ms_per_step = (recon_s / max(detect_t, 1)) * 1000.0
            ms_per_img = (recon_s / max(int(x.shape[0]), 1)) * 1000.0

            msg = (
                f"[batch] {bidx+1:4d}/{n_batches} | "
                f"processed {processed:5d}/{n_imgs} | "
                f"{it_per_sec:.3f} img/s | elapsed {fmt_sec(elapsed)} | ETA {fmt_sec(eta)} | "
                f"batch_total={batch_total_s:.2f}s recon={recon_s:.2f}s "
                f"({ms_per_step:.2f} ms/step, {ms_per_img:.2f} ms/img)"
            )
            if args_cli.show_mem:
                msg += f" | {cuda_mem_str()}"
            print(msg, flush=True)

        # periodic checkpoint writes
        if args_cli.write_every_batches and args_cli.write_every_batches > 0:
            if (bidx + 1) % int(args_cli.write_every_batches) == 0:
                tmp = {}
                for fn, yy, ss in rows:
                    if fn not in tmp:
                        tmp[fn] = (fn, yy, ss)
                    else:
                        if ss > tmp[fn][2]:
                            tmp[fn] = (fn, yy, ss)
                rows = list(tmp.values())

                write_scores_csv(partial_csv, rows)
                prog = dict(
                    ckpt_argnum=str(ckpt_argnum),
                    argsfile=str(args_path),
                    tag=tag,
                    detect_t=int(detect_t),
                    denoise_mode=str(denoise_mode),
                    group=str(group),
                    group_fa=bool(group_fa),
                    group_en=bool(group_en),
                    batch=int(batch),
                    processed=int(processed),
                    total=int(n_imgs),
                    batches=int(n_batches),
                    last_batch=int(bidx + 1),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    cuda_mem=cuda_mem_str() if args_cli.show_mem else None,
                )
                with open(progress_json, "w") as f:
                    json.dump(prog, f, indent=2)
                print(f"[PHASE2] checkpoint write -> {partial_csv.name} (+ progress.json)", flush=True)

    # final de-dup (max score per filename)
    tmp = {}
    for fn, yy, ss in rows:
        if fn not in tmp:
            tmp[fn] = (fn, yy, ss)
        else:
            if ss > tmp[fn][2]:
                tmp[fn] = (fn, yy, ss)
    rows = list(tmp.values())

    write_scores_csv(final_csv, rows)

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.array([r[1] for r in rows], dtype=int)
        y_score = np.array([r[2] for r in rows], dtype=float)
        auc = roc_auc_score(y_true, y_score)
        print(f"[PHASE2] ROC-AUC: {auc:.6f}", flush=True)
    except Exception as e:
        print(f"[PHASE2] (AUC skipped) {type(e).__name__}: {e}", flush=True)

    if args_cli.show_mem and torch.cuda.is_available():
        print(f"[PHASE2] CUDA peak: {cuda_mem_str()}", flush=True)

    print(f"[PHASE2] Saved final CSV: {final_csv}", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()