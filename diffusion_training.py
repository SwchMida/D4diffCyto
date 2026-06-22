import os
import json
import sys
import time
import copy
import collections
from random import seed

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from torch import optim

import dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from helpers import *
from UNet import UNetModel, update_ema_params

torch.cuda.empty_cache()

ROOT_DIR = "./"


def train(training_dataset_loader, testing_dataset_loader, args, resume):
    """
    :param training_dataset_loader: cycle(dataloader) instance for training
    :param testing_dataset_loader:  cycle(dataloader) instance for testing
    :param args: dictionary of parameters
    :param resume: dictionary of parameters if continuing training from checkpoint
    """

    # ----------------------------
    # channels
    # ----------------------------
    in_channels = 1
    if args["dataset"].lower() in ("cifar", "leather"):
        in_channels = 3

    if args.get("channels", "") != "":
        in_channels = int(args["channels"])

    # ----------------------------
    # group-equivariance controls (args90-compatible defaults)
    # ----------------------------
    group_equiv = args.get("group", "none")  # 'none' | 'C4' | 'D4'
    group_fa = bool(args.get("group_fa", False))  # frame averaging at inference
    group_en = bool(args.get("group_en", False))  # equivariant noise alignment
    group_fa_train = bool(args.get("group_fa_train", False))  # FA during training (slow)
    use_attention = args.get("use_attention", None)  # None -> UNet decides default; or True/False explicitly

    print(
        f"[CONFIG] in_channels={in_channels} | group={group_equiv} | "
        f"group_fa={group_fa} | group_en={group_en} | group_fa_train={group_fa_train} | "
        f"use_attention={use_attention}",
        flush=True
    )

    # ----------------------------
    # model
    # ----------------------------
    model = UNetModel(
        args["img_size"][0],
        args["base_channels"],
        channel_mults=args["channel_mults"],
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        n_head_channels=args["num_head_channels"],
        attention_resolutions=args.get("attention_resolutions", "32,16,8"),
        in_channels=in_channels,
        group=group_equiv,
        use_attention=use_attention
    )

    betas = get_beta_schedule(args["T"], args["beta_schedule"])

    diffusion = GaussianDiffusionModel(
        args["img_size"],
        betas,
        loss_weight=args["loss_weight"],
        loss_type=args["loss-type"],
        noise=args["noise_fn"],
        img_channels=in_channels,
        group=group_equiv,
        group_fa=group_fa,
        group_fa_train=group_fa_train,
        group_en=group_en
    )

    # ----------------------------
    # resume / ema
    # ----------------------------
    if resume:
        # resume dict saved by save() contains model_state_dict + ema
        if "model_state_dict" in resume:
            model.load_state_dict(resume["model_state_dict"])
        elif "unet" in resume:  # backward compat
            model.load_state_dict(resume["unet"])
        else:
            # some checkpoints might only have ema
            model.load_state_dict(resume["ema"])

        ema = UNetModel(
            args["img_size"][0],
            args["base_channels"],
            channel_mults=args["channel_mults"],
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args["num_head_channels"],
            attention_resolutions=args.get("attention_resolutions", "32,16,8"),
            in_channels=in_channels,
            group=group_equiv,
            use_attention=use_attention
        )
        ema.load_state_dict(resume["ema"])
        start_epoch = int(resume.get("n_epoch", 0))
    else:
        start_epoch = 0
        ema = copy.deepcopy(model)

    model.to(device)
    ema.to(device)

    optimiser = optim.AdamW(
        model.parameters(),
        lr=float(args["lr"]),
        weight_decay=float(args["weight_decay"]),
        betas=(0.9, 0.999)
    )
    if resume and "optimizer_state_dict" in resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])

    del resume

    # ----------------------------
    # logging / runtime controls
    # ----------------------------
    # prints every N steps (default 50)
    print_every = int(args.get("print_every", 50))

    # skip expensive total VLB completely (default True for large datasets)
    skip_vlb = bool(args.get("skip_vlb", True))

    # if not skipping, compute VLB only every N epochs (default 50)
    vlb_every = int(args.get("vlb_every", 50))

    # steps per epoch
    steps_per_epoch = int(args.get("steps_per_epoch", 100 // max(int(args["Batch_Size"]), 1)))
    iters = range(steps_per_epoch) if args["dataset"].lower() != "cifar" else range(200)

    # epoch loop
    tqdm_epoch = range(start_epoch, int(args["EPOCHS"]))

    start_time = time.time()
    losses = []
    vlb_hist = collections.deque([], maxlen=10)

    for epoch in tqdm_epoch:
        mean_loss = []

        for step in iters:
            data = next(training_dataset_loader)

            if args["dataset"].lower() == "cifar":
                x = data[0].to(device)
            else:
                x = data["image"].to(device)

            # IMPORTANT: diffusion.p_loss returns:
            # (loss_scalar_tensor, (loss_dict, x_t, eps_t))
            loss_scalar, estimates = diffusion.p_loss(model, x, args)

            # progress print
            if (step % print_every) == 0:
                print(
                    f"[epoch {epoch+1}/{args['EPOCHS']}] step {step}/{steps_per_epoch} "
                    f"loss={loss_scalar.item():.6f}",
                    flush=True
                )

            noisy = estimates[1]
            est = estimates[2]

            optimiser.zero_grad(set_to_none=True)
            loss_scalar.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            update_ema_params(ema, model)

            mean_loss.append(loss_scalar.detach().cpu().item())

            # occasional visual outputs
            if epoch % 50 == 0 and step == 0:
                row_size = min(8, int(args["Batch_Size"]))
                training_outputs(
                    diffusion,
                    x,
                    est,
                    noisy,
                    epoch,
                    row_size,
                    save_imgs=args["save_imgs"],
                    save_vids=args["save_vids"],
                    ema=ema,
                    args=args
                )

        # epoch end bookkeeping
        losses.append(float(np.mean(mean_loss)) if len(mean_loss) else 0.0)

        # epoch time estimate
        time_taken = time.time() - start_time
        done_epochs = (epoch + 1 - start_epoch)
        time_per_epoch = time_taken / max(done_epochs, 1)
        remaining_epochs = int(args["EPOCHS"]) - (epoch + 1)
        est_remaining_sec = remaining_epochs * time_per_epoch
        hh = int(est_remaining_sec // 3600)
        mm = int((est_remaining_sec % 3600) // 60)

        # expensive VLB computation (optional)
        if (not skip_vlb) and (epoch % vlb_every == 0):
            # uses last x from epoch; ok for monitoring
            vlb_terms = diffusion.calc_total_vlb(x, model, args)
            vlb_val = vlb_terms["total_vlb"].mean(dim=-1).cpu().item()
            vlb_hist.append(vlb_val)

            print(
                f"epoch: {epoch}, most recent total VLB: {vlb_hist[-1]:.6f} "
                f"mean total VLB: {np.mean(vlb_hist):.4f}, "
                f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, "
                f"vb: {torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, "
                f"x_0_mse: {torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, "
                f"mse: {torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f} "
                f"time elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) // 60):02d}, "
                f"est time remaining: {hh}:{mm:02d}",
                flush=True
            )
        else:
            # cheap epoch summary always
            print(
                f"epoch: {epoch} done | mean loss: {losses[-1]:.6f} | "
                f"time elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) // 60):02d} | "
                f"est remaining {hh}:{mm:02d} | "
                f"skip_vlb={skip_vlb}",
                flush=True
            )

        # checkpoints
        if (epoch % 1000 == 0) and (epoch >= 0):
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

    # final save + evaluation
    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)
    evaluation.testing(testing_dataset_loader, diffusion, ema=ema, args=args, model=model)


def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    """
    Save model final or checkpoint
    """
    if final:
        torch.save(
            {
                "n_epoch": args["EPOCHS"],
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "ema": ema.state_dict(),
                "args": args,
            },
            f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
        )
    else:
        torch.save(
            {
                "n_epoch": epoch,
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "args": args,
                "ema": ema.state_dict(),
                "loss": loss,
            },
            f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
        )


def training_outputs(diffusion, x, est, noisy, epoch, row_size, ema, args, save_imgs=False, save_vids=False):
    """
    Saves video & images based on args info
    """
    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}', exist_ok=True)
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}', exist_ok=True)
    except OSError:
        pass

    if save_imgs:
        if epoch % 100 == 0:
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t)
            out = torch.cat(
                (x[:row_size, ...].cpu(),
                 temp["sample"][:row_size, ...].cpu(),
                 temp["pred_x_0"][:row_size, ...].cpu())
            )
            plt.title(f"real,sample,prediction x_0-{epoch}epoch")
        else:
            out = torch.cat(
                (x[:row_size, ...].cpu(),
                 noisy[:row_size, ...].cpu(),
                 est[:row_size, ...].cpu(),
                 (est - noisy).square().cpu()[:row_size, ...])
            )
            plt.title(f"real,noisy,noise prediction,mse-{epoch}epoch")

        plt.rcParams["figure.dpi"] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap="gray")
        plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}/EPOCH={epoch}.png')
        plt.clf()

    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 500 == 0:
            plt.rcParams["figure.dpi"] = 200
            if epoch % 1000 == 0:
                out = diffusion.forward_backward(ema, x, "half", args["sample_distance"] // 2, denoise_fn="noise_fn")
            else:
                out = diffusion.forward_backward(ema, x, "half", args["sample_distance"] // 4, denoise_fn="noise_fn")
            imgs = [[ax.imshow(gridify_output(xx, row_size), animated=True)] for xx in out]
            ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)
            ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/sample-EPOCH={epoch}.mp4')

    plt.close("all")


def main():
    """
    Load arguments, run training and testing functions, then remove checkpoint directory
    """
    for d in ["./model/", "./diffusion-videos/", "./diffusion-training-images/"]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass

    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

    resume = 0
    if files[0] == "RESUME_RECENT":
        resume = 1
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")
    elif files[0] == "RESUME_FINAL":
        resume = 2
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")

    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    with open(f"{ROOT_DIR}test_args/{file}", "r") as f:
        args = json.load(f)
    args["arg_num"] = file[4:-5]
    args = defaultdict_from_json(args)

    # make arg specific directories
    for d in [
        f'./model/diff-params-ARGS={args["arg_num"]}',
        f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint',
        f'./diffusion-videos/ARGS={args["arg_num"]}',
        f'./diffusion-training-images/ARGS={args["arg_num"]}'
    ]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass

    print(file, args, flush=True)

    in_channels = 1
    if args.get("channels", "") != "":
        in_channels = int(args["channels"])

    # dataset selection
    if args["dataset"].lower() == "cifar":
        training_dataset_loader_, testing_dataset_loader_ = dataset.load_CIFAR10(args, True), dataset.load_CIFAR10(args, False)
        training_dataset_loader = dataset.cycle(training_dataset_loader_)
        testing_dataset_loader = dataset.cycle(testing_dataset_loader_)
    elif args["dataset"].lower() == "carpet":
        training_dataset = dataset.DAGM("./DATASETS/CARPET/Class1", False, args["img_size"], False)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset = dataset.DAGM("./DATASETS/CARPET/Class1", True, args["img_size"], False)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    elif args["dataset"].lower() == "leather":
        if in_channels == 3:
            training_dataset = dataset.MVTec("./DATASETS/leather", anomalous=False, img_size=args["img_size"], rgb=True)
            testing_dataset = dataset.MVTec("./DATASETS/leather", anomalous=True, img_size=args["img_size"], rgb=True, include_good=True)
        else:
            training_dataset = dataset.MVTec("./DATASETS/leather", anomalous=False, img_size=args["img_size"], rgb=False)
            testing_dataset = dataset.MVTec("./DATASETS/leather", anomalous=True, img_size=args["img_size"], rgb=False, include_good=True)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    else:
        training_dataset, testing_dataset = dataset.init_datasets(ROOT_DIR, args)
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    loaded_model = {}
    if resume:
        if resume == 1:
            ckpt_dir = f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'
            checkpoints = os.listdir(ckpt_dir)
            checkpoints.sort(reverse=True)
            for ck in checkpoints:
                try:
                    file_dir = os.path.join(ckpt_dir, ck)
                    loaded_model = torch.load(file_dir, map_location=device, weights_only=False)
                    break
                except RuntimeError:
                    continue
        else:
            file_dir = f'./model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device, weights_only=False)

    train(training_dataset_loader, testing_dataset_loader, args, loaded_model)

    # remove checkpoints after final_param is saved
    ckpt_path = f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'
    if os.path.isdir(ckpt_path):
        for file_remove in os.listdir(ckpt_path):
            os.remove(os.path.join(ckpt_path, file_remove))
        try:
            os.removedirs(ckpt_path)
        except OSError:
            pass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)
    main()