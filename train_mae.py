"""Pixel-space MAE-ResNet pretraining on CIFAR-10 (E001 drift feature encoder).

Trains `models.mae_model.MAEResNetJAX` with masked reconstruction (+ optional
class-head finetune phase), keeps an EMA copy, and exports the EMA weights as a
`params_ema/` artifact that `build_feature_model_and_params(path=<run_dir>)`
loads directly (feature.mae_path in the generator config).

Usage:
  python train_mae.py --config configs/mae/cifar10_pixel.yaml --workdir $OUTPUT_ROOT/E001-.../<run>
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tqdm import tqdm

from dataset.dataset import create_cifar10_split, infinite_sampler
from models.mae_model import MAEResNetJAX, make_patch_mask
from utils.ckpt_util import restore_checkpoint, save_checkpoint, save_params_ema_artifact
from utils.dist_util import dist_is_initialized, maybe_ddp_model, process_count, process_index, unwrap_ddp
from utils.logging import WandbLogger, is_rank_zero
from utils.misc import EasyDict, apply_overrides, load_config, run_init, seed_everything
from utils.model_builder import create_learning_rate_fn
from utils.run_meta import assert_outside_repo, collect_run_meta, update_run_meta, write_run_files


@torch.no_grad()
def _update_ema(ema_model, model, decay: float):
    model = unwrap_ddp(model)
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def _sum_all(t: torch.Tensor) -> torch.Tensor:
    if dist_is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


@torch.no_grad()
def evaluate(model, loader, preprocess_fn, fd, device, amp_ctx, seed: int = 0):
    """Masked-recon MSE on the test split with a fixed mask RNG (identical masks every eval).

    Also reports the zero-predictor baseline on the same masks: predicting 0 (the [-1,1] pixel
    midpoint) for every masked pixel, i.e. mean x^2 over masked pixels.
    """
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + process_index())
    totals = torch.zeros(4, device=device, dtype=torch.float64)  # recon_sum, correct, n, zero_baseline_sum
    loader.sampler.set_epoch(0)
    for batch in loader:
        b = preprocess_fn(batch)
        x, y = b["images"].to(device), b["labels"].to(device)
        rng_state = rng.get_state()
        with amp_ctx():
            _, m = model(x, y, lambda_cls=0.0, mask_ratio_min=fd.mask_ratio_min, mask_ratio_max=fd.mask_ratio_max, train=False, rng=rng)
        # Re-draw the identical mask (same RNG state, same calls as the model) for the baseline.
        rng_after = rng.get_state()
        rng.set_state(rng_state)
        b_ = x.shape[0]
        ratio = torch.rand((b_,), dtype=torch.float32, device=device, generator=rng) * (fd.mask_ratio_max - fd.mask_ratio_min) + fd.mask_ratio_min
        mask = make_patch_mask(x.float(), rng, ratio, unwrap_ddp(model).patch_size)
        rng.set_state(rng_after)
        zero_base = ((x.float() ** 2) * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
        totals += torch.stack([
            m["recon_loss"].float().sum().double(), m["accuracy"].float().sum().double(),
            torch.tensor(float(b_), device=device, dtype=torch.float64), zero_base.sum().double(),
        ])
    totals = _sum_all(totals)
    n = max(1.0, float(totals[2]))
    return {
        "eval/recon_loss": float(totals[0]) / n,
        "eval/recon_loss_zero_baseline": float(totals[3]) / n,
        "eval/accuracy": float(totals[1]) / n,
        "eval/n": n,
    }


def main(args):
    run_init()
    config = apply_overrides(load_config(args.config), args.override)
    workdir = str(assert_outside_repo(args.workdir, "--workdir"))
    if "logging" not in config:
        config.logging = EasyDict()
    config.logging.name = Path(workdir).name

    run_meta = collect_run_meta(args.config)
    if is_rank_zero():
        write_run_files(workdir, config, run_meta)

    tcfg = config.train
    seed = int(tcfg.get("seed", 0))
    seed_everything(seed + process_index())

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_amp = bool(tcfg.get("amp_bf16", True)) and device.type == "cuda"
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else nullcontext

    world = process_count()
    kwargs = dict(config.dataset.get("kwargs", {}))
    train_loader, preprocess_fn, _ = create_cifar10_split(
        batch_size=int(config.dataset.batch_size) // world, split="train",
        aug=str(config.dataset.get("train_aug", "crop_flip")), **kwargs,
    )
    eval_loader, _, _ = create_cifar10_split(
        batch_size=int(config.dataset.eval_batch_size) // world, split="test",
        aug="none", shuffle=False, drop_last=False, **kwargs,
    )

    # model_config is saved in the artifact metadata and must match MAEResNetJAX.__init__.
    model_config = dict(config.model)
    model_config["num_classes"] = int(config.dataset.num_classes)
    if model_config.get("use_bf16", False):
        raise ValueError("Keep model.use_bf16=false (bf16 params); use train.amp_bf16 for mixed precision.")
    model = MAEResNetJAX(**model_config).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    model = maybe_ddp_model(model, device_ids=[local_rank] if device.type == "cuda" else None)

    lr_fn = create_learning_rate_fn(**config.optimizer.lr_schedule)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr_fn(0),
        weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
        betas=(float(config.optimizer.adam_b1), float(config.optimizer.adam_b2)),
    )
    state = SimpleNamespace(step=0, model=model, optimizer=opt, ema_model=ema_model, ema_decay=float(tcfg.ema_decay))
    state = restore_checkpoint(state=state, workdir=workdir)
    if state.step and is_rank_zero():
        print(f"Resumed MAE from step {state.step}", flush=True)

    logger = WandbLogger()
    w_cfg = EasyDict(dict(config.get("logging", {})))
    use_wandb = bool(w_cfg.pop("use_wandb", True))
    logger.set_logging(config={**dict(config), "_run_meta": run_meta}, use_wandb=use_wandb, workdir=workdir, **w_cfg)
    if is_rank_zero():
        update_run_meta(workdir, wandb_run_id=logger.wandb_run_id, wandb_url=logger.wandb_url)

    fd = EasyDict(dict(tcfg.forward_dict))
    total_steps = int(tcfg.total_steps)
    finetune_from = total_steps - int(tcfg.get("finetune_last_steps", 0))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    train_iter = infinite_sampler(train_loader, int(state.step))
    rng = torch.Generator(device=device)

    steps = range(int(state.step), total_steps)
    pbar = tqdm(steps, initial=int(state.step), total=total_steps) if is_rank_zero() else steps
    for step in pbar:
        t0 = time.time()
        logger.set_step(step)
        b = preprocess_fn(next(train_iter))
        x, y = b["images"].to(device, non_blocking=True), b["labels"].to(device, non_blocking=True)
        lambda_cls = float(tcfg.get("finetune_cls", 0.0)) if step >= finetune_from else float(fd.get("lambda_cls", 0.0))
        rng.manual_seed((seed * 1_000_003 + step) * world + process_index())

        lr = float(lr_fn(step))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            loss, m = model(x, y, lambda_cls=lambda_cls, mask_ratio_min=fd.mask_ratio_min, mask_ratio_max=fd.mask_ratio_max, train=True, rng=rng)
        loss = loss.float().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite MAE loss at step {step}: {loss.item()}")
        loss.backward()
        g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        _update_ema(state.ema_model, model, state.ema_decay)
        state.step = step + 1

        logger.log_dict({
            "loss": loss, "recon_loss": m["recon_loss"], "cls_loss": m["cls_loss"], "accuracy": m["accuracy"],
            "lambda_cls": lambda_cls, "g_norm": g_norm, "lr": lr, "step_time": time.time() - t0,
        })

        done = state.step == total_steps
        if state.step % int(tcfg.eval_per_step) == 0 or done:
            ev = evaluate(state.ema_model, eval_loader, preprocess_fn, fd, device, amp_ctx)
            logger.log_dict(ev)
            if is_rank_zero():
                print(f"[eval step {state.step}] " + " ".join(f"{k}={v:.6g}" for k, v in ev.items()), flush=True)
        if state.step % int(tcfg.save_per_step) == 0 or done:
            save_checkpoint(state, keep=int(tcfg.get("keep_last", 2)), keep_every=tcfg.get("keep_every"), workdir=workdir)
            if is_rank_zero():
                save_params_ema_artifact(state, workdir=workdir, kind="mae", model_config=model_config)

    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True, help="Unique run dir under $OUTPUT_ROOT (outside the repo).")
    parser.add_argument("--override", nargs="*", default=[], help="Dotted config overrides, e.g. train.total_steps=100")
    main(parser.parse_args())
