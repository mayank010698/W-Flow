from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset.latent import LatentDataset
from utils.dist_util import process_count, process_index
from utils.env import CIFAR10_PATH, IMAGENET_CACHE_PATH, IMAGENET_PATH, require
from utils.logging import log_for_0


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center-crop image with ADM preprocessing style."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _build_transforms(resolution: int, use_aug: bool, split: str):
    if use_aug and split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _build_imagenet_dataset(*, resolution: int, use_aug: bool, use_cache: bool, split: str):
    if use_cache:
        return LatentDataset(root=os.path.join(IMAGENET_CACHE_PATH, split))

    transform = _build_transforms(resolution, use_aug=use_aug, split=split)
    return ImageFolder(root=os.path.join(IMAGENET_PATH, split), transform=transform)


def worker_init_fn(worker_id: int, rank: int) -> None:
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_imagenet_split(
    *,
    resolution: int,
    batch_size: int,
    split: str,
    use_aug: bool = False,
    use_latent: bool = False,
    use_cache: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    local: bool | None = None,
):
    del local
    ds = _build_imagenet_dataset(
        resolution=resolution,
        use_aug=use_aug,
        use_cache=use_cache,
        split=split,
    )
    log_for_0(ds)

    world = process_count()
    rank = process_index()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=(split == "train"),
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    if use_latent or use_cache:
        from dataset.vae import vae_enc_decode

        encode_fn, decode_fn = vae_enc_decode()
        if use_cache:

            def preprocess_fn(batch, rng=0):
                del rng
                cached_latent, label = batch
                return {
                    "images": torch.as_tensor(cached_latent, dtype=torch.float32),
                    "labels": torch.as_tensor(label, dtype=torch.long),
                }

        else:

            def preprocess_fn(batch, rng=0):
                image, label = batch
                image = torch.as_tensor(image, dtype=torch.float32)
                latents = encode_fn(image, rng)
                return {
                    "images": latents,
                    "labels": torch.as_tensor(label, dtype=torch.long),
                }

        def postprocess_fn(images):
            out = (decode_fn(images) + 1.0) / 2.0
            return torch.clamp(out, 0.0, 1.0)

        return loader, preprocess_fn, postprocess_fn

    def preprocess_fn(batch, rng=0):
        del rng
        image, label = batch
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.permute(0, 2, 3, 1).contiguous()
        return {
            "images": image,
            "labels": torch.as_tensor(label, dtype=torch.long),
        }

    def postprocess_fn(images):
        out = (torch.as_tensor(images) + 1.0) / 2.0
        out = torch.clamp(out, 0.0, 1.0)
        return out.permute(0, 3, 1, 2).contiguous()

    return loader, preprocess_fn, postprocess_fn


class CIFAR10Arrays(torch.utils.data.Dataset):
    """CIFAR-10 from `$CIFAR10_PATH/cifar10_{train,test}.npz` (uint8 NHWC `data`, int64 `labels`),
    built by `scripts/prepare_cifar10.py convert` from the pinned HF parquet (see SOURCE.json)."""

    def __init__(self, root: str, split: str, transform=None):
        path = os.path.join(root, f"cifar10_{split}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} missing; run `python -m scripts.prepare_cifar10 download` then `convert`.")
        with np.load(path) as f:
            self.data = f["data"]
            self.targets = f["labels"].astype(np.int64)
        assert self.data.shape[1:] == (32, 32, 3) and self.data.dtype == np.uint8, self.data.shape
        self.split, self.transform = split, transform

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        return (self.transform(img) if self.transform else img), int(self.targets[i])

    def __repr__(self):
        return f"CIFAR10Arrays(split={self.split}, n={len(self)}, transform={self.transform})"


def _build_cifar10_dataset(*, split: str, aug: str):
    """Images -> CHW float in [-1, 1]. aug: "none" | "flip" (generator) | "crop_flip" (MAE)."""
    tfs = []
    if aug == "crop_flip":
        tfs.append(transforms.RandomCrop(32, padding=4, padding_mode="reflect"))
    if aug in ("flip", "crop_flip"):
        tfs.append(transforms.RandomHorizontalFlip())
    elif aug != "none":
        raise ValueError(f"Unknown CIFAR-10 aug: {aug}")
    tfs += [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    return CIFAR10Arrays(require(CIFAR10_PATH, "CIFAR10_PATH"), split, transforms.Compose(tfs))


def create_cifar10_split(
    *,
    batch_size: int,
    split: str,
    aug: str = "none",
    shuffle: bool = True,
    drop_last: bool | None = None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    **unused,
):
    """Pixel-space CIFAR-10 loader with the same (loader, preprocess, postprocess) contract
    as `create_imagenet_split`. preprocess -> BHWC in [-1,1]; postprocess -> BCHW in [0,1]."""
    del unused
    ds = _build_cifar10_dataset(split=split, aug=aug)
    log_for_0(ds)
    world, rank = process_count(), process_index()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=(split == "train") if drop_last is None else drop_last,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    def preprocess_fn(batch, rng=0):
        del rng
        image, label = batch
        image = torch.as_tensor(image, dtype=torch.float32).permute(0, 2, 3, 1).contiguous()
        return {"images": image, "labels": torch.as_tensor(label, dtype=torch.long)}

    def postprocess_fn(images):
        out = torch.clamp((torch.as_tensor(images) + 1.0) / 2.0, 0.0, 1.0)
        return out.permute(0, 3, 1, 2).contiguous()

    return loader, preprocess_fn, postprocess_fn


def get_postprocess_fn(*, use_aug: bool = False, use_latent: bool = False, use_cache: bool = False, has_clip: bool = True):
    if use_latent or use_cache:
        from dataset.vae import vae_enc_decode

        _, decode_fn = vae_enc_decode()

        def postprocess(images):
            out = (decode_fn(images) + 1.0) / 2.0
            return torch.clamp(out, 0.0, 1.0) if has_clip else out

        return postprocess

    if use_aug or (not use_latent and not use_cache):

        def postprocess(images):
            out = (torch.as_tensor(images) + 1.0) / 2.0
            out = torch.clamp(out, 0.0, 1.0) if has_clip else out
            return out.permute(0, 3, 1, 2).contiguous()

        return postprocess

    raise ValueError("Unsupported dataset flags.")


def infinite_sampler(it, start_step: int = 0):
    step_per_epoch = len(it)
    epoch_idx = start_step // step_per_epoch
    it.sampler.set_epoch(epoch_idx)
    skip_batches = start_step % step_per_epoch
    while True:
        for i, batch in enumerate(it):
            if skip_batches > 0 and i < skip_batches:
                continue
            image, label = batch
            yield (image, label)
        skip_batches = 0
        epoch_idx += 1
        it.sampler.set_epoch(epoch_idx)


def epoch0_sampler(it):
    it.sampler.set_epoch(0)
    for batch in it:
        image, label = batch
        yield (image, label)
