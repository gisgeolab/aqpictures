"""Frozen EfficientNet-B0 embedding extraction and caching."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EFFICIENTNET_B0_EMBEDDING_DIMENSION = 1280
BACKBONE_NAME = "efficientnet_b0"
WEIGHTS_NAME = str(EfficientNet_B0_Weights.DEFAULT)
PREPROCESSING_VERSION = "fixed-roi-resize-imagenet-normalization-v1"


def _crop_fixed_roi(image, roi_config):
    left = int(roi_config["left"])
    top = int(roi_config["top"])
    right = left + int(roi_config["width"])
    bottom = top + int(roi_config["height"])
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        raise ValueError(f"ROI outside image boundaries: {image.size}")
    return image.crop((left, top, right, bottom))


class ROIImageDataset(Dataset):
    """Load source images, crop the fixed ROI, and apply ImageNet transforms."""

    def __init__(self, data, roi_config, image_size=224):
        self.data = data.reset_index(drop=True).copy()
        self.roi_config = dict(roi_config)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with Image.open(self.data.iloc[index]["image_path"]) as image_file:
            image = image_file.convert("RGB").copy()
        return self.transform(_crop_fixed_roi(image, self.roi_config))


def extract_embeddings(data, roi_config, device, image_size=224, batch_size=64):
    """Extract frozen ImageNet-pretrained EfficientNet-B0 embeddings."""
    loader = DataLoader(
        ROIImageDataset(data, roi_config, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    encoder.classifier = nn.Identity()
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    batches = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, images in enumerate(loader, start=1):
            batches.append(encoder(images.to(device)).cpu().numpy().astype(np.float32))
            if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(loader):
                processed = min(batch_number * batch_size, len(data))
                print(f"Embedding extraction: {processed}/{len(data)}")

    embeddings = np.concatenate(batches, axis=0)
    elapsed = time.perf_counter() - started
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()
    return embeddings, elapsed


def load_or_extract_embeddings(
    data,
    roi_config,
    cache_path,
    device,
    image_size=224,
    batch_size=64,
):
    """Load a compatible cache or extract and save development embeddings."""
    cache_path = Path(cache_path)
    time_keys = data["time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=str)
    image_paths = [Path(path) for path in data["image_path"]]
    image_keys = np.asarray([str(path) for path in image_paths], dtype=str)
    image_sizes = np.asarray([path.stat().st_size for path in image_paths], dtype=np.int64)
    image_mtimes = np.asarray(
        [path.stat().st_mtime_ns for path in image_paths], dtype=np.int64
    )
    roi_signature = np.array(
        [
            roi_config["top"],
            roi_config["left"],
            roi_config["height"],
            roi_config["width"],
        ],
        dtype=np.int32,
    )

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            required_keys = {
                "embeddings",
                "time_keys",
                "image_keys",
                "image_sizes",
                "image_mtimes",
                "roi_signature",
                "image_size",
                "backbone_name",
                "weights_name",
                "preprocessing_version",
            }
            compatible = required_keys.issubset(cached.files)
            if compatible:
                compatible = (
                    np.array_equal(cached["time_keys"], time_keys)
                    and np.array_equal(cached["image_keys"], image_keys)
                    and np.array_equal(cached["image_sizes"], image_sizes)
                    and np.array_equal(cached["image_mtimes"], image_mtimes)
                    and np.array_equal(cached["roi_signature"], roi_signature)
                    and int(cached["image_size"]) == image_size
                    and str(cached["backbone_name"]) == BACKBONE_NAME
                    and str(cached["weights_name"]) == WEIGHTS_NAME
                    and str(cached["preprocessing_version"])
                    == PREPROCESSING_VERSION
                )
            if compatible:
                embeddings = cached["embeddings"].astype(np.float32)
                if embeddings.shape != (len(data), EFFICIENTNET_B0_EMBEDDING_DIMENSION):
                    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
                return embeddings, True, None

    embeddings, elapsed = extract_embeddings(
        data,
        roi_config,
        device,
        image_size=image_size,
        batch_size=batch_size,
    )
    if embeddings.shape != (len(data), EFFICIENTNET_B0_EMBEDDING_DIMENSION):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        time_keys=time_keys,
        image_keys=image_keys,
        image_sizes=image_sizes,
        image_mtimes=image_mtimes,
        roi_signature=roi_signature,
        image_size=np.asarray(image_size, dtype=np.int32),
        backbone_name=np.asarray(BACKBONE_NAME),
        weights_name=np.asarray(WEIGHTS_NAME),
        preprocessing_version=np.asarray(PREPROCESSING_VERSION),
    )
    return embeddings, False, elapsed
