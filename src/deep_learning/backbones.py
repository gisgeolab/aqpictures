"""Frozen ImageNet backbone embeddings for controlled architecture comparisons."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)

from .embeddings import ROIImageDataset


BACKBONE_SPECS = {
    "efficientnet_b0": {
        "dimension": 1280,
        "weights_name": str(EfficientNet_B0_Weights.DEFAULT),
    },
    "resnet18": {
        "dimension": 512,
        "weights_name": str(ResNet18_Weights.DEFAULT),
    },
    "resnet50": {
        "dimension": 2048,
        "weights_name": str(ResNet50_Weights.DEFAULT),
    },
    "mobilenet_v3_small": {
        "dimension": 576,
        "weights_name": str(MobileNet_V3_Small_Weights.DEFAULT),
    },
}
PREPROCESSING_VERSION = "fixed-roi-resize-imagenet-normalization-v1"


def _build_encoder(backbone_name):
    if backbone_name == "efficientnet_b0":
        encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        encoder.classifier = nn.Identity()
        return encoder
    if backbone_name == "resnet18":
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder.fc = nn.Identity()
        return encoder
    if backbone_name == "resnet50":
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder.fc = nn.Identity()
        return encoder
    if backbone_name == "mobilenet_v3_small":
        encoder = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        encoder.classifier = nn.Identity()
        return encoder
    raise ValueError(
        f"Unsupported backbone {backbone_name!r}; "
        f"choose from {sorted(BACKBONE_SPECS)}."
    )


def extract_backbone_embeddings(
    data,
    roi_config,
    device,
    backbone_name,
    image_size=224,
    batch_size=64,
):
    """Extract frozen embeddings with a supported ImageNet backbone."""
    if backbone_name not in BACKBONE_SPECS:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    loader = DataLoader(
        ROIImageDataset(data, roi_config, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    encoder = _build_encoder(backbone_name).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    batches = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, images in enumerate(loader, start=1):
            output = encoder(images.to(device))
            batches.append(output.cpu().numpy().astype(np.float32))
            if batch_number == 1 or batch_number % 10 == 0 or batch_number == len(loader):
                processed = min(batch_number * batch_size, len(data))
                print(f"{backbone_name} embedding extraction: {processed}/{len(data)}")
    embeddings = np.concatenate(batches, axis=0)
    elapsed = time.perf_counter() - started
    del encoder
    if device.type == "mps":
        torch.mps.empty_cache()

    expected_shape = (len(data), BACKBONE_SPECS[backbone_name]["dimension"])
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected {backbone_name} embedding shape: {embeddings.shape}; "
            f"expected {expected_shape}."
        )
    return embeddings, elapsed


def load_or_extract_backbone_embeddings(
    data,
    roi_config,
    cache_path,
    device,
    backbone_name,
    image_size=224,
    batch_size=64,
):
    """Load a compatible backbone cache or extract and save embeddings."""
    if backbone_name not in BACKBONE_SPECS:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    cache_path = Path(cache_path)
    time_keys = data["time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=str)
    image_paths = [Path(path) for path in data["image_path"]]
    image_keys = np.asarray([str(path) for path in image_paths], dtype=str)
    image_sizes = np.asarray([path.stat().st_size for path in image_paths], dtype=np.int64)
    image_mtimes = np.asarray([path.stat().st_mtime_ns for path in image_paths], dtype=np.int64)
    roi_signature = np.asarray(
        [roi_config["top"], roi_config["left"], roi_config["height"], roi_config["width"]],
        dtype=np.int32,
    )
    spec = BACKBONE_SPECS[backbone_name]

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            required_keys = {
                "embeddings", "time_keys", "image_keys", "image_sizes",
                "image_mtimes", "roi_signature", "image_size", "backbone_name",
                "weights_name", "preprocessing_version",
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
                    and str(cached["backbone_name"]) == backbone_name
                    and str(cached["weights_name"]) == spec["weights_name"]
                    and str(cached["preprocessing_version"]) == PREPROCESSING_VERSION
                )
            if compatible:
                embeddings = cached["embeddings"].astype(np.float32)
                expected_shape = (len(data), spec["dimension"])
                if embeddings.shape != expected_shape:
                    raise ValueError(f"Unexpected cached shape: {embeddings.shape}")
                return embeddings, True, None

    embeddings, elapsed = extract_backbone_embeddings(
        data=data,
        roi_config=roi_config,
        device=device,
        backbone_name=backbone_name,
        image_size=image_size,
        batch_size=batch_size,
    )
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
        backbone_name=np.asarray(backbone_name),
        weights_name=np.asarray(spec["weights_name"]),
        preprocessing_version=np.asarray(PREPROCESSING_VERSION),
    )
    return embeddings, False, elapsed
