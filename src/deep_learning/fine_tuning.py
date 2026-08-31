"""Date-grouped limited fine-tuning for MobileNetV3-Small Fusion."""

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.features import evaluation

from .embeddings import ROIImageDataset
from .models import EmbeddingFusionRegressor


DEFAULT_FINE_TUNE_CONFIG = {
    "batch_size": 64,
    "maximum_epochs": 15,
    "early_stopping_patience": 4,
    "backbone_learning_rate": 1e-5,
    "head_learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "dropout": 0.25,
    "internal_validation_size": 0.15,
    "image_hidden": 64,
    "tabular_hidden": 32,
    "tabular_output": 16,
    "fusion_hidden": 32,
    "image_size": 224,
}


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_mean_std(values):
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.where(std > 0, std, 1.0).astype(np.float32)


class _FusionImageDataset(Dataset):
    def __init__(self, data, roi_config, tabular, targets, image_size):
        self.images = ROIImageDataset(data, roi_config, image_size=image_size)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.tabular[index], self.targets[index]


class MobileNetLimitedFusion(nn.Module):
    def __init__(self, tabular_dimension, config):
        super().__init__()
        self.backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )
        self.backbone.classifier = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.features[-1].parameters():
            parameter.requires_grad = True
        self.regressor = EmbeddingFusionRegressor(
            image_dimension=576,
            tabular_dimension=tabular_dimension,
            dropout=config["dropout"],
            image_hidden=config["image_hidden"],
            tabular_hidden=config["tabular_hidden"],
            tabular_output=config["tabular_output"],
            fusion_hidden=config["fusion_hidden"],
        )

    def forward(self, images, tabular):
        embedding = self.backbone(images)
        return self.regressor(embedding, tabular)


def _make_loader(data, tabular, targets, roi_config, config, shuffle, seed):
    dataset = _FusionImageDataset(
        data=data,
        roi_config=roi_config,
        tabular=tabular,
        targets=targets,
        image_size=config["image_size"],
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def _predict(model, loader, target_mean, target_std, device):
    model.eval()
    observed, predicted = [], []
    with torch.inference_mode():
        for images, tabular, targets in loader:
            output = model(images.to(device), tabular.to(device))
            observed.append(targets.numpy())
            predicted.append(output.cpu().numpy())
    observed = np.concatenate(observed) * target_std + target_mean
    predicted = np.concatenate(predicted) * target_std + target_mean
    return evaluation.evaluate_regression(observed, predicted), observed, predicted


def train_limited_mobilenet_cross_validation(
    data,
    tabular_values,
    targets,
    outer_splits,
    roi_config,
    device,
    config=None,
    random_state=42,
    group_column="date_group",
):
    """Fine-tune only the last MobileNet feature block within grouped CV."""
    config = {**DEFAULT_FINE_TUNE_CONFIG, **(config or {})}
    outputs = []
    for fold_number, (outer_train_index, outer_validation_index) in enumerate(
        outer_splits, start=1
    ):
        seed = random_state + fold_number
        _set_seed(seed)
        outer_train_data = data.iloc[outer_train_index].reset_index(drop=True)
        inner_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config["internal_validation_size"],
            random_state=seed,
        )
        inner_train_position, inner_validation_position = next(
            inner_splitter.split(
                outer_train_data,
                groups=outer_train_data[group_column],
            )
        )
        inner_train_index = outer_train_index[inner_train_position]
        inner_validation_index = outer_train_index[inner_validation_position]

        tabular_mean, tabular_std = _safe_mean_std(tabular_values[inner_train_index])
        target_mean = float(targets[inner_train_index].mean())
        target_std = float(targets[inner_train_index].std())
        if target_std <= 0:
            target_std = 1.0

        def transformed(indices):
            return (
                data.iloc[indices].reset_index(drop=True),
                (tabular_values[indices] - tabular_mean) / tabular_std,
                (targets[indices] - target_mean) / target_std,
            )

        train_loader = _make_loader(
            *transformed(inner_train_index), roi_config, config, True, seed
        )
        internal_loader = _make_loader(
            *transformed(inner_validation_index), roi_config, config, False, seed
        )
        outer_loader = _make_loader(
            *transformed(outer_validation_index), roi_config, config, False, seed
        )

        model = MobileNetLimitedFusion(
            tabular_dimension=tabular_values.shape[1], config=config
        ).to(device)
        backbone_parameters = [
            parameter
            for parameter in model.backbone.features[-1].parameters()
            if parameter.requires_grad
        ]
        head_parameters = list(model.regressor.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": config["backbone_learning_rate"]},
                {"params": head_parameters, "lr": config["head_learning_rate"]},
            ],
            weight_decay=config["weight_decay"],
        )

        best_rmse = np.inf
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        history = []
        started = time.perf_counter()
        for epoch in range(1, config["maximum_epochs"] + 1):
            model.train()
            # Keep batch-normalisation statistics fixed in frozen blocks.
            model.backbone.eval()
            model.backbone.features[-1].train()
            model.regressor.train()
            losses = []
            for images, tabular, target in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(images.to(device), tabular.to(device))
                loss = torch.mean((prediction - target.to(device)) ** 2)
                loss.backward()
                optimizer.step()
                losses.append(loss.item() * len(target))
            train_mse = sum(losses) / len(train_loader.dataset)
            internal_metrics, _, _ = _predict(
                model, internal_loader, target_mean, target_std, device
            )
            history.append({
                "Fold": fold_number,
                "Epoch": epoch,
                "Train MSE": train_mse,
                "Internal validation RMSE": internal_metrics["RMSE"],
                "Internal validation R2": internal_metrics["R2"],
            })
            if internal_metrics["RMSE"] < best_rmse - 1e-4:
                best_rmse = internal_metrics["RMSE"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"Limited fine-tune fold {fold_number} | Epoch {epoch:03d} | "
                    f"Internal RMSE {internal_metrics['RMSE']:.3f} | Best {best_epoch}"
                )
            if epochs_without_improvement >= config["early_stopping_patience"]:
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold_number} did not produce a model.")
        model.load_state_dict(best_state)
        metrics, observed, predictions = _predict(
            model, outer_loader, target_mean, target_std, device
        )
        elapsed = time.perf_counter() - started
        print(
            f"Limited fine-tune fold {fold_number} completed | "
            f"Best epoch {best_epoch} | Outer R2 {metrics['R2']:.3f} | {elapsed:.1f} s"
        )
        outputs.append({
            "metrics": metrics,
            "history": pd.DataFrame(history),
            "best_epoch": best_epoch,
            "training_time": elapsed,
            "targets": observed,
            "predictions": predictions,
            "validation_index": outer_validation_index,
        })
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    return outputs
