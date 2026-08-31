"""Date-grouped cross-validation for frozen-embedding regressors."""

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.features import evaluation

from .models import EmbeddingFusionRegressor, ImageOnlyRegressor


DEFAULT_TRAINING_CONFIG = {
    "batch_size": 64,
    "maximum_epochs": 150,
    "early_stopping_patience": 12,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.30,
    "internal_validation_size": 0.15,
    "image_hidden": 128,
    "tabular_hidden": 64,
    "tabular_output": 32,
    "fusion_hidden": 64,
}


def build_outer_splits(data, n_splits=5, group_column="date_group"):
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(data, groups=data[group_column].to_numpy()))


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_mean_std(values):
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.where(std > 0, std, 1.0).astype(np.float32)


def _make_loader(arrays, batch_size, shuffle, seed):
    dataset = TensorDataset(
        *(torch.tensor(values, dtype=torch.float32) for values in arrays)
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def _unweighted_mse(prediction, target):
    return torch.mean((prediction - target) ** 2)


def _inner_indices(data, outer_train_index, validation_size, seed, group_column):
    outer_train_data = data.iloc[outer_train_index].reset_index(drop=True)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=seed
    )
    train_position, validation_position = next(
        splitter.split(outer_train_data, groups=outer_train_data[group_column])
    )
    return outer_train_index[train_position], outer_train_index[validation_position]


def _metrics(targets, predictions):
    return evaluation.evaluate_regression(targets, predictions)


def _evaluate_fusion(model, loader, target_mean, target_std, device):
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for image_values, tabular_values, target_values in loader:
            prediction = model(
                image_values.to(device), tabular_values.to(device)
            ).cpu().numpy()
            targets.append(target_values.numpy())
            predictions.append(prediction)
    targets = np.concatenate(targets) * target_std + target_mean
    predictions = np.concatenate(predictions) * target_std + target_mean
    return _metrics(targets, predictions), targets, predictions


def _evaluate_image_only(model, loader, target_mean, target_std, device):
    model.eval()
    targets, predictions = [], []
    with torch.inference_mode():
        for image_values, target_values in loader:
            predictions.append(model(image_values.to(device)).cpu().numpy())
            targets.append(target_values.numpy())
    targets = np.concatenate(targets) * target_std + target_mean
    predictions = np.concatenate(predictions) * target_std + target_mean
    return _metrics(targets, predictions), targets, predictions


def _fit_fold(
    model,
    optimizer,
    train_loader,
    internal_validation_loader,
    outer_validation_loader,
    evaluate_loader,
    forward_batch,
    target_mean,
    target_std,
    fold_number,
    config,
    device,
    label,
):
    best_rmse = np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, target = forward_batch(model, batch, device)
            loss = _unweighted_mse(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item() * len(target))
        train_mse = sum(losses) / len(train_loader.dataset)

        internal_metrics, _, _ = evaluate_loader(
            model, internal_validation_loader, target_mean, target_std, device
        )
        history.append(
            {
                "Fold": fold_number,
                "Epoch": epoch,
                "Train MSE": train_mse,
                "Internal validation RMSE": internal_metrics["RMSE"],
                "Internal validation R2": internal_metrics["R2"],
            }
        )
        if internal_metrics["RMSE"] < best_rmse - 1e-4:
            best_rmse = internal_metrics["RMSE"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{label} fold {fold_number} | Epoch {epoch:03d} | "
                f"Internal RMSE {internal_metrics['RMSE']:.3f} | "
                f"Best epoch {best_epoch}"
            )
        if epochs_without_improvement >= config["early_stopping_patience"]:
            break

    if best_state is None:
        raise RuntimeError(f"{label} fold {fold_number} did not produce a model.")
    model.load_state_dict(best_state)
    outer_metrics, outer_targets, outer_predictions = evaluate_loader(
        model, outer_validation_loader, target_mean, target_std, device
    )
    elapsed = time.perf_counter() - started
    print(
        f"{label} fold {fold_number} completed | Best epoch {best_epoch} | "
        f"Outer R2 {outer_metrics['R2']:.3f} | {elapsed:.1f} s"
    )
    return {
        "metrics": outer_metrics,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "training_time": elapsed,
        "targets": outer_targets,
        "predictions": outer_predictions,
    }


def train_fusion_cross_validation(
    data,
    embeddings,
    tabular_values,
    targets,
    outer_splits,
    device,
    config=None,
    random_state=42,
    group_column="date_group",
):
    config = {**DEFAULT_TRAINING_CONFIG, **(config or {})}
    outputs = []
    for fold_number, (outer_train_index, outer_validation_index) in enumerate(
        outer_splits, start=1
    ):
        seed = random_state + fold_number
        _set_seed(seed)
        inner_train_index, inner_validation_index = _inner_indices(
            data,
            outer_train_index,
            config["internal_validation_size"],
            seed,
            group_column,
        )
        image_mean, image_std = _safe_mean_std(embeddings[inner_train_index])
        tabular_mean, tabular_std = _safe_mean_std(tabular_values[inner_train_index])
        target_mean = float(targets[inner_train_index].mean())
        target_std = float(targets[inner_train_index].std())
        if target_std <= 0:
            target_std = 1.0

        def transform(indices):
            return (
                (embeddings[indices] - image_mean) / image_std,
                (tabular_values[indices] - tabular_mean) / tabular_std,
                (targets[indices] - target_mean) / target_std,
            )

        loaders = [
            _make_loader(
                transform(indices), config["batch_size"], shuffle, seed
            )
            for indices, shuffle in [
                (inner_train_index, True),
                (inner_validation_index, False),
                (outer_validation_index, False),
            ]
        ]
        model = EmbeddingFusionRegressor(
            embeddings.shape[1],
            tabular_values.shape[1],
            config["dropout"],
            image_hidden=config["image_hidden"],
            tabular_hidden=config["tabular_hidden"],
            tabular_output=config["tabular_output"],
            fusion_hidden=config["fusion_hidden"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        def forward_batch(current_model, batch, current_device):
            image, tabular, target = batch
            return (
                current_model(image.to(current_device), tabular.to(current_device)),
                target.to(current_device),
            )

        output = _fit_fold(
            model,
            optimizer,
            *loaders,
            _evaluate_fusion,
            forward_batch,
            target_mean,
            target_std,
            fold_number,
            config,
            device,
            "Fusion",
        )
        output["validation_index"] = outer_validation_index
        outputs.append(output)
    return outputs


def train_image_only_cross_validation(
    data,
    embeddings,
    targets,
    outer_splits,
    device,
    config=None,
    random_state=42,
    group_column="date_group",
):
    config = {**DEFAULT_TRAINING_CONFIG, **(config or {})}
    outputs = []
    for fold_number, (outer_train_index, outer_validation_index) in enumerate(
        outer_splits, start=1
    ):
        seed = random_state + fold_number
        _set_seed(seed)
        inner_train_index, inner_validation_index = _inner_indices(
            data,
            outer_train_index,
            config["internal_validation_size"],
            seed,
            group_column,
        )
        image_mean, image_std = _safe_mean_std(embeddings[inner_train_index])
        target_mean = float(targets[inner_train_index].mean())
        target_std = float(targets[inner_train_index].std())
        if target_std <= 0:
            target_std = 1.0

        def transform(indices):
            return (
                (embeddings[indices] - image_mean) / image_std,
                (targets[indices] - target_mean) / target_std,
            )

        loaders = [
            _make_loader(
                transform(indices), config["batch_size"], shuffle, seed
            )
            for indices, shuffle in [
                (inner_train_index, True),
                (inner_validation_index, False),
                (outer_validation_index, False),
            ]
        ]
        model = ImageOnlyRegressor(embeddings.shape[1], config["dropout"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        def forward_batch(current_model, batch, current_device):
            image, target = batch
            return current_model(image.to(current_device)), target.to(current_device)

        output = _fit_fold(
            model,
            optimizer,
            *loaders,
            _evaluate_image_only,
            forward_batch,
            target_mean,
            target_std,
            fold_number,
            config,
            device,
            "Image-only",
        )
        output["validation_index"] = outer_validation_index
        outputs.append(output)
    return outputs


def fit_and_evaluate_fusion_holdout(
    development_embeddings,
    development_tabular,
    development_targets,
    holdout_embeddings,
    holdout_tabular,
    holdout_targets,
    device,
    epochs,
    config=None,
    random_state=42,
):
    """Fit Fusion on all development data and evaluate one locked hold-out."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    config = {**DEFAULT_TRAINING_CONFIG, **(config or {})}
    _set_seed(random_state)

    image_mean, image_std = _safe_mean_std(development_embeddings)
    tabular_mean, tabular_std = _safe_mean_std(development_tabular)
    target_mean = float(development_targets.mean())
    target_std = float(development_targets.std())
    if target_std <= 0:
        target_std = 1.0

    def transform(embeddings, tabular, targets):
        return (
            (embeddings - image_mean) / image_std,
            (tabular - tabular_mean) / tabular_std,
            (targets - target_mean) / target_std,
        )

    train_loader = _make_loader(
        transform(
            development_embeddings,
            development_tabular,
            development_targets,
        ),
        config["batch_size"],
        True,
        random_state,
    )
    holdout_loader = _make_loader(
        transform(holdout_embeddings, holdout_tabular, holdout_targets),
        config["batch_size"],
        False,
        random_state,
    )

    model = EmbeddingFusionRegressor(
        development_embeddings.shape[1],
        development_tabular.shape[1],
        config["dropout"],
        image_hidden=config["image_hidden"],
        tabular_hidden=config["tabular_hidden"],
        tabular_output=config["tabular_output"],
        fusion_hidden=config["fusion_hidden"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for image, tabular, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image.to(device), tabular.to(device))
            loss = _unweighted_mse(prediction, target.to(device))
            loss.backward()
            optimizer.step()
    training_time = time.perf_counter() - started

    started = time.perf_counter()
    metrics, observed, predictions = _evaluate_fusion(
        model,
        holdout_loader,
        target_mean,
        target_std,
        device,
    )
    prediction_time = time.perf_counter() - started
    return {
        "model": model,
        "metrics": metrics,
        "observed": observed,
        "predictions": predictions,
        "epochs": epochs,
        "training_time": training_time,
        "prediction_time": prediction_time,
    }
