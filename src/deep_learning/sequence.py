"""Controlled CNN--LSTM sequence diagnostic for webcam embeddings."""

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.evaluation import evaluate_regression


class CNNLSTMRegressor(nn.Module):
    """Small image-sequence regressor; the CNN backbone is external/frozen."""

    def __init__(self, embedding_dim, channels=64, hidden=32, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, channels, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, sequence):
        # Apply the convolution over the time dimension, then retain one vector.
        features = self.cnn(sequence.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(features)
        return self.head(output[:, -1]).squeeze(1)


def make_within_day_sequences(data, embeddings, sequence_length=3):
    """Create sequences without crossing independent dates."""
    data = data.reset_index(drop=True).copy()
    data["date_group"] = pd.to_datetime(data["time"]).dt.normalize()
    sequences, targets, groups = [], [], []
    for date, frame in data.groupby("date_group", sort=True):
        indices = frame.sort_values("time").index.to_numpy()
        for end in range(sequence_length - 1, len(indices)):
            window = indices[end - sequence_length + 1:end + 1]
            sequences.append(embeddings[window])
            targets.append(float(data.loc[indices[end], "PM25"]))
            groups.append(date)
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(groups)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_cnn_lstm_cv(sequences, targets, groups, n_splits=5, seed=42, epochs=80):
    """Evaluate the controlled sequence model with date-grouped CV."""
    results = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(sequences, targets, groups), 1):
        _seed(seed + fold)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold)
        train_pos, early_pos = next(inner.split(train_idx, groups=groups[train_idx]))
        fit_idx, early_idx = train_idx[train_pos], train_idx[early_pos]
        x_mean, x_std = sequences[fit_idx].mean((0, 1)), sequences[fit_idx].std((0, 1))
        x_std = np.where(x_std > 0, x_std, 1.0)
        y_mean, y_std = targets[fit_idx].mean(), targets[fit_idx].std() or 1.0
        model = CNNLSTMRegressor(sequences.shape[-1])
        optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        loader = DataLoader(TensorDataset(torch.tensor((sequences[fit_idx]-x_mean)/x_std), torch.tensor((targets[fit_idx]-y_mean)/y_std)), batch_size=64, shuffle=True)
        best_state, best_rmse, patience = None, np.inf, 0
        for _ in range(epochs):
            model.train()
            for x_batch, y_batch in loader:
                optimiser.zero_grad(); loss = ((model(x_batch) - y_batch) ** 2).mean(); loss.backward(); optimiser.step()
            model.eval()
            with torch.inference_mode():
                pred = model(torch.tensor((sequences[early_idx]-x_mean)/x_std)).numpy() * y_std + y_mean
            rmse = float(np.sqrt(np.mean((pred-targets[early_idx])**2)))
            if rmse < best_rmse:
                best_rmse, best_state, patience = rmse, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
            if patience >= 12: break
        model.load_state_dict(best_state); model.eval()
        with torch.inference_mode():
            pred = model(torch.tensor((sequences[valid_idx]-x_mean)/x_std)).numpy() * y_std + y_mean
        metrics = evaluate_regression(targets[valid_idx], pred)
        results.append({"Fold": fold, **metrics})
    return pd.DataFrame(results)


def evaluate_cnn_lstm_split(
    sequences, targets, groups, train_groups, test_groups, seed=42, epochs=80
):
    """Evaluate one fixed date split with training-only early stopping."""
    # Normalise all date labels to the same nanosecond-resolution representation
    # before matching; Pandas Timestamp and NumPy datetime64 otherwise may fail
    # to compare even when they represent the same calendar date.
    groups = pd.to_datetime(groups).normalize().to_numpy(dtype="datetime64[ns]")
    train_groups = pd.to_datetime(train_groups).normalize().to_numpy(dtype="datetime64[ns]")
    test_groups = pd.to_datetime(test_groups).normalize().to_numpy(dtype="datetime64[ns]")
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Fixed split contains no sequences: train={train_mask.sum()}, "
            f"test={test_mask.sum()}. Check date labels and cache alignment."
        )
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    fit_pos, early_pos = next(inner.split(train_indices, groups=groups[train_indices]))
    fit_idx, early_idx = train_indices[fit_pos], train_indices[early_pos]
    _seed(seed)
    x_mean, x_std = sequences[fit_idx].mean((0, 1)), sequences[fit_idx].std((0, 1))
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_mean, y_std = targets[fit_idx].mean(), targets[fit_idx].std() or 1.0
    model = CNNLSTMRegressor(sequences.shape[-1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    x_fit = torch.tensor((sequences[fit_idx] - x_mean) / x_std)
    y_fit = torch.tensor((targets[fit_idx] - y_mean) / y_std)
    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=64, shuffle=True)
    best_state, best_rmse, patience = None, np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            loss = ((model(x_batch) - y_batch) ** 2).mean()
            loss.backward(); optimiser.step()
        model.eval()
        with torch.inference_mode():
            early_pred = model(torch.tensor((sequences[early_idx]-x_mean)/x_std)).numpy()*y_std+y_mean
        rmse = float(np.sqrt(np.mean((early_pred-targets[early_idx])**2)))
        if rmse < best_rmse:
            best_rmse, best_state, patience = rmse, copy.deepcopy(model.state_dict()), 0
        else: patience += 1
        if patience >= 12: break
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        predictions = model(torch.tensor((sequences[test_indices]-x_mean)/x_std)).numpy()*y_std+y_mean
    return {"metrics": evaluate_regression(targets[test_indices], predictions), "best_epoch": epoch, "n_test": len(test_indices)}
