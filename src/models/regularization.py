"""Development-only regularization and model-seed diagnostics."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def _ridge(alpha):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _mlp(hidden_layers, alpha, random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            alpha=alpha,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=False,
            random_state=random_state,
        )),
    ])


def get_regularization_candidates(random_state=42):
    """Return focused candidates centred on the current benchmark settings."""

    return {
        "Ridge": [
            {"Configuration": "Alpha 100", "Complexity Rank": 1,
             "Model": _ridge(100.0)},
            {"Configuration": "Alpha 10", "Complexity Rank": 2,
             "Model": _ridge(10.0)},
            {"Configuration": "Current (alpha 1)", "Complexity Rank": 3,
             "Model": _ridge(1.0)},
            {"Configuration": "Alpha 0.1", "Complexity Rank": 4,
             "Model": _ridge(0.1)},
        ],
        "Decision Tree": [
            {"Configuration": "Depth 3; leaf 15", "Complexity Rank": 1,
             "Model": DecisionTreeRegressor(
                 max_depth=3, min_samples_leaf=15,
                 random_state=random_state)},
            {"Configuration": "Depth 4; leaf 25", "Complexity Rank": 2,
             "Model": DecisionTreeRegressor(
                 max_depth=4, min_samples_leaf=25,
                 random_state=random_state)},
            {"Configuration": "Current (depth 4; leaf 15)",
             "Complexity Rank": 3,
             "Model": DecisionTreeRegressor(
                 max_depth=4, min_samples_leaf=15,
                 random_state=random_state)},
            {"Configuration": "Depth 5; leaf 10", "Complexity Rank": 4,
             "Model": DecisionTreeRegressor(
                 max_depth=5, min_samples_leaf=10,
                 random_state=random_state)},
        ],
        "Random Forest": [
            {"Configuration": "Depth 6; leaf 6", "Complexity Rank": 1,
             "Model": RandomForestRegressor(
                 n_estimators=500, max_depth=6, min_samples_leaf=6,
                 min_samples_split=8, max_features=0.6,
                 random_state=random_state, n_jobs=-1)},
            {"Configuration": "Depth 8; leaf 8", "Complexity Rank": 2,
             "Model": RandomForestRegressor(
                 n_estimators=500, max_depth=8, min_samples_leaf=8,
                 min_samples_split=8, max_features=0.6,
                 random_state=random_state, n_jobs=-1)},
            {"Configuration": "Max features sqrt", "Complexity Rank": 3,
             "Model": RandomForestRegressor(
                 n_estimators=500, max_depth=8, min_samples_leaf=4,
                 min_samples_split=8, max_features="sqrt",
                 random_state=random_state, n_jobs=-1)},
            {"Configuration": "Current (depth 8; leaf 4)",
             "Complexity Rank": 4,
             "Model": RandomForestRegressor(
                 n_estimators=500, max_depth=8, min_samples_leaf=4,
                 min_samples_split=8, max_features=0.6,
                 random_state=random_state, n_jobs=-1)},
            {"Configuration": "Depth 10; leaf 4", "Complexity Rank": 5,
             "Model": RandomForestRegressor(
                 n_estimators=500, max_depth=10, min_samples_leaf=4,
                 min_samples_split=8, max_features=0.6,
                 random_state=random_state, n_jobs=-1)},
        ],
        "Gradient Boosting": [
            {"Configuration": "Depth 1", "Complexity Rank": 1,
             "Model": GradientBoostingRegressor(
                 n_estimators=450, learning_rate=0.08, max_depth=1,
                 min_samples_leaf=3, max_features="sqrt", ccp_alpha=0.001,
                 random_state=random_state)},
            {"Configuration": "Leaf 10", "Complexity Rank": 2,
             "Model": GradientBoostingRegressor(
                 n_estimators=450, learning_rate=0.08, max_depth=2,
                 min_samples_leaf=10, max_features="sqrt", ccp_alpha=0.001,
                 random_state=random_state)},
            {"Configuration": "Leaf 5", "Complexity Rank": 3,
             "Model": GradientBoostingRegressor(
                 n_estimators=450, learning_rate=0.08, max_depth=2,
                 min_samples_leaf=5, max_features="sqrt", ccp_alpha=0.001,
                 random_state=random_state)},
            {"Configuration": "CCP alpha 0.005", "Complexity Rank": 4,
             "Model": GradientBoostingRegressor(
                 n_estimators=450, learning_rate=0.08, max_depth=2,
                 min_samples_leaf=3, max_features="sqrt", ccp_alpha=0.005,
                 random_state=random_state)},
            {"Configuration": "Subsample 0.8", "Complexity Rank": 4,
             "Model": GradientBoostingRegressor(
                 n_estimators=450, learning_rate=0.08, max_depth=2,
                 min_samples_leaf=3, max_features="sqrt", subsample=0.8,
                 ccp_alpha=0.001, random_state=random_state)},
            {"Configuration": "Current", "Complexity Rank": 5,
             "Model": GradientBoostingRegressor(
                 n_estimators=450, learning_rate=0.08, max_depth=2,
                 min_samples_leaf=3, max_features="sqrt", ccp_alpha=0.001,
                 random_state=random_state)},
            {"Configuration": "Learning rate 0.05; 600 estimators",
             "Complexity Rank": 5,
             "Model": GradientBoostingRegressor(
                 n_estimators=600, learning_rate=0.05, max_depth=2,
                 min_samples_leaf=3, max_features="sqrt", ccp_alpha=0.001,
                 random_state=random_state)},
        ],
        "MLP": [
            {"Configuration": "Compact; alpha 0.01", "Complexity Rank": 1,
             "Model": _mlp((16,), 0.01, random_state)},
            {"Configuration": "Compact; alpha 0.001", "Complexity Rank": 2,
             "Model": _mlp((16,), 0.001, random_state)},
            {"Configuration": "Layers 32-16; alpha 0.01", "Complexity Rank": 3,
             "Model": _mlp((32, 16), 0.01, random_state)},
            {"Configuration": "Current (layers 32-16; alpha 0.001)",
             "Complexity Rank": 4,
             "Model": _mlp((32, 16), 0.001, random_state)},
        ],
    }


def candidate_inventory(candidates):
    """Return display metadata without serialising estimator objects."""

    return pd.DataFrame([
        {
            "Model": family,
            "Configuration": candidate["Configuration"],
            "Complexity Rank": candidate["Complexity Rank"],
        }
        for family, family_candidates in candidates.items()
        for candidate in family_candidates
    ])


def _metrics(y_true, prediction):
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_true, prediction)),
        "R2": r2_score(y_true, prediction),
    }


def evaluate_candidate_family(
    family,
    candidates,
    development_df,
    features,
    target,
    group_column="date_group",
    n_splits=5,
):
    """Evaluate one model family on fixed date-grouped folds."""

    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(
        development_df[features],
        development_df[target],
        groups=development_df[group_column],
    ))
    records = []
    for candidate in candidates[family]:
        name = candidate["Configuration"]
        print(f"Starting {family} — {name}", flush=True)
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            estimator = clone(candidate["Model"])
            train = development_df.iloc[train_index]
            validation = development_df.iloc[validation_index]
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(train[features], train[target])
            fit_seconds = time.perf_counter() - started
            train_metrics = _metrics(
                train[target], estimator.predict(train[features])
            )
            validation_metrics = _metrics(
                validation[target], estimator.predict(validation[features])
            )
            records.append({
                "Model": family,
                "Configuration": name,
                "Complexity Rank": candidate["Complexity Rank"],
                "Fold": fold,
                "Train R2": train_metrics["R2"],
                "CV MAE": validation_metrics["MAE"],
                "CV RMSE": validation_metrics["RMSE"],
                "CV R2": validation_metrics["R2"],
                "Train-CV R2 Gap": train_metrics["R2"] - validation_metrics["R2"],
                "Fit Time (s)": fit_seconds,
                "Convergence Warnings": sum(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                ),
            })
    folds = pd.DataFrame(records)
    summary = (
        folds.groupby(
            ["Model", "Configuration", "Complexity Rank"], as_index=False
        )
        .agg(
            **{
                "Mean Train R2": ("Train R2", "mean"),
                "CV MAE": ("CV MAE", "mean"),
                "CV MAE Std": ("CV MAE", "std"),
                "CV RMSE": ("CV RMSE", "mean"),
                "CV RMSE Std": ("CV RMSE", "std"),
                "CV R2": ("CV R2", "mean"),
                "CV R2 Std": ("CV R2", "std"),
                "Train-CV R2 Gap": ("Train-CV R2 Gap", "mean"),
                "Fit Time (s)": ("Fit Time (s)", "sum"),
                "Convergence Warnings": ("Convergence Warnings", "sum"),
            }
        )
        .sort_values(["CV RMSE", "CV R2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary["CV RMSE SE"] = summary["CV RMSE Std"] / np.sqrt(n_splits)
    return summary, folds


def select_one_standard_error(candidate_summary, n_splits=5):
    """Select the simplest candidate within one SE of the best CV RMSE."""

    selections = []
    annotated = candidate_summary.copy()
    annotated["Within one SE"] = False
    for family, family_results in annotated.groupby("Model", sort=False):
        best_index = family_results["CV RMSE"].idxmin()
        threshold = (
            annotated.loc[best_index, "CV RMSE"]
            + annotated.loc[best_index, "CV RMSE Std"] / np.sqrt(n_splits)
        )
        eligible_index = family_results.index[
            family_results["CV RMSE"].le(threshold)
        ]
        annotated.loc[eligible_index, "Within one SE"] = True
        selected_index = (
            annotated.loc[eligible_index]
            .sort_values(
                ["Complexity Rank", "Train-CV R2 Gap", "CV RMSE"],
                ascending=[True, True, True],
            )
            .index[0]
        )
        selected = annotated.loc[selected_index].copy()
        selected["One-SE RMSE Threshold"] = threshold
        selections.append(selected)
    return annotated, pd.DataFrame(selections).reset_index(drop=True)


def selected_estimators(candidates, selections):
    """Recover estimator objects corresponding to selected configuration rows."""

    selected = {}
    for row in selections.itertuples(index=False):
        family = row.Model
        configuration = row.Configuration
        selected[family] = next(
            clone(candidate["Model"])
            for candidate in candidates[family]
            if candidate["Configuration"] == configuration
        )
    return selected


def _set_random_state(estimator, seed):
    estimator = clone(estimator)
    parameters = estimator.get_params(deep=True)
    if "random_state" in parameters:
        return estimator.set_params(random_state=seed)
    if "model__random_state" in parameters:
        return estimator.set_params(model__random_state=seed)
    raise ValueError("Estimator does not expose a random_state parameter.")


def evaluate_seed_sensitivity(
    selected_models,
    model_seeds,
    development_df,
    features,
    target,
    group_column="date_group",
    n_splits=5,
):
    """Evaluate selected stochastic models across pre-specified model seeds."""

    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(
        development_df[features],
        development_df[target],
        groups=development_df[group_column],
    ))
    records = []
    for family, base_model in selected_models.items():
        for seed in model_seeds:
            print(f"Starting {family} — model seed {seed}", flush=True)
            for fold, (train_index, validation_index) in enumerate(splits, start=1):
                estimator = _set_random_state(base_model, seed)
                train = development_df.iloc[train_index]
                validation = development_df.iloc[validation_index]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(train[features], train[target])
                train_r2 = r2_score(
                    train[target], estimator.predict(train[features])
                )
                metrics = _metrics(
                    validation[target], estimator.predict(validation[features])
                )
                records.append({
                    "Model": family,
                    "Model Seed": seed,
                    "Fold": fold,
                    "CV MAE": metrics["MAE"],
                    "CV RMSE": metrics["RMSE"],
                    "CV R2": metrics["R2"],
                    "Train-CV R2 Gap": train_r2 - metrics["R2"],
                    "Convergence Warnings": sum(
                        issubclass(item.category, ConvergenceWarning)
                        for item in caught
                    ),
                })
    folds = pd.DataFrame(records)
    by_seed = (
        folds.groupby(["Model", "Model Seed"], as_index=False)
        .agg(
            **{
                "CV MAE": ("CV MAE", "mean"),
                "CV RMSE": ("CV RMSE", "mean"),
                "CV R2": ("CV R2", "mean"),
                "Train-CV R2 Gap": ("Train-CV R2 Gap", "mean"),
                "Convergence Warnings": ("Convergence Warnings", "sum"),
            }
        )
    )
    return by_seed, folds


def representative_model_seeds(seed_summary, reference_seed=42):
    """Retain the reference seed unless it is an across-seed RMSE outlier."""

    records = []
    for family, results in seed_summary.groupby("Model", sort=False):
        median_rmse = results["CV RMSE"].median()
        rmse_std = results["CV RMSE"].std()
        closest_to_median = (
            results.assign(
                Distance_from_median=(results["CV RMSE"] - median_rmse).abs()
            )
            .sort_values(["Distance_from_median", "Model Seed"])
            .iloc[0]
        )
        reference_rows = results.loc[results["Model Seed"].eq(reference_seed)]
        reference_is_representative = False
        if not reference_rows.empty:
            reference_row = reference_rows.iloc[0]
            tolerance = 0.0 if pd.isna(rmse_std) else rmse_std
            reference_is_representative = (
                abs(reference_row["CV RMSE"] - median_rmse) <= tolerance
            )
        selected = (
            reference_row if reference_is_representative else closest_to_median
        )
        records.append({
            "Model": family,
            "Representative Model Seed": int(selected["Model Seed"]),
            "Selection Basis": (
                "Reference seed within 1 across-seed SD of median"
                if reference_is_representative
                else "Closest tested seed to median CV RMSE"
            ),
            "Median CV RMSE": median_rmse,
            "Selected-seed CV RMSE": selected["CV RMSE"],
            "Selected-seed CV R2": selected["CV R2"],
            "Across-seed CV RMSE Std": rmse_std,
            "Across-seed CV R2 Std": results["CV R2"].std(),
        })
    return pd.DataFrame(records)
