import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def get_benchmark_models(random_state=42):
    return {
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.08,
            max_depth=4,
            min_samples_leaf=5,
            min_samples_split=2,
            max_features="sqrt",
            subsample=1.0,
            random_state=random_state,
        ),

        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=0.0001,
                        learning_rate_init=0.001,
                        max_iter=1000,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def get_model_summary(n_features=None):
    """Return display metadata for the configured benchmark models."""
    input_description = (
        f"{n_features} engineered features"
        if n_features is not None
        else "Engineered features"
    )
    return pd.DataFrame(
        {
            "Model": [
                "Ridge",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "MLP",
            ],
            "Method Category": [
                "Conventional ML",
                "Conventional ML",
                "Conventional ML",
                "Conventional ML",
                "Feature-based DNN",
            ],
            "Model Family": [
                "Linear regression",
                "Single regression tree",
                "Bagging ensemble",
                "Boosting ensemble",
                "Multilayer neural network",
            ],
            "Input": [
                input_description,
                input_description,
                input_description,
                input_description,
                input_description,
            ],
            "Scaling": [
                "Standardised",
                "Not required",
                "Not required",
                "Not required",
                "Standardised",
            ],
        }
    )


def random_split(
    df,
    test_size=0.2,
    random_state=42,
):
    development_df, test_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    return (
        development_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def evaluate_regression(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_model(
    model,
    model_name,
    development_df,
    test_df,
    features,
    target,
    n_splits=5,
    random_state=42,
):
    X_dev = development_df[features].reset_index(drop=True)
    y_dev = development_df[target].reset_index(drop=True)

    X_test = test_df[features].reset_index(drop=True)
    y_test = test_df[target].reset_index(drop=True)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(X_dev),
        start=1,
    ):
        fold_model = clone(model)

        X_train = X_dev.iloc[train_idx]
        X_val = X_dev.iloc[val_idx]

        y_train = y_dev.iloc[train_idx]
        y_val = y_dev.iloc[val_idx]

        fold_model.fit(X_train, y_train)

        y_train_pred = fold_model.predict(X_train)
        y_val_pred = fold_model.predict(X_val)

        train_metrics = evaluate_regression(
            y_train,
            y_train_pred,
        )

        validation_metrics = evaluate_regression(
            y_val,
            y_val_pred,
        )

        fold_results.append(
            {
                "Fold": fold,

                "Train MAE": train_metrics["MAE"],
                "Train RMSE": train_metrics["RMSE"],
                "Train R2": train_metrics["R2"],

                "Validation MAE": validation_metrics["MAE"],
                "Validation RMSE": validation_metrics["RMSE"],
                "Validation R2": validation_metrics["R2"],

                # Retain existing names for compatibility
                "MAE": validation_metrics["MAE"],
                "RMSE": validation_metrics["RMSE"],
                "R2": validation_metrics["R2"],
            }
        )

    fold_results = pd.DataFrame(fold_results)

    final_model = clone(model)

    train_start = time.perf_counter()
    final_model.fit(X_dev, y_dev)
    train_time = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    y_test_pred = final_model.predict(X_test)
    predict_time = time.perf_counter() - predict_start

    test_metrics = evaluate_regression(
        y_test,
        y_test_pred,
    )

    summary = pd.DataFrame(
        [
            {
                "Model": model_name,
                "CV MAE": fold_results["MAE"].mean(),
                "CV MAE Std": fold_results["MAE"].std(),
                "CV RMSE": fold_results["RMSE"].mean(),
                "CV RMSE Std": fold_results["RMSE"].std(),
                "CV R2": fold_results["R2"].mean(),
                "CV R2 Std": fold_results["R2"].std(),
                "Test MAE": test_metrics["MAE"],
                "Test RMSE": test_metrics["RMSE"],
                "Test R2": test_metrics["R2"],
                "Train Time (s)": train_time,
                "Predict Time (s)": predict_time,
            }
        ]
    )

    predictions = pd.DataFrame(
        {
            "Observed": y_test.to_numpy(),
            "Predicted": y_test_pred,
            "Residual": (
                y_test.to_numpy()
                - y_test_pred
            ),
        }
    )

    return (
        summary,
        fold_results,
        final_model,
        predictions,
    )


def benchmark_models(
    models,
    development_df,
    test_df,
    features,
    target,
    n_splits=5,
    random_state=42,
):
    summaries = []
    fold_results = {}
    fitted_models = {}
    predictions = {}

    for model_name, model in models.items():
        (
            summary,
            model_folds,
            fitted_model,
            model_predictions,
        ) = evaluate_model(
            model=model,
            model_name=model_name,
            development_df=development_df,
            test_df=test_df,
            features=features,
            target=target,
            n_splits=n_splits,
            random_state=random_state,
        )

        summaries.append(summary)
        fold_results[model_name] = model_folds
        fitted_models[model_name] = fitted_model
        predictions[model_name] = model_predictions

    benchmark_results = pd.concat(
        summaries,
        ignore_index=True,
    )

    return (
        benchmark_results,
        fold_results,
        fitted_models,
        predictions,
    )

def get_best_model_name(
    results,
    metric="Test R2",
):
    return results.loc[
        results[metric].idxmax(),
        "Model",
    ]
