import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def get_benchmark_models(random_state=42):
    return {
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=100.0)),
            ]
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=25,
            random_state=random_state,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=8,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=450,
            learning_rate=0.08,
            max_depth=2,
            min_samples_leaf=5,
            min_samples_split=2,
            max_features="sqrt",
            subsample=1.0,
            max_leaf_nodes=None,
            ccp_alpha=0.001,
            random_state=random_state,
        ),

        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(16,),
                        activation="relu",
                        alpha=1.0,
                        learning_rate_init=0.001,
                        max_iter=3000,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def get_model_summary():
    """Return display metadata for the configured benchmark models."""
    return pd.DataFrame(
        {
            "Model": [
                "Ridge",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "MLP",
            ],
            "Model family": [
                "Linear regression",
                "Regression tree",
                "Bagging ensemble",
                "Boosting ensemble",
                "Feed-forward neural network",
            ],
            "Scaling": [
                "Standardised",
                "Not required",
                "Not required",
                "Not required",
                "Standardised",
            ],
            "Key configuration": [
                "alpha=100",
                "max_depth=4; min_samples_leaf=25",
                "n_estimators=500; max_depth=8; max_features=sqrt",
                "n_estimators=450; learning_rate=0.08; max_depth=2; min_samples_leaf=5",
                "layers=(16,); alpha=1.0; max_iter=3000",
            ],
        }
    )


def summarize_clean_reference_by_concentration(
    predictions,
    thresholds=(20.0, 35.0),
):
    """Summarize raw and clean-reference CV errors by observed concentration."""

    required_columns = {"Observed", "Raw Predicted", "Normalized Predicted"}
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    low, high = (float(value) for value in thresholds)
    if low >= high:
        raise ValueError("Concentration thresholds must be increasing.")

    range_order = [f"≤{low:g}", f"{low:g}–{high:g}", f">{high:g}"]
    frames = []
    for representation, prediction_column in {
        "Raw": "Raw Predicted",
        "Clean-reference normalized": "Normalized Predicted",
    }.items():
        frame = predictions[["Observed", prediction_column]].copy()
        frame = frame.rename(columns={prediction_column: "Predicted"})
        frame["Representation"] = representation
        frame["Error"] = frame["Predicted"] - frame["Observed"]
        frame["Absolute Error"] = frame["Error"].abs()
        frame["PM2.5 Range"] = pd.cut(
            frame["Observed"],
            bins=[-np.inf, low, high, np.inf],
            labels=range_order,
            include_lowest=True,
        )
        frames.append(frame)

    long_summary = (
        pd.concat(frames, ignore_index=True)
        .groupby(["PM2.5 Range", "Representation"], observed=False)
        .agg(
            Samples=("Observed", "size"),
            MAE=("Absolute Error", "mean"),
            Mean_Bias=("Error", "mean"),
        )
        .reset_index()
        .rename(columns={"Mean_Bias": "Mean Bias"})
    )

    display_summary = pd.DataFrame({"PM2.5 Range": range_order}).set_index(
        "PM2.5 Range"
    )
    for representation, prefix in {
        "Raw": "Raw",
        "Clean-reference normalized": "Normalized",
    }.items():
        values = (
            long_summary.loc[
                long_summary["Representation"].eq(representation)
            ]
            .set_index("PM2.5 Range")
            .reindex(range_order)
        )
        display_summary[f"{prefix} MAE"] = values["MAE"]
        display_summary[f"{prefix} bias"] = values["Mean Bias"]

    sample_counts = (
        long_summary.drop_duplicates("PM2.5 Range")
        .set_index("PM2.5 Range")["Samples"]
        .reindex(range_order)
    )
    display_summary.insert(0, "Samples", sample_counts)
    display_summary.insert(
        3,
        "MAE change",
        display_summary["Normalized MAE"] - display_summary["Raw MAE"],
    )
    display_summary = display_summary[[
        "Samples", "Raw MAE", "Normalized MAE", "MAE change",
        "Raw bias", "Normalized bias",
    ]]
    return long_summary, display_summary.reset_index()
