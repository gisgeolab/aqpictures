"""Reusable evaluation utilities for feature and split-strategy experiments."""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline


def random_split(df, test_size=0.2, random_state=42):
    """Return the original row-wise random split used as a reference experiment."""
    development_df, test_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    return development_df.reset_index(drop=True), test_df.reset_index(drop=True)


def balanced_month_group_split(
    df,
    time_column="time",
    test_size=0.2,
    random_state=42,
):
    """Allocate complete dates to development/test within each calendar month.

    Month imbalance is handled later with inverse-frequency sample weights,
    which retains the longest possible observation period without allowing
    months with greater webcam coverage to dominate model fitting or metrics.
    """
    working = df.copy()
    timestamp = pd.to_datetime(working[time_column])
    working["__month"] = timestamp.dt.month
    working["__date"] = timestamp.dt.normalize()
    rng = np.random.default_rng(random_state)
    test_dates = []

    for _, month_df in working.groupby("__month"):
        dates = month_df["__date"].drop_duplicates().to_numpy(copy=True)
        rng.shuffle(dates)
        if len(dates) < 2:
            raise ValueError("Each represented month needs at least two dates.")
        n_test = min(max(1, int(round(len(dates) * test_size))), len(dates) - 1)
        test_dates.extend(dates[:n_test])

    is_test = working["__date"].isin(test_dates)
    development = working.loc[~is_test].sort_values(time_column).copy()
    test = working.loc[is_test].sort_values(time_column).copy()
    helper_columns = ["__month", "__date"]
    return (
        development.drop(columns=helper_columns).reset_index(drop=True),
        test.drop(columns=helper_columns).reset_index(drop=True),
    )


def temporal_split(df, time_column="time", test_size=0.2):
    """Use the most recent continuous dates as the test set."""
    ordered = df.sort_values(time_column).reset_index(drop=True)
    dates = pd.to_datetime(ordered[time_column]).dt.normalize()
    unique_dates = dates.drop_duplicates().to_numpy()
    split_index = int(np.floor(len(unique_dates) * (1 - test_size)))
    test_start = unique_dates[split_index]
    development = ordered.loc[dates < test_start].copy()
    test = ordered.loc[dates >= test_start].copy()
    return development, test


def evaluate_regression(y_true, y_pred, sample_weight=None):
    return {
        "MAE": mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
        "R2": r2_score(y_true, y_pred, sample_weight=sample_weight),
    }


def inverse_frequency_weights(values):
    """Give every represented category equal total weight."""
    values = pd.Series(values).reset_index(drop=True)
    counts = values.value_counts()
    weights = values.map(1.0 / counts)
    return (weights / weights.mean()).to_numpy()


def _fit_model(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        final_step = model.steps[-1][0]
        return model.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def _fold_iterator(development_df, n_splits, random_state, group_column):
    if group_column is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return splitter.split(development_df)

    groups = development_df[group_column]
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(development_df, groups=groups)


def evaluate_feature_set(
    model,
    model_name,
    feature_set_name,
    feature_list,
    development_df,
    test_df,
    target,
    n_splits=5,
    random_state=42,
    group_column=None,
    balance_column=None,
    evaluate_test=True,
):
    """Evaluate one feature set using CV and an independent hold-out set."""
    X_dev = development_df[feature_list].reset_index(drop=True)
    y_dev = development_df[target].reset_index(drop=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        _fold_iterator(development_df, n_splits, random_state, group_column),
        start=1,
    ):
        fold_model = clone(model)
        train_weights = None
        validation_weights = None
        if balance_column is not None:
            train_weights = inverse_frequency_weights(
                development_df.iloc[train_idx][balance_column]
            )
            validation_weights = inverse_frequency_weights(
                development_df.iloc[val_idx][balance_column]
            )
        _fit_model(
            fold_model, X_dev.iloc[train_idx], y_dev.iloc[train_idx], train_weights
        )
        y_val_pred = fold_model.predict(X_dev.iloc[val_idx])
        fold_results.append(
            {"Fold": fold, **evaluate_regression(
                y_dev.iloc[val_idx], y_val_pred, validation_weights
            )}
        )

    fold_results = pd.DataFrame(fold_results)
    summary_values = {
            "Model": model_name,
            "Feature Set": feature_set_name,
            "Variables": len(feature_list),
            "CV MAE": fold_results["MAE"].mean(),
            "CV MAE Std": fold_results["MAE"].std(),
            "CV RMSE": fold_results["RMSE"].mean(),
            "CV RMSE Std": fold_results["RMSE"].std(),
            "CV R2": fold_results["R2"].mean(),
            "CV R2 Std": fold_results["R2"].std(),
    }
    y_test = None
    y_test_pred = None
    if evaluate_test:
        X_test = test_df[feature_list].reset_index(drop=True)
        y_test = test_df[target].reset_index(drop=True)
        final_model = clone(model)
        development_weights = (
            inverse_frequency_weights(development_df[balance_column])
            if balance_column is not None else None
        )
        test_weights = (
            inverse_frequency_weights(test_df[balance_column])
            if balance_column is not None else None
        )
        _fit_model(final_model, X_dev, y_dev, development_weights)
        y_test_pred = final_model.predict(X_test)
        test_metrics = evaluate_regression(y_test, y_test_pred, test_weights)
        summary_values.update({f"Test {key}": value for key, value in test_metrics.items()})
    summary = pd.DataFrame([summary_values])
    return summary, fold_results, y_test, y_test_pred


def evaluate_holdout(
    model,
    feature_list,
    development_df,
    test_df,
    target,
    balance_column=None,
):
    """Fit once on a development set and evaluate one held-out set."""
    fitted_model = clone(model)
    development_weights = (
        inverse_frequency_weights(development_df[balance_column])
        if balance_column is not None else None
    )
    test_weights = (
        inverse_frequency_weights(test_df[balance_column])
        if balance_column is not None else None
    )
    _fit_model(
        fitted_model,
        development_df[feature_list],
        development_df[target],
        development_weights,
    )
    prediction = fitted_model.predict(test_df[feature_list])
    metrics = evaluate_regression(test_df[target], prediction, test_weights)
    return metrics, prediction


def clean_reference_normalize(
    reference_df,
    transform_df,
    feature_list,
    target,
    clean_quantile=0.2,
    condition_columns=("season", "is_daytime"),
    minimum_reference_samples=10,
):
    """Normalize image features against clean-air references fitted on training data."""
    threshold = reference_df[target].quantile(clean_quantile)
    clean = reference_df.loc[reference_df[target] <= threshold]
    transformed = transform_df.copy()
    global_median = clean[feature_list].median()
    global_iqr = clean[feature_list].quantile(0.75) - clean[feature_list].quantile(0.25)
    global_iqr = global_iqr.replace(0, 1.0)

    for feature in feature_list:
        transformed[feature] = (
            transformed[feature] - global_median[feature]
        ) / global_iqr[feature]

    grouped_clean = clean.groupby(list(condition_columns), dropna=False)
    for condition, group in grouped_clean:
        if len(group) < minimum_reference_samples:
            continue
        if not isinstance(condition, tuple):
            condition = (condition,)
        mask = pd.Series(True, index=transform_df.index)
        for column, value in zip(condition_columns, condition):
            mask &= transform_df[column].eq(value)
        median = group[feature_list].median()
        iqr = group[feature_list].quantile(0.75) - group[feature_list].quantile(0.25)
        iqr = iqr.where(iqr.ne(0), global_iqr)
        transformed.loc[mask, feature_list] = (
            transform_df.loc[mask, feature_list] - median
        ) / iqr

    return transformed, threshold


def evaluate_clean_reference_feature_set(
    model,
    model_name,
    feature_list,
    development_df,
    test_df,
    target,
    n_splits=5,
    random_state=42,
    group_column=None,
    balance_column=None,
    clean_quantile=0.2,
    evaluate_test=False,
):
    """Evaluate clean-reference normalization fitted independently in every fold."""
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(
        _fold_iterator(development_df, n_splits, random_state, group_column),
        start=1,
    ):
        train = development_df.iloc[train_idx]
        validation = development_df.iloc[val_idx]
        train_norm, _ = clean_reference_normalize(
            train, train, feature_list, target, clean_quantile
        )
        validation_norm, _ = clean_reference_normalize(
            train, validation, feature_list, target, clean_quantile
        )
        fold_model = clone(model)
        train_weights = (
            inverse_frequency_weights(train[balance_column])
            if balance_column is not None else None
        )
        validation_weights = (
            inverse_frequency_weights(validation[balance_column])
            if balance_column is not None else None
        )
        _fit_model(
            fold_model, train_norm[feature_list], train[target], train_weights
        )
        prediction = fold_model.predict(validation_norm[feature_list])
        fold_results.append(
            {"Fold": fold, **evaluate_regression(
                validation[target], prediction, validation_weights
            )}
        )

    fold_results = pd.DataFrame(fold_results)
    _, threshold = clean_reference_normalize(
        development_df, development_df, feature_list, target, clean_quantile
    )
    summary_values = {
            "Model": model_name,
            "Feature Set": "Clean-reference normalized image",
            "Variables": len(feature_list),
            "Clean threshold": threshold,
            "CV MAE": fold_results["MAE"].mean(),
            "CV MAE Std": fold_results["MAE"].std(),
            "CV RMSE": fold_results["RMSE"].mean(),
            "CV RMSE Std": fold_results["RMSE"].std(),
            "CV R2": fold_results["R2"].mean(),
            "CV R2 Std": fold_results["R2"].std(),
    }
    y_test = None
    test_prediction = None
    if evaluate_test:
        development_norm, _ = clean_reference_normalize(
            development_df, development_df, feature_list, target, clean_quantile
        )
        test_norm, _ = clean_reference_normalize(
            development_df, test_df, feature_list, target, clean_quantile
        )
        final_model = clone(model)
        development_weights = (
            inverse_frequency_weights(development_df[balance_column])
            if balance_column is not None else None
        )
        _fit_model(
            final_model,
            development_norm[feature_list],
            development_df[target],
            development_weights,
        )
        test_prediction = final_model.predict(test_norm[feature_list])
        y_test = test_df[target].reset_index(drop=True)
        test_weights = (
            inverse_frequency_weights(test_df[balance_column])
            if balance_column is not None else None
        )
        test_metrics = evaluate_regression(y_test, test_prediction, test_weights)
        summary_values.update({f"Test {key}": value for key, value in test_metrics.items()})
    summary = pd.DataFrame([summary_values])
    return summary, fold_results, y_test, test_prediction
