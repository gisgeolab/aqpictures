"""Reusable evaluation utilities for feature and split-strategy experiments."""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_feature_evaluation_models(random_state=42):
    """Return the analytical models used for Notebook 3 feature evaluation."""

    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=100.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=8,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def balanced_month_group_split(
    df,
    time_column="time",
    test_size=0.2,
    random_state=42,
):
    """Allocate complete dates within each observed year-month.

    This preserves coverage across the full observation period and prevents
    records from the same date appearing in both development and hold-out data.
    """
    working = df.copy()
    timestamp = pd.to_datetime(working[time_column])
    working["__year_month"] = timestamp.dt.to_period("M")
    working["__date"] = timestamp.dt.normalize()
    rng = np.random.default_rng(random_state)
    test_dates = []

    for _, month_df in working.groupby("__year_month"):
        dates = month_df["__date"].drop_duplicates().to_numpy(copy=True)
        rng.shuffle(dates)
        if len(dates) < 2:
            raise ValueError("Each represented month needs at least two dates.")
        n_test = min(max(1, int(round(len(dates) * test_size))), len(dates) - 1)
        test_dates.extend(dates[:n_test])

    is_test = working["__date"].isin(test_dates)
    development = working.loc[~is_test].sort_values(time_column).copy()
    test = working.loc[is_test].sort_values(time_column).copy()
    helper_columns = ["__year_month", "__date"]
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
    """Return absolute and dimensionless regression metrics.

    MAPE and SMAPE are expressed as percentages. NRMSE is normalised by the
    (optionally weighted) mean observed concentration and is also expressed as
    a percentage. Zero observations are excluded from MAPE only; SMAPE assigns
    a zero contribution when both the observation and prediction are zero.
    """
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    absolute_error = np.abs(observed - predicted)
    rmse = np.sqrt(
        mean_squared_error(observed, predicted, sample_weight=weights)
    )

    nonzero_observed = np.abs(observed) > np.finfo(float).eps
    if nonzero_observed.any():
        mape_weights = None if weights is None else weights[nonzero_observed]
        mape = 100 * np.average(
            absolute_error[nonzero_observed]
            / np.abs(observed[nonzero_observed]),
            weights=mape_weights,
        )
    else:
        mape = np.nan

    smape_denominator = np.abs(observed) + np.abs(predicted)
    smape_terms = np.divide(
        2 * absolute_error,
        smape_denominator,
        out=np.zeros_like(absolute_error),
        where=smape_denominator > np.finfo(float).eps,
    )
    smape = 100 * np.average(smape_terms, weights=weights)

    mean_observed = np.average(observed, weights=weights)
    nrmse = (
        100 * rmse / mean_observed
        if np.abs(mean_observed) > np.finfo(float).eps
        else np.nan
    )

    return {
        "MAE": mean_absolute_error(observed, predicted, sample_weight=weights),
        "RMSE": rmse,
        "R2": r2_score(observed, predicted, sample_weight=weights),
        "MAPE": mape,
        "SMAPE": smape,
        "NRMSE": nrmse,
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
    include_train_metrics=False,
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
        fold_record = {
            "Fold": fold,
            **evaluate_regression(
                y_dev.iloc[val_idx],
                y_val_pred,
                validation_weights,
            ),
        }
        if include_train_metrics:
            y_train_pred = fold_model.predict(X_dev.iloc[train_idx])
            train_metrics = evaluate_regression(
                y_dev.iloc[train_idx],
                y_train_pred,
                train_weights,
            )
            fold_record.update(
                {
                    f"Train {metric}": value
                    for metric, value in train_metrics.items()
                }
            )
        fold_results.append(
            fold_record
        )

    fold_results = pd.DataFrame(fold_results)
    summary_values = {
        "Model": model_name,
        "Feature Set": feature_set_name,
        "Variables": len(feature_list),
    }
    for metric in ("MAE", "RMSE", "R2", "MAPE", "SMAPE", "NRMSE"):
        summary_values[f"CV {metric}"] = fold_results[metric].mean()
        summary_values[f"CV {metric} Std"] = fold_results[metric].std()
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


def build_leave_one_source_out_sets(feature_groups):
    """Build the complete and leave-one-source-out feature configurations."""
    all_features = [
        feature
        for features in feature_groups.values()
        for feature in features
    ]
    if len(all_features) != len(set(all_features)):
        raise ValueError("Feature groups contain duplicate variables.")

    feature_sets = {"All Features": all_features}
    for source, removed_features in feature_groups.items():
        removed_features = set(removed_features)
        feature_sets[f"All - {source}"] = [
            feature
            for feature in all_features
            if feature not in removed_features
        ]
    return feature_sets


def evaluate_leave_one_source_out(
    model,
    model_name,
    feature_sets,
    development_df,
    target,
    n_splits=5,
    random_state=42,
    group_column=None,
    balance_column=None,
    baseline_name="All Features",
):
    """Evaluate source removal and return paired fold-level changes."""
    if baseline_name not in feature_sets:
        raise ValueError(f"Missing baseline feature set: {baseline_name}")

    summaries = []
    fold_tables = []

    for feature_set_name, feature_list in feature_sets.items():
        summary, fold_results, _, _ = evaluate_feature_set(
            model=model,
            model_name=model_name,
            feature_set_name=feature_set_name,
            feature_list=feature_list,
            development_df=development_df,
            test_df=development_df,
            target=target,
            n_splits=n_splits,
            random_state=random_state,
            group_column=group_column,
            balance_column=balance_column,
            evaluate_test=False,
        )
        summaries.append(summary)
        fold_tables.append(
            fold_results.assign(**{"Feature Set": feature_set_name})
        )

    results = pd.concat(summaries, ignore_index=True)
    fold_results = pd.concat(fold_tables, ignore_index=True)

    baseline_row = results.loc[
        results["Feature Set"].eq(baseline_name)
    ].iloc[0]
    results["CV RMSE Change"] = (
        results["CV RMSE"] - baseline_row["CV RMSE"]
    )
    results["CV R2 Change"] = (
        results["CV R2"] - baseline_row["CV R2"]
    )

    baseline_folds = (
        fold_results.loc[
            fold_results["Feature Set"].eq(baseline_name),
            ["Fold", "RMSE", "R2"],
        ]
        .rename(
            columns={
                "RMSE": "All Features RMSE",
                "R2": "All Features R2",
            }
        )
    )
    fold_changes = (
        fold_results.loc[
            ~fold_results["Feature Set"].eq(baseline_name)
        ]
        .merge(
            baseline_folds,
            on="Fold",
            how="left",
            validate="many_to_one",
        )
    )
    fold_changes["Removed Source"] = (
        fold_changes["Feature Set"]
        .str.replace("All - ", "", regex=False)
    )
    fold_changes["RMSE Change"] = (
        fold_changes["RMSE"] - fold_changes["All Features RMSE"]
    )
    fold_changes["R2 Change"] = (
        fold_changes["R2"] - fold_changes["All Features R2"]
    )

    change_summary = (
        fold_changes
        .groupby("Removed Source", as_index=False)
        .agg(
            **{
                "RMSE Change Mean": ("RMSE Change", "mean"),
                "RMSE Change Std": ("RMSE Change", "std"),
                "R2 Change Mean": ("R2 Change", "mean"),
                "R2 Change Std": ("R2 Change", "std"),
            }
        )
    )
    return results, fold_results, fold_changes, change_summary


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
    clean_limit=20.0,
    condition_columns=("month", "is_daytime"),
    minimum_reference_samples=10,
):
    """Normalize image features using training-only clean-air references."""

    clean_reference = reference_df.loc[reference_df[target].le(clean_limit)]
    if clean_reference.empty:
        raise ValueError("No clean-air reference observations are available.")

    transformed = transform_df.copy()
    global_median = clean_reference[feature_list].median()
    global_iqr = (
        clean_reference[feature_list].quantile(0.75)
        - clean_reference[feature_list].quantile(0.25)
    ).replace(0, 1.0)
    transformed.loc[:, feature_list] = (
        transform_df[feature_list] - global_median
    ) / global_iqr

    grouped_reference = clean_reference.groupby(
        list(condition_columns), dropna=False
    )
    for condition, group in grouped_reference:
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

    return transformed, len(clean_reference)


def evaluate_clean_reference_comparison(
    model,
    model_name,
    model_features,
    reference_features,
    development_df,
    test_df,
    target,
    n_splits=5,
    random_state=42,
    group_column=None,
    clean_limit=20.0,
    condition_columns=("month", "is_daytime"),
    minimum_reference_samples=10,
):
    """Compare raw and clean-reference features on identical grouped folds."""

    required = set(model_features) | set(reference_features) | {
        target, *condition_columns,
    }
    for name, frame in (("development", development_df), ("test", test_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")

    fold_records = []
    prediction_frames = []
    for fold, (train_idx, validation_idx) in enumerate(
        _fold_iterator(development_df, n_splits, random_state, group_column),
        start=1,
    ):
        train = development_df.iloc[train_idx]
        validation = development_df.iloc[validation_idx]

        raw_model = clone(model)
        raw_model.fit(train[model_features], train[target])
        raw_prediction = raw_model.predict(validation[model_features])

        normalized_train, reference_samples = clean_reference_normalize(
            train,
            train,
            reference_features,
            target,
            clean_limit=clean_limit,
            condition_columns=condition_columns,
            minimum_reference_samples=minimum_reference_samples,
        )
        normalized_validation, _ = clean_reference_normalize(
            train,
            validation,
            reference_features,
            target,
            clean_limit=clean_limit,
            condition_columns=condition_columns,
            minimum_reference_samples=minimum_reference_samples,
        )
        normalized_model = clone(model)
        normalized_model.fit(normalized_train[model_features], train[target])
        normalized_prediction = normalized_model.predict(
            normalized_validation[model_features]
        )

        for representation, prediction in (
            ("Raw", raw_prediction),
            ("Clean-reference normalized", normalized_prediction),
        ):
            fold_records.append({
                "Fold": fold,
                "Representation": representation,
                "Clean reference samples": reference_samples,
                **evaluate_regression(validation[target], prediction),
            })
        prediction_frames.append(pd.DataFrame({
            "Fold": fold,
            "Observed": validation[target].to_numpy(),
            "Raw Predicted": raw_prediction,
            "Normalized Predicted": normalized_prediction,
        }))

    fold_results = pd.DataFrame(fold_records)
    cv_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_records = []
    for representation, group in fold_results.groupby("Representation", sort=False):
        summary_records.append({
            "Model": model_name,
            "Representation": representation,
            "Variables": len(model_features),
            "Clean-air limit": clean_limit,
            "CV MAE": group["MAE"].mean(),
            "CV MAE Std": group["MAE"].std(),
            "CV RMSE": group["RMSE"].mean(),
            "CV RMSE Std": group["RMSE"].std(),
            "CV R2": group["R2"].mean(),
            "CV R2 Std": group["R2"].std(),
        })
    summary = pd.DataFrame(summary_records)

    raw_model = clone(model)
    raw_model.fit(development_df[model_features], development_df[target])
    raw_test_prediction = raw_model.predict(test_df[model_features])
    normalized_development, _ = clean_reference_normalize(
        development_df,
        development_df,
        reference_features,
        target,
        clean_limit=clean_limit,
        condition_columns=condition_columns,
        minimum_reference_samples=minimum_reference_samples,
    )
    normalized_test, _ = clean_reference_normalize(
        development_df,
        test_df,
        reference_features,
        target,
        clean_limit=clean_limit,
        condition_columns=condition_columns,
        minimum_reference_samples=minimum_reference_samples,
    )
    normalized_model = clone(model)
    normalized_model.fit(
        normalized_development[model_features], development_df[target]
    )
    normalized_test_prediction = normalized_model.predict(
        normalized_test[model_features]
    )

    for representation, prediction in (
        ("Raw", raw_test_prediction),
        ("Clean-reference normalized", normalized_test_prediction),
    ):
        metrics = evaluate_regression(test_df[target], prediction)
        mask = summary["Representation"].eq(representation)
        for metric, value in metrics.items():
            summary.loc[mask, f"Test {metric}"] = value

    test_predictions = pd.DataFrame({
        "Observed": test_df[target].to_numpy(),
        "Raw Predicted": raw_test_prediction,
        "Normalized Predicted": normalized_test_prediction,
    })
    return summary, fold_results, cv_predictions, test_predictions
