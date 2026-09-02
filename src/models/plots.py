"""Publication figures used by the traditional-model benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.plot_style import COLORS, style_axes


def plot_dimensionless_metric_comparison(
    benchmark_results,
    output_path,
    model_order=None,
    figure_size=(9.2, 4.8),
    dpi=300,
):
    """Plot MAPE, SMAPE, and NRMSE as grouped CV bars on one y-axis."""

    required = [
        "Model", "CV MAPE", "CV MAPE Std", "CV SMAPE", "CV SMAPE Std",
        "CV NRMSE", "CV NRMSE Std",
    ]
    missing = [column for column in required if column not in benchmark_results]
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")

    plot_df = benchmark_results[required].copy()
    if model_order is not None:
        order = [model for model in model_order if model in plot_df["Model"].tolist()]
        order += [model for model in plot_df["Model"] if model not in order]
        plot_df["Model"] = pd.Categorical(plot_df["Model"], order, ordered=True)
        plot_df = plot_df.sort_values("Model").reset_index(drop=True)

    models = plot_df["Model"].astype(str).tolist()
    labels = [
        "MAPE", "SMAPE", "NRMSE",
    ]
    value_columns = ["CV MAPE", "CV SMAPE", "CV NRMSE"]
    std_columns = ["CV MAPE Std", "CV SMAPE Std", "CV NRMSE Std"]
    colours = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]

    fig, ax = plt.subplots(figsize=figure_size)
    positions = np.arange(len(plot_df))
    width = 0.24
    for offset, (label, value_column, std_column, colour) in enumerate(
        zip(labels, value_columns, std_columns, colours)
    ):
        ax.bar(
            positions + (offset - 1) * width,
            plot_df[value_column].to_numpy(dtype=float),
            width=width,
            yerr=plot_df[std_column].to_numpy(dtype=float),
            label=label,
            color=colour,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.7,
            error_kw={"ecolor": COLORS["reference"], "elinewidth": 0.9, "capsize": 2},
        )

    label_map = {
        "Decision Tree": "Decision\nTree",
        "Random Forest": "Random\nForest",
        "Gradient Boosting": "Gradient\nBoosting",
    }
    ax.set_xticks(positions, [label_map.get(model, model) for model in models])
    ax.set_ylabel("Metric value (%)")
    ax.set_title("Supplementary dimensionless error metrics", pad=8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    style_axes(ax, grid_axis="y")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return fig, ax


def plot_cv_test_comparison(
    benchmark_results,
    fold_results,
    output_path,
    model_order=None,
    figure_size=(10.5, 4.6),
    cv_color=COLORS["primary_light"],
    fold_point_color=COLORS["primary"],
    mean_color=COLORS["tertiary"],
    median_color=COLORS["reference"],
    test_color=COLORS["secondary"],
    separator_color=COLORS["grid"],
    dpi=300,
):
    """Compare grouped-CV fold distributions with the common hold-out."""

    from matplotlib.lines import Line2D

    required_columns = [
        "Model", "CV RMSE", "Test RMSE", "CV R2", "Test R2",
    ]
    missing_columns = [
        column for column in required_columns
        if column not in benchmark_results.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {benchmark_results.columns.tolist()}"
        )

    required_fold_columns = ["Model", "RMSE", "R2"]
    missing_fold_columns = [
        column for column in required_fold_columns
        if column not in fold_results.columns
    ]
    if missing_fold_columns:
        raise ValueError(
            f"Missing fold-result columns: {missing_fold_columns}. "
            f"Available columns: {fold_results.columns.tolist()}"
        )

    plot_df = benchmark_results[required_columns].copy()
    metric_columns = [column for column in required_columns if column != "Model"]
    for column in metric_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    if plot_df[metric_columns].isna().any().any():
        raise ValueError("CV or hold-out metrics contain missing or non-numeric values.")

    fold_df = fold_results[required_fold_columns].copy()
    for column in ["RMSE", "R2"]:
        fold_df[column] = pd.to_numeric(fold_df[column], errors="coerce")
    if fold_df[["RMSE", "R2"]].isna().any().any():
        raise ValueError("Fold metrics contain missing or non-numeric values.")

    if model_order is not None:
        available_models = plot_df["Model"].tolist()
        final_order = [model for model in model_order if model in available_models]
        final_order += [
            model for model in available_models if model not in final_order
        ]
        plot_df["Model"] = pd.Categorical(
            plot_df["Model"], categories=final_order, ordered=True
        )
        plot_df = plot_df.sort_values("Model").reset_index(drop=True)

    models = plot_df["Model"].astype(str).tolist()
    missing_fold_models = sorted(set(models) - set(fold_df["Model"]))
    if missing_fold_models:
        raise ValueError(f"Missing fold results for models: {missing_fold_models}")

    label_map = {
        "Decision Tree": "Decision\nTree",
        "Random Forest": "Random\nForest",
        "Gradient Boosting": "Gradient\nBoosting",
    }
    model_labels = [label_map.get(model, model) for model in models]
    positions = np.arange(len(plot_df))
    box_positions = positions - 0.09
    holdout_positions = positions + 0.18

    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    panel_specs = [
        (axes[0], "RMSE", "Test RMSE", "RMSE",
         "RMSE (µg m$^{-3}$)", "(a)", "Lower is better"),
        (axes[1], "R2", "Test R2", "R²", "R²", "(b)", "Higher is better"),
    ]

    for ax, fold_column, test_column, title, ylabel, panel_label, direction in panel_specs:
        distributions = [
            fold_df.loc[fold_df["Model"].eq(model), fold_column].to_numpy(dtype=float)
            for model in models
        ]
        test_values = plot_df[test_column].to_numpy()

        boxes = ax.boxplot(
            distributions,
            positions=box_positions,
            widths=0.34,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            medianprops={"color": median_color, "linewidth": 1.2},
            meanprops={
                "marker": "o", "markerfacecolor": mean_color,
                "markeredgecolor": "white", "markersize": 4.8,
            },
            whiskerprops={"color": fold_point_color, "linewidth": 1.0},
            capprops={"color": fold_point_color, "linewidth": 1.0},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(cv_color)
            box.set_edgecolor(fold_point_color)
            box.set_alpha(0.48)

        for x_position, values in zip(box_positions, distributions):
            offsets = np.linspace(-0.045, 0.045, len(values))
            ax.scatter(
                x_position + offsets,
                values,
                s=24,
                color=fold_point_color,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

        ax.scatter(
            holdout_positions,
            test_values,
            marker="D",
            s=58,
            color=test_color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

        ax.set_xticks(positions, model_labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_title(title, pad=8)
        for separator in positions[:-1] + 0.5:
            ax.axvline(
                separator, color=separator_color,
                linestyle="--", linewidth=0.7, alpha=0.75, zorder=0,
            )
        style_axes(ax, grid_axis="y")
        ax.text(
            -0.12, 1.02, panel_label,
            transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold",
        )
        ax.text(
            0.99, 0.98, direction,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=COLORS["reference"],
        )

    legend_handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=6,
            markerfacecolor=fold_point_color, markeredgecolor="white",
            label="Individual CV fold",
        ),
        Line2D(
            [], [], marker="o", linestyle="none", markersize=6,
            markerfacecolor=mean_color, markeredgecolor="white",
            label="CV mean",
        ),
        Line2D(
            [], [], color=median_color, linewidth=1.2,
            label="CV median",
        ),
        Line2D(
            [], [], marker="D", linestyle="none", markersize=7,
            markerfacecolor=test_color, markeredgecolor="white",
            label="Common hold-out",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=dpi, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight", facecolor="white",
    )
    return fig, axes


def plot_best_model_holdout_diagnostics(
    predictions,
    benchmark_results,
    model_name,
    output_path,
    development_maximum,
    concentration_thresholds=(20.0, 35.0),
    figure_size=(12.0, 5.2),
    point_color=COLORS["primary"],
    point_edge_color=COLORS["primary_dark"],
    residual_color=COLORS["secondary_light"],
    trend_color=COLORS["secondary"],
    residual_trend_color=COLORS["tertiary"],
    threshold_colors=(COLORS["moderate"], COLORS["high"]),
    development_color=COLORS["tertiary"],
    dpi=300,
):
    """Plot hold-out agreement with marginal and residual diagnostics."""

    if model_name not in predictions:
        raise KeyError(f"Model '{model_name}' not found in predictions.")
    required_results = ["Model", "Test R2", "Test RMSE", "Test MAE"]
    missing = [
        column for column in required_results
        if column not in benchmark_results.columns
    ]
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")

    prediction_df = predictions[model_name].copy()
    missing = [
        column for column in ["Observed", "Predicted"]
        if column not in prediction_df
    ]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    observed = pd.to_numeric(prediction_df["Observed"], errors="coerce")
    predicted = pd.to_numeric(prediction_df["Predicted"], errors="coerce")
    valid = observed.notna() & predicted.notna()
    observed = observed[valid].to_numpy()
    predicted = predicted[valid].to_numpy()
    if len(observed) < 2:
        raise ValueError("At least two valid hold-out predictions are required.")

    development_maximum = float(development_maximum)
    thresholds = tuple(float(value) for value in concentration_thresholds)
    if len(thresholds) != 2 or thresholds[0] >= thresholds[1]:
        raise ValueError("Two increasing concentration thresholds are required.")
    if len(thresholds) != len(threshold_colors):
        raise ValueError("Each concentration threshold requires one colour.")

    residual = predicted - observed
    mean_bias = float(np.mean(residual))
    metric_row = benchmark_results.set_index("Model").loc[model_name]
    axis_max = float(np.ceil(max(observed.max(), predicted.max()) / 5) * 5)
    histogram_bins = np.linspace(0, axis_max, 19)

    fig = plt.figure(figsize=figure_size)
    outer = fig.add_gridspec(
        1, 2, width_ratios=(1.06, 1.0), wspace=0.28,
    )
    left = outer[0].subgridspec(
        2, 2,
        height_ratios=(0.82, 4.0),
        width_ratios=(4.0, 0.82),
        hspace=0.05,
        wspace=0.05,
    )
    ax_hist_observed = fig.add_subplot(left[0, 0])
    ax_prediction = fig.add_subplot(left[1, 0])
    ax_hist_predicted = fig.add_subplot(left[1, 1], sharey=ax_prediction)
    ax_residual = fig.add_subplot(outer[1])

    # Marginal distributions aligned with the agreement plot.
    ax_hist_observed.hist(
        observed,
        bins=histogram_bins,
        color=COLORS["primary_light"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.85,
    )
    ax_hist_observed.set_xlim(0, axis_max)
    ax_hist_observed.set_ylabel("Count", fontsize=8)
    ax_hist_observed.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_hist_observed.tick_params(axis="y", labelsize=7)

    ax_hist_predicted.hist(
        predicted,
        bins=histogram_bins,
        orientation="horizontal",
        color=COLORS["secondary_light"],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
    )
    ax_hist_predicted.set_xlabel("Count", fontsize=8)
    ax_hist_predicted.tick_params(axis="x", labelsize=7)
    ax_hist_predicted.tick_params(axis="y", labelleft=False, left=False)

    # Observed versus predicted agreement.
    ax_prediction.scatter(
        observed,
        predicted,
        s=19,
        color=point_color,
        alpha=0.34,
        edgecolor=point_edge_color,
        linewidth=0.32,
        rasterized=True,
        zorder=2,
    )
    line_x = np.array([0.0, axis_max])
    ax_prediction.plot(
        line_x, line_x,
        linestyle="--", color=COLORS["reference"],
        linewidth=1.2, label="1:1 line",
    )
    slope, intercept, _, _, _ = linregress(observed, predicted)
    ax_prediction.plot(
        line_x,
        intercept + slope * line_x,
        color=trend_color,
        linewidth=1.7,
        label="Calibration line",
    )
    if development_maximum < axis_max:
        ax_prediction.axvspan(
            development_maximum, axis_max,
            color=development_color, alpha=0.07, zorder=0,
        )
        ax_prediction.text(
            (development_maximum + axis_max) / 2,
            0.985,
            "Outside development range",
            transform=ax_prediction.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.8,
            color=development_color,
        )
    ax_prediction.set_xlim(0, axis_max)
    ax_prediction.set_ylim(0, axis_max)
    ax_prediction.set_aspect("equal", adjustable="box")
    ax_prediction.set_xlabel("Observed PM$_{2.5}$ (µg m$^{-3}$)")
    ax_prediction.set_ylabel("Predicted PM$_{2.5}$ (µg m$^{-3}$)")
    ax_prediction.text(
        0.04, 0.96,
        f"R² = {float(metric_row['Test R2']):.3f}\n"
        f"RMSE = {float(metric_row['Test RMSE']):.2f} µg m$^{{-3}}$\n"
        f"MAE = {float(metric_row['Test MAE']):.2f} µg m$^{{-3}}$\n"
        f"Bias = {mean_bias:+.2f} µg m$^{{-3}}$",
        transform=ax_prediction.transAxes,
        ha="left", va="top", fontsize=8.7,
        bbox={
            "facecolor": "white", "edgecolor": COLORS["grid"],
            "alpha": 0.92,
        },
    )
    ax_prediction.legend(
        loc="lower right", frameon=False, fontsize=8.3,
    )
    style_axes(ax_prediction, grid_axis="both")

    # Residual distributions within the three reporting ranges.
    residual_frame = pd.DataFrame({"Observed": observed, "Residual": residual})
    bin_labels = [
        f"≤{thresholds[0]:g}",
        f"{thresholds[0]:g}–{thresholds[1]:g}",
        f">{thresholds[1]:g}",
    ]
    residual_frame["Bin"] = pd.cut(
        residual_frame["Observed"],
        bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
        labels=bin_labels,
        include_lowest=True,
    )
    range_values = [
        residual_frame.loc[residual_frame["Bin"] == label, "Residual"].to_numpy()
        for label in bin_labels
    ]
    range_colors = [COLORS["primary"], *threshold_colors]
    boxplot = ax_residual.boxplot(
        range_values,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["reference"], "linewidth": 1.4},
        whiskerprops={"color": COLORS["reference"], "linewidth": 1.0},
        capprops={"color": COLORS["reference"], "linewidth": 1.0},
    )
    for box, color in zip(boxplot["boxes"], range_colors):
        box.set(facecolor=color, edgecolor=color, alpha=0.20, linewidth=1.2)

    random_generator = np.random.default_rng(42)
    for position, (values, color) in enumerate(
        zip(range_values, range_colors), start=1,
    ):
        jitter = random_generator.normal(position, 0.055, size=len(values))
        ax_residual.scatter(
            jitter,
            values,
            s=13,
            color=color,
            alpha=0.16,
            edgecolor="none",
            rasterized=True,
            zorder=2,
        )
        range_bias = float(np.mean(values))
        ax_residual.scatter(
            position,
            range_bias,
            marker="D",
            s=45,
            color=residual_trend_color,
            zorder=4,
            label="Mean bias" if position == 1 else None,
        )

    ax_residual.axhline(0, color=COLORS["reference"], linewidth=1.0)
    residual_limit = float(np.ceil(np.max(np.abs(residual)) / 5) * 5)
    ax_residual.set_ylim(-residual_limit, residual_limit)
    ax_residual.set_xticks(
        range(1, len(bin_labels) + 1),
        [
            f"{label}\n($n$={len(values):,})"
            for label, values in zip(bin_labels, range_values)
        ],
    )
    ax_residual.set_xlabel("Observed PM$_{2.5}$ range (µg m$^{-3}$)")
    ax_residual.set_ylabel("Residual (predicted − observed, µg m$^{-3}$)")
    ax_residual.legend(
        loc="lower left", frameon=False, fontsize=8.2,
    )
    style_axes(ax_residual, grid_axis="y")

    for marginal_axis in (ax_hist_observed, ax_hist_predicted):
        marginal_axis.grid(False)
        marginal_axis.spines["top"].set_visible(False)
        marginal_axis.spines["right"].set_visible(False)

    fig.text(0.015, 0.965, "(a)", ha="left", va="top", fontweight="bold")
    fig.text(0.535, 0.965, "(b)", ha="left", va="top", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.93)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight", facecolor="white",
    )
    return fig, (ax_prediction, ax_residual)





def plot_clean_reference_fold_distributions(
    comparison_results,
    fold_results,
    output_path,
    figure_size=(7.0, 3.6),
    raw_color=COLORS["primary"],
    normalized_color=COLORS["secondary"],
    holdout_color=COLORS["tertiary"],
    dpi=300,
):
    """Summarize directional performance changes after normalization."""

    representations = ["Raw", "Clean-reference normalized"]
    required_summary = {"Representation", "Test RMSE", "Test R2"}
    required_folds = {"Fold", "Representation", "RMSE", "R2"}
    if missing := required_summary - set(comparison_results.columns):
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    if missing := required_folds - set(fold_results.columns):
        raise ValueError(f"Missing fold columns: {sorted(missing)}")

    summary_indexed = comparison_results.set_index("Representation")
    fold_means = fold_results.groupby("Representation")[["RMSE", "R2"]].mean()

    raw_cv_rmse = float(fold_means.loc["Raw", "RMSE"])
    norm_cv_rmse = float(fold_means.loc["Clean-reference normalized", "RMSE"])
    raw_cv_r2 = float(fold_means.loc["Raw", "R2"])
    norm_cv_r2 = float(fold_means.loc["Clean-reference normalized", "R2"])
    raw_test_rmse = float(summary_indexed.loc["Raw", "Test RMSE"])
    norm_test_rmse = float(summary_indexed.loc[
        "Clean-reference normalized", "Test RMSE"
    ])
    raw_test_r2 = float(summary_indexed.loc["Raw", "Test R2"])
    norm_test_r2 = float(summary_indexed.loc[
        "Clean-reference normalized", "Test R2"
    ])

    # Positive values always mean improvement, irrespective of metric direction.
    changes = np.asarray([
        (raw_cv_rmse - norm_cv_rmse) / raw_cv_rmse * 100,
        (norm_cv_r2 - raw_cv_r2) / abs(raw_cv_r2) * 100,
        (raw_test_rmse - norm_test_rmse) / raw_test_rmse * 100,
        (norm_test_r2 - raw_test_r2) / abs(raw_test_r2) * 100,
    ])
    raw_values = [raw_cv_rmse, raw_cv_r2, raw_test_rmse, raw_test_r2]
    normalized_values = [norm_cv_rmse, norm_cv_r2, norm_test_rmse, norm_test_r2]
    row_labels = [
        "Development CV · RMSE",
        "Development CV · R²",
        "Common hold-out · RMSE",
        "Common hold-out · R²",
    ]
    y_positions = np.arange(4)[::-1]
    point_colors = [raw_color, raw_color, holdout_color, holdout_color]

    fig, axis = plt.subplots(figsize=figure_size)
    x_min = min(-1.0, float(np.min(changes) - 0.55))
    x_max = max(1.0, float(np.max(changes) + 0.55))
    axis.axvline(0, color=COLORS["reference"], linewidth=1.1, zorder=1)
    axis.barh(
        y_positions, changes, height=0.48,
        color=point_colors, alpha=0.88, edgecolor="white",
        linewidth=0.7, zorder=2,
    )

    for y_value, change, color, raw, normalized in zip(
        y_positions, changes, point_colors, raw_values, normalized_values,
    ):
        offset = 6 if change >= 0 else -6
        axis.annotate(
            f"{change:+.1f}%  ({raw:.3g} → {normalized:.3g})",
            (change, y_value), xytext=(offset, 0),
            textcoords="offset points",
            ha="left" if change >= 0 else "right", va="center",
            fontsize=8.5, color=color,
        )

    axis.set_yticks(y_positions, row_labels)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(-0.65, 3.65)
    axis.set_xlabel("Relative performance change after normalization (%)")
    axis.text(
        0.01, 1.02, "← Deterioration", transform=axis.transAxes,
        ha="left", va="bottom", color=COLORS["moderate"], fontsize=9,
    )
    axis.text(
        0.99, 1.02, "Improvement →", transform=axis.transAxes,
        ha="right", va="bottom", color=COLORS["secondary"], fontsize=9,
    )
    style_axes(axis, grid_axis="x")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight", facecolor="white",
    )
    return fig, axis
