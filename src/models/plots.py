"""

Benchmark visualization utilities.

This module provides publication-quality figures for machine learning model benchmarking.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# =============================================================================

# Figure 1: Overall Cross-validation Benchmark

# =============================================================================

def plot_cross_validation_benchmark(
    benchmark_results,
    output_path,
    model_order=None,
    figure_size=(9, 5.5),
    point_color="#4C78A8",
    error_color="#A9C5DF",
    separator_color="#E0E0E0",
    best_color="#4C6A92",
    marker_size=9,
    cap_size=5,
    dpi=300,
):
    """
    Plot mean cross-validation R² with ±1 standard deviation error bars.

    Parameters
    ----------
    benchmark_results : pandas.DataFrame
        Benchmark summary table containing the columns:
        - Model
        - CV R2
        - CV R2 Std

    output_path : str or pathlib.Path
        Path used to save the figure.

    model_order : list, optional
        Desired model display order.

    figure_size : tuple, default=(9, 5.5)
        Figure size in inches.

    point_color : str
        Color used for the mean R² markers.

    error_color : str
        Color used for the standard deviation error bars.

    separator_color : str
        Color used for the vertical grid lines.

    best_color : str
        Color used to highlight the model with the highest mean R².

    marker_size : float
        Marker size.

    cap_size : float
        Error-bar cap size.

    dpi : int
        Figure output resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Figure axis.
    """

    required_columns = [
        "Model",
        "CV R2",
        "CV R2 Std",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in benchmark_results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: "
            f"{benchmark_results.columns.tolist()}"
        )

    plot_df = benchmark_results[required_columns].copy()

    # Convert metric columns to numeric values
    numeric_columns = [
        "CV R2",
        "CV R2 Std",
    ]

    for column in numeric_columns:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    if plot_df[numeric_columns].isna().any().any():
        raise ValueError(
            "The CV R² columns contain missing or "
            "non-numeric values."
        )

    # Apply requested model order
    if model_order is not None:
        available_models = plot_df["Model"].tolist()

        valid_order = [
            model
            for model in model_order
            if model in available_models
        ]

        remaining_models = [
            model
            for model in available_models
            if model not in valid_order
        ]

        final_order = valid_order + remaining_models

        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=final_order,
            ordered=True,
        )

        plot_df = (
            plot_df
            .sort_values("Model")
            .reset_index(drop=True)
        )

    models = plot_df["Model"].astype(str).tolist()
    x_positions = np.arange(len(models))

    cv_r2_mean = plot_df["CV R2"].to_numpy()
    cv_r2_std = plot_df["CV R2 Std"].to_numpy()

    # Create figure
    fig, ax = plt.subplots(
        figsize=figure_size,
    )

    best_index = int(np.argmax(cv_r2_mean))

    # Plot each model separately so the strongest model can be highlighted.
    for index, (value, std) in enumerate(zip(cv_r2_mean, cv_r2_std)):
        is_best = index == best_index
        marker_color = best_color if is_best else point_color

        ax.errorbar(
            value,
            y_positions[index],
            xerr=std,
            fmt="o",
            markersize=marker_size + (1 if is_best else 0),
            markerfacecolor=marker_color if is_best else "white",
            markeredgecolor=marker_color,
            markeredgewidth=2.0,
            ecolor=best_color if is_best else error_color,
            elinewidth=1.8 if is_best else 1.5,
            capsize=cap_size,
            capthick=1.6,
            linestyle="none",
            zorder=4 if is_best else 3,
        )

    # Use a data-driven axis range so no error bar is clipped.
    lower_limit = np.floor(
        (cv_r2_mean - cv_r2_std).min() * 20
    ) / 20 - 0.01
    upper_limit = np.ceil(
        (cv_r2_mean + cv_r2_std).max() * 20
    ) / 20 + 0.03
    ax.set_xlim(max(-1.0, lower_limit), min(1.0, upper_limit))

    label_offset = 0.008
    for index, value in enumerate(cv_r2_mean):
        ax.text(
            value + cv_r2_std[index] + label_offset,
            y_positions[index],
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            color=best_color if index == best_index else "black",
            fontweight="bold" if index == best_index else "normal",
        )

    # Axis labels
    ax.set_xlabel(
        "Mean cross-validation R²",
        fontsize=12,
        color="black",
    )
    ax.set_ylabel("")

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        models,
        ha="right",
        fontsize=10,
        color="black",
        fontweight="medium",
    )

    # Preserve the requested model order from top to bottom.
    ax.invert_yaxis()

    ax.tick_params(
        axis="x",
        labelsize=10,
        labelcolor="black",
    )

    ax.tick_params(
        axis="y",
        colors="black",
        length=0,
    )

    # Vertical reference grid only
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.8,
        color=separator_color,
        alpha=0.55,
    )

    ax.grid(
        axis="y",
        visible=False,
    )

    ax.set_axisbelow(True)

    # Clean figure borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.spines["bottom"].set_linewidth(0.8)

    
    # Annotation explaining the error bars
    ax.text(
        0.99,
        1.02,
        "Points: mean R²   |   Error bars: ±1 SD (5-fold CV)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4A4A4A",
    )

    fig.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print(f"Figure saved to: {output_path}")

    return fig, ax

# =============================================================================

# Figure 2：Held-out test performance

# =============================================================================

def plot_heldout_test_performance(
    benchmark_results,
    output_path,
    model_order=None,
    figure_size=(9, 5.5),
    r2_color="#A8D5BA",
    best_color="#355C9A",
    rmse_color="#D55E00",
    separator_color="#E0E0E0",
    dpi=300,
):
    """
    Plot held-out test R² as bars and test RMSE as point markers.

    Parameters
    ----------
    benchmark_results : pandas.DataFrame
        Benchmark summary containing:
        Model, Test R2, and Test RMSE.

    output_path : str or pathlib.Path
        Path used to save the figure.

    model_order : list, optional
        Desired model display order.

    figure_size : tuple
        Figure size in inches.

    dpi : int
        Output resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax_r2 : matplotlib.axes.Axes
        R² axis.

    ax_rmse : matplotlib.axes.Axes
        RMSE axis.
    """

    required_columns = [
        "Model",
        "Test R2",
        "Test RMSE",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in benchmark_results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: "
            f"{benchmark_results.columns.tolist()}"
        )

    plot_df = benchmark_results[required_columns].copy()

    for column in ["Test R2", "Test RMSE"]:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    if plot_df[["Test R2", "Test RMSE"]].isna().any().any():
        raise ValueError(
            "Test metric columns contain missing or "
            "non-numeric values."
        )

    # Apply model order
    if model_order is not None:
        available_models = plot_df["Model"].tolist()

        valid_order = [
            model
            for model in model_order
            if model in available_models
        ]

        remaining_models = [
            model
            for model in available_models
            if model not in valid_order
        ]

        final_order = valid_order + remaining_models

        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=final_order,
            ordered=True,
        )

        plot_df = (
            plot_df
            .sort_values("Model")
            .reset_index(drop=True)
        )

    models = plot_df["Model"].astype(str).tolist()
    x_positions = np.arange(len(models))

    test_r2 = plot_df["Test R2"].to_numpy()
    test_rmse = plot_df["Test RMSE"].to_numpy()

    best_index = int(np.argmax(test_r2))

    bar_colors = [
        best_color if index == best_index else r2_color
        for index in range(len(plot_df))
    ]

    fig, ax_r2 = plt.subplots(figsize=figure_size)

    bars = ax_r2.bar(
        x_positions,
        test_r2,
        width=0.50,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax_rmse = ax_r2.twinx()
    ax_rmse.grid(False)

    ax_rmse.scatter(
        x_positions,
        test_rmse,
        s=58,
        marker="D",
        color=rmse_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Test RMSE",
    )

    # R² labels
    for bar, value in zip(bars, test_r2):
        ax_r2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="black",
        )

    # RMSE labels
    for index, value in enumerate(test_rmse):
        ax_rmse.annotate(
            f"{value:.2f}",
            xy=(x_positions[index], value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    ax_r2.set_ylabel(
        "Common hold-out R²",
        fontsize=11,
        color="black",
    )

    ax_rmse.set_ylabel(
        "Common hold-out RMSE (µg/m³)",
        fontsize=11,
        color="black",
    )

    ax_r2.set_xticks(x_positions)

    ax_r2.set_xticklabels(
        models,
        rotation=0,
        ha="center",
        fontsize=10,
        color="black",
    )

    ax_r2.tick_params(
        axis="y",
        labelcolor="black",
        color="#B0B0B0",
        width=0.8,
        length=0,
    )

    ax_rmse.tick_params(
        axis="y",
        labelcolor="black",
        color="#B0B0B0",
        width=0.8,
        length=0,
    )

    ax_r2.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        color=separator_color,
        alpha=0.55,
    )

    ax_r2.grid(
        axis="x",
        visible=False,
    )

    ax_r2.set_axisbelow(True)

    # R² bars use a zero baseline; RMSE points use a data-driven right axis.
    r2_lower = min(0.0, float(test_r2.min()) - 0.05)
    r2_upper = min(1.0, float(test_r2.max()) + 0.12)
    ax_r2.set_ylim(r2_lower, r2_upper)

    rmse_lower = max(
        0.0,
        np.floor((float(test_rmse.min()) - 0.5) * 2) / 2,
    )
    rmse_upper = np.ceil(
        (float(test_rmse.max()) + 0.6) * 2
    ) / 2
    ax_rmse.set_ylim(rmse_lower, rmse_upper)

    ax_r2.spines["top"].set_visible(False)
    ax_rmse.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)

    ax_r2.spines["left"].set_color("#B0B0B0")
    ax_r2.spines["bottom"].set_color("#B0B0B0")
    ax_rmse.spines["right"].set_color("#B0B0B0")
    ax_rmse.spines["left"].set_visible(False)
    ax_rmse.spines["bottom"].set_visible(False)

    ax_r2.spines["left"].set_linewidth(0.8)
    ax_r2.spines["bottom"].set_linewidth(0.8)
    ax_rmse.spines["right"].set_linewidth(0.8)


    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(
            facecolor=r2_color,
            edgecolor="none",
            label="Common hold-out R²",
        ),
        Patch(
            facecolor=best_color,
            edgecolor="none",
            label="Best common hold-out R²",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=rmse_color,
            markeredgecolor="white",
            markersize=7,
            label="Common hold-out RMSE",
        ),
    ]

    ax_r2.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=False,
        ncol=3,
        fontsize=10,
        handlelength=1.4,
        columnspacing=1.2,
        borderaxespad=0.6,
    )

    fig.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print(f"Figure saved to: {output_path}")

    return fig, ax_r2, ax_rmse

# =============================================================================

# Figure 3：Compare mean cross-validation R² and held-out test R².

# =============================================================================

def plot_cv_test_comparison(
    benchmark_results,
    output_path,
    model_order=None,
    figure_size=(9.5, 5.8),
    cv_color="#A8C5E5",
    test_color="#355C9A",
    separator_color="#E0E0E0",
    bar_width=0.30,
    dpi=300,
):
    """
    Compare mean cross-validation R² and held-out test R².

    Parameters
    ----------
    benchmark_results : pandas.DataFrame
        Benchmark results containing:
        - Model
        - CV R2
        - Test R2

    output_path : str or pathlib.Path
        Figure output path.

    model_order : list, optional
        Desired display order of models.

    figure_size : tuple
        Figure size in inches.

    cv_color : str
        Bar color for mean cross-validation R².

    test_color : str
        Bar color for held-out test R².

    separator_color : str
        Horizontal grid-line color.

    bar_width : float
        Width of each bar.

    dpi : int
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Figure axis.
    """

    required_columns = [
        "Model",
        "CV R2",
        "CV R2 Std",
        "Test R2",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in benchmark_results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: "
            f"{benchmark_results.columns.tolist()}"
        )

    plot_df = benchmark_results[required_columns].copy()

    # Convert metrics to numeric
    for column in ["CV R2", "CV R2 Std", "Test R2"]:
        plot_df[column] = pd.to_numeric(
            plot_df[column],
            errors="coerce",
        )

    if plot_df[["CV R2", "CV R2 Std", "Test R2"]].isna().any().any():
        raise ValueError(
            "CV R² or Test R² contains missing "
            "or non-numeric values."
        )

    # Apply requested model order
    if model_order is not None:
        available_models = plot_df["Model"].tolist()

        valid_order = [
            model
            for model in model_order
            if model in available_models
        ]

        remaining_models = [
            model
            for model in available_models
            if model not in valid_order
        ]

        final_order = valid_order + remaining_models

        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=final_order,
            ordered=True,
        )

        plot_df = (
            plot_df
            .sort_values("Model")
            .reset_index(drop=True)
        )

    models = plot_df["Model"].astype(str).tolist()

    cv_r2 = plot_df["CV R2"].to_numpy()
    cv_r2_std = plot_df["CV R2 Std"].to_numpy()
    test_r2 = plot_df["Test R2"].to_numpy()

    r2_gap = cv_r2 - test_r2

    x_positions = np.arange(len(models))

    fig, ax = plt.subplots(
        figsize=figure_size,
    )

    best_index = int(np.argmax(test_r2))

    cv_bars = ax.bar(
        x_positions - bar_width / 2,
        cv_r2,
        width=bar_width,
        color=cv_color,
        edgecolor="white",
        linewidth=0.8,
        label="Date-grouped CV R²",
        zorder=3,
    )
    test_bars = ax.bar(
        x_positions + bar_width / 2,
        test_r2,
        width=bar_width,
        color=test_color,
        edgecolor="white",
        linewidth=0.8,
        label="Common hold-out R²",
        zorder=3,
    )

    ax.errorbar(
        x_positions - bar_width / 2,
        cv_r2,
        yerr=cv_r2_std,
        fmt="none",
        ecolor="#806A91",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
        zorder=5,
    )

    for index, (bar, value, std) in enumerate(zip(cv_bars, cv_r2, cv_r2_std)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + std + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    for bar, value in zip(test_bars, test_r2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    for index, gap in enumerate(r2_gap):
        label_height = max(cv_r2[index] + cv_r2_std[index], test_r2[index]) + 0.055
        ax.text(
            x_positions[index],
            label_height,
            f"ΔR² {gap:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8.7,
            color="#4A4A4A",
        )

    ax.set_ylabel("R²", fontsize=11, color="black")
    ax.set_xlabel("")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        models,
        rotation=0,
        ha="center",
        fontsize=10,
        color="black",
    )
    ax.get_xticklabels()[best_index].set_fontweight("bold")
    ax.tick_params(axis="y", labelsize=10, labelcolor="black", length=0)

    upper_limit = min(
        1.0,
        max(np.max(cv_r2 + cv_r2_std), np.max(test_r2)) + 0.12,
    )
    ax.set_ylim(0, upper_limit)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        color=separator_color,
        alpha=0.55,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.6,
    )

    fig.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print(f"Figure saved to: {output_path}")

    return fig, ax

# =============================================================================

# Best-model hold-out diagnostics

# =============================================================================

def plot_best_model_holdout_diagnostics(
    predictions,
    benchmark_results,
    model_name,
    output_path,
    figure_size=(10.5, 4.8),
    density_low="#E9DFF0",
    density_high="#5F789B",
    residual_color="#478E7F",
    trend_color="#4C6A92",
    dpi=300,
):
    """Plot common hold-out predictions and residual behaviour for one model."""
    from matplotlib.colors import LinearSegmentedColormap

    if model_name not in predictions:
        raise KeyError(f"Model '{model_name}' not found in predictions.")
    required_results = ["Model", "Test R2", "Test RMSE", "Test MAE"]
    missing = [column for column in required_results if column not in benchmark_results]
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

    residual = predicted - observed
    mean_bias = float(np.mean(residual))
    metric_row = benchmark_results.set_index("Model").loc[model_name]
    axis_max = float(np.ceil(max(observed.max(), predicted.max()) / 5) * 5)
    density_cmap = LinearSegmentedColormap.from_list(
        "holdout_density", [density_low, density_high]
    )

    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    ax_prediction, ax_residual = axes
    density = ax_prediction.hexbin(
        observed, predicted, gridsize=28, mincnt=1, cmap=density_cmap,
        linewidths=0.15, extent=(0, axis_max, 0, axis_max),
    )
    ax_prediction.plot(
        [0, axis_max], [0, axis_max], linestyle="--", color="#666666",
        linewidth=1.4, label="1:1 line",
    )
    slope, intercept, _, _, _ = linregress(observed, predicted)
    line_x = np.array([0.0, axis_max])
    ax_prediction.plot(
        line_x, intercept + slope * line_x, color=trend_color,
        linewidth=1.8, label="Calibration line",
    )
    ax_prediction.set_xlim(0, axis_max)
    ax_prediction.set_ylim(0, axis_max)
    ax_prediction.set_aspect("equal", adjustable="box")
    ax_prediction.set_xlabel("Observed PM2.5 (µg/m³)")
    ax_prediction.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax_prediction.set_title("(a) Hold-out predictions", loc="left", fontweight="bold")
    ax_prediction.legend(loc="lower right", frameon=False, fontsize=9)
    ax_prediction.text(
        0.04, 0.96,
        f"R² = {float(metric_row['Test R2']):.3f}\n"
        f"RMSE = {float(metric_row['Test RMSE']):.2f} µg/m³\n"
        f"MAE = {float(metric_row['Test MAE']):.2f} µg/m³\n"
        f"Bias = {mean_bias:+.2f} µg/m³",
        transform=ax_prediction.transAxes, ha="left", va="top", fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#D8D8D8", "alpha": 0.92},
    )
    colorbar = fig.colorbar(density, ax=ax_prediction, fraction=0.046, pad=0.03)
    colorbar.set_label("Observation count", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    ax_residual.scatter(
        observed, residual, s=18, color=residual_color, alpha=0.42,
        edgecolor="none", rasterized=True, label="Hold-out observations",
    )
    residual_frame = pd.DataFrame({"Observed": observed, "Residual": residual})
    residual_frame["Bin"] = pd.qcut(residual_frame["Observed"], q=10, duplicates="drop")
    binned = residual_frame.groupby("Bin", observed=False).agg(
        Observed=("Observed", "mean"), Residual=("Residual", "mean")
    )
    ax_residual.plot(
        binned["Observed"], binned["Residual"], color=trend_color,
        marker="o", markersize=4.5, linewidth=1.8,
        label="Binned mean error", zorder=4,
    )
    ax_residual.axhline(0, color="#666666", linestyle="--", linewidth=1.3)
    ax_residual.set_xlim(0, axis_max)
    residual_limit = float(np.ceil(np.max(np.abs(residual)) / 5) * 5)
    ax_residual.set_ylim(-residual_limit, residual_limit)
    ax_residual.set_xlabel("Observed PM2.5 (µg/m³)")
    ax_residual.set_ylabel("Residual (predicted − observed, µg/m³)")
    ax_residual.set_title("(b) Residual pattern", loc="left", fontweight="bold")
    ax_residual.legend(loc="lower left", frameon=False, fontsize=9)

    for axis in axes:
        axis.grid(True, linestyle="--", linewidth=0.7, color="#E0E0E0", alpha=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#B0B0B0")
        axis.spines["bottom"].set_color("#B0B0B0")

    fig.suptitle(model_name, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Figure saved to: {output_path}")
    return fig, axes


# =============================================================================

# Validation-strategy heatmap

# =============================================================================

def plot_validation_strategy_heatmap(
    validation_results,
    output_path,
    model_order=None,
    figure_size=(9.2, 3.8),
    low_color="#F5EFF7",
    high_color="#4F6F9F",
    dpi=300,
):
    """Plot annotated R² values across four validation protocols."""
    from matplotlib.colors import LinearSegmentedColormap

    protocol_columns = [
        "Date-Grouped CV R2", "Row-Wise CV R2",
        "Common Hold-out R2", "Row-Wise Hold-out R2",
    ]
    required_columns = ["Model", *protocol_columns]
    missing = [column for column in required_columns if column not in validation_results]
    if missing:
        raise ValueError(f"Missing validation columns: {missing}")
    plot_df = validation_results[required_columns].copy()
    for column in protocol_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    if plot_df[protocol_columns].isna().any().any():
        raise ValueError("Validation R² columns contain missing or non-numeric values.")

    if model_order is not None:
        available = plot_df["Model"].tolist()
        order = [model for model in model_order if model in available]
        order += [model for model in available if model not in order]
        plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("Model").reset_index(drop=True)

    matrix = plot_df[protocol_columns].to_numpy(dtype=float)
    vmin = np.floor(matrix.min() * 10) / 10
    vmax = np.ceil(matrix.max() * 10) / 10
    cmap = LinearSegmentedColormap.from_list("validation_r2", [low_color, high_color])
    fig, ax = plt.subplots(figsize=figure_size)
    ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    labels = [
        "Date-grouped", "Row-wise",
        "Date-disjoint", "Row-wise",
    ]
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(plot_df)), labels=plot_df["Model"].astype(str))
    ax.tick_params(axis="x", length=0, labelsize=10, pad=8)
    ax.tick_params(axis="y", length=0, labelsize=10, pad=8)

    # Hierarchical headers distinguish the two evaluation stages.
    ax.text(
        0.5, 1.09, "Cross-validation", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
    ax.text(
        2.5, 1.09, "Hold-out evaluation", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

    column_maxima = matrix.max(axis=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column, row, f"{value:.3f}", ha="center", va="center",
                color="white" if value >= 0.66 else "#222222", fontsize=10.5,
                fontweight="bold" if np.isclose(value, column_maxima[column]) else "normal",
            )

    # Fine cell borders and a wider divider between CV and hold-out columns.
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4, alpha=0.95)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(1.5, color="white", linewidth=6.0, zorder=3)

    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.15, top=0.82)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Figure saved to: {output_path}")
    return fig, ax


# =============================================================================

# Figure 4：Plot observed versus predicted PM2.5 for all benchmark models.

# =============================================================================

def plot_all_model_predictions(
    predictions,
    fitted_models,
    development_df,
    features,
    target,
    benchmark_results,
    output_path,
    model_order=None,
    figure_size=(12.5, 8.4),
    train_color="#4C956C",
    test_color="#7B5EA7",
    identity_color="#666666",
    fitted_color="#222222",
    train_point_size=10,
    test_point_size=7,
    train_alpha=0.8,
    test_alpha=0.65,
    dpi=300,
):
    """
    Plot observed versus predicted PM2.5 for all benchmark models.

    Each panel includes:
    - development-set observations and predictions
    - held-out test observations and predictions
    - 1:1 reference line
    - fitted regression line based on held-out test data
    - Test R², RMSE, and MAE
    """

    from matplotlib.lines import Line2D
    from scipy.stats import linregress

    required_result_columns = [
        "Model",
        "Test R2",
        "Test RMSE",
        "Test MAE",
    ]

    missing_result_columns = [
        column
        for column in required_result_columns
        if column not in benchmark_results.columns
    ]

    if missing_result_columns:
        raise ValueError(
            f"Missing benchmark columns: "
            f"{missing_result_columns}"
        )

    if not isinstance(predictions, dict):
        raise TypeError(
            "`predictions` must be a dictionary keyed by model name."
        )

    if not isinstance(fitted_models, dict):
        raise TypeError(
            "`fitted_models` must be a dictionary keyed by model name."
        )

    available_models = [
        model
        for model in predictions.keys()
        if model in fitted_models
    ]

    if model_order is None:
        model_order = available_models
    else:
        valid_models = [
            model
            for model in model_order
            if model in available_models
        ]

        remaining_models = [
            model
            for model in available_models
            if model not in valid_models
        ]

        model_order = valid_models + remaining_models

    if not model_order:
        raise ValueError(
            "No valid models were found in both predictions "
            "and fitted_models."
        )

    # Development-set values
    X_development = development_df[features].copy()

    y_development = pd.to_numeric(
        development_df[target],
        errors="coerce",
    )

    # Validate held-out prediction tables
    for model_name in model_order:
        prediction_df = predictions[model_name]

        required_prediction_columns = [
            "Observed",
            "Predicted",
        ]

        missing_columns = [
            column
            for column in required_prediction_columns
            if column not in prediction_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{model_name} prediction data are missing: "
                f"{missing_columns}"
            )

    # Calculate development predictions
    development_predictions = {}

    for model_name in model_order:
        model = fitted_models[model_name]

        development_predictions[model_name] = pd.DataFrame(
            {
                "Observed": y_development.to_numpy(),
                "Predicted": model.predict(X_development),
            }
        )

    # Common axis limits across development and test data
    all_values = []

    for model_name in model_order:
        train_df = development_predictions[model_name]
        test_df = predictions[model_name]

        all_values.extend(
            pd.to_numeric(
                train_df["Observed"],
                errors="coerce",
            ).dropna().tolist()
        )

        all_values.extend(
            pd.to_numeric(
                train_df["Predicted"],
                errors="coerce",
            ).dropna().tolist()
        )

        all_values.extend(
            pd.to_numeric(
                test_df["Observed"],
                errors="coerce",
            ).dropna().tolist()
        )

        all_values.extend(
            pd.to_numeric(
                test_df["Predicted"],
                errors="coerce",
            ).dropna().tolist()
        )

    all_values = np.asarray(
        all_values,
        dtype=float,
    )

    # Fixed axis limits for all benchmark models
    lower_limit = 0
    upper_limit = 60

    n_models = len(model_order)
    n_columns = 3
    n_rows = int(
        np.ceil(n_models / n_columns)
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figure_size,
        sharex=False,
        sharey=False,
    )

    axes = np.asarray(axes).reshape(-1)

    result_lookup = benchmark_results.set_index("Model")

    for index, model_name in enumerate(model_order):
        ax = axes[index]

        train_df = development_predictions[
            model_name
        ].copy()

        test_df = predictions[
            model_name
        ].copy()

        train_observed = pd.to_numeric(
            train_df["Observed"],
            errors="coerce",
        )

        train_predicted = pd.to_numeric(
            train_df["Predicted"],
            errors="coerce",
        )

        test_observed = pd.to_numeric(
            test_df["Observed"],
            errors="coerce",
        )

        test_predicted = pd.to_numeric(
            test_df["Predicted"],
            errors="coerce",
        )

        train_valid = (
            train_observed.notna()
            & train_predicted.notna()
        )

        test_valid = (
            test_observed.notna()
            & test_predicted.notna()
        )

        train_observed = train_observed[train_valid]
        train_predicted = train_predicted[train_valid]

        test_observed = test_observed[test_valid]
        test_predicted = test_predicted[test_valid]

        # Development / train points
        ax.scatter(
            train_observed,
            train_predicted,
            s=train_point_size,
            alpha=train_alpha,
            color=train_color,
            edgecolors="none",
            rasterized=True,
            label="Train data",
            zorder=1,
        )

        # Held-out test points
        ax.scatter(
            test_observed,
            test_predicted,
            s=test_point_size,
            alpha=test_alpha,
            color=test_color,
            edgecolors="white",
            linewidth=0.45,
            rasterized=True,
            label="Test data",
            zorder=2,
        )

        # 1:1 reference line
        ax.plot(
            [lower_limit, upper_limit],
            [lower_limit, upper_limit],
            linestyle="--",
            linewidth=1.2,
            color=identity_color,
            label="1:1 line",
            zorder=3,
        )

        # Fitted line using held-out test points
        slope, intercept, _, _, _ = linregress(
            test_observed,
            test_predicted,
        )

        fit_x = np.linspace(
            lower_limit,
            upper_limit,
            200,
        )

        fit_y = slope * fit_x + intercept

        ax.plot(
            fit_x,
            fit_y,
            color=fitted_color,
            linewidth=1.55,
            label="Fitted line",
            zorder=4,
        )

        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)

        # Fixed ticks
        ax.set_xticks(np.arange(0, 61, 10))
        ax.set_yticks(np.arange(0, 61, 10))

        ax.set_aspect(
            "equal",
            adjustable="box",
        )


        ax.set_title(
            model_name,
            fontsize=11.5,
            fontweight="semibold",
            pad=6,
            color="black",
        )

        test_r2 = result_lookup.loc[
            model_name,
            "Test R2",
        ]

        test_rmse = result_lookup.loc[
            model_name,
            "Test RMSE",
        ]

        test_mae = result_lookup.loc[
            model_name,
            "Test MAE",
        ]

        ax.text(
            0.04,
            0.96,
            (
                f"Test R² = {test_r2:.3f}\n"
                f"Test RMSE = {test_rmse:.2f} µg/m³\n"
                f"Test MAE = {test_mae:.2f} µg/m³"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "#CFCFCF",
                "linewidth": 0.5,
                "alpha": 0.82,
                "pad": 3.0,
            },
        )

        # Ensure numerical ticks appear on every panel
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
            labelsize=8.5,
            labelcolor="black",
            color="#B0B0B0",
            width=0.75,
            length=3.5,
        )

        ax.grid(
            linestyle="--",
            linewidth=0.65,
            color="#E0E0E0",
            alpha=0.45,
        )

        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#B0B0B0")
            spine.set_linewidth(0.75)

    fig.supxlabel(
        "Observed PM2.5 (µg/m³)",
        fontsize=11,
        fontweight="semibold",
        y=0.065,
    )

    fig.supylabel(
        "Predicted PM2.5 (µg/m³)",
        fontsize=11,
        fontweight="semibold",
        x=0.060,
    )

    # Shared legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=train_color,
            markeredgecolor="none",
            markersize=6,
            alpha=0.75,
            label="Train data",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=test_color,
            markeredgecolor="white",
            markersize=7,
            label="Test data",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=1.3,
            color=identity_color,
            label="1:1 line",
        ),
        Line2D(
            [0],
            [0],
            linestyle="-",
            linewidth=1.6,
            color=fitted_color,
            label="Fitted line",
        ),
    ]

    # Use the unused subplot as a legend panel
    for index in range(n_models, len(axes)):
        legend_ax = axes[index]

        legend_ax.axis("off")

        legend_ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.08, 0.82),
            frameon=False,
            fontsize=10,
            ncol=1,
            labelspacing=1.2,
            handlelength=2.0,
        )

    # Tighter panel spacing
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.12,
        top=0.96,
        wspace=0.16,
        hspace=0.20,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print(
        f"Figure saved to: {output_path}"
    )

    return fig, axes


# =============================================================================

# Figure 5：Residual diagnostics

# =============================================================================

def plot_best_model_composite_diagnostics(
    predictions,
    fitted_models,
    development_df,
    features,
    target,
    benchmark_results,
    model_name,
    output_path,
    figure_size=(9.5, 10.5),
    train_color="#A8D5BA",
    test_color="#6C4AB6",
    fitted_color="#222222",
    identity_color="#666666",
    zero_line_color="#666666",
    train_alpha=0.85,
    test_alpha=0.25,
    dpi=300,
):
    """
    Create a composite diagnostic figure for the best-performing model.

    Layout
    ------
    - Top: observed-value marginal distributions
    - Centre: observed versus predicted scatter
    - Right: predicted-value marginal distributions
    - Bottom: residuals versus predicted values

    Parameters
    ----------
    predictions : dict
        Held-out prediction DataFrames keyed by model name.
        Each DataFrame must contain:
        - Observed
        - Predicted

    fitted_models : dict
        Fitted estimators keyed by model name.

    development_df : pandas.DataFrame
        Development/training dataset.

    features : list
        Predictor columns.

    target : str
        Target column.

    benchmark_results : pandas.DataFrame
        Model summary containing Test R2, Test RMSE and Test MAE.

    model_name : str
        Best-performing model to visualise.

    output_path : str or pathlib.Path
        Figure output path.

    figure_size : tuple
        Figure size in inches.

    train_color : str
        Colour used for development/training data.

    test_color : str
        Colour used for held-out test data.

    fitted_color : str
        Colour used for the regression line.

    identity_color : str
        Colour used for the 1:1 reference line.

    zero_line_color : str
        Colour used for the zero-residual line.

    train_alpha : float
        Transparency used for training observations.

    test_alpha : float
        Transparency used for test observations.

    dpi : int
        Figure output resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    axes : dict
        Dictionary containing top, main, right and residual axes.
    """

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from scipy.stats import gaussian_kde, linregress
    from matplotlib.lines import Line2D

    # ---------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------
    required_result_columns = [
        "Model",
        "Test R2",
        "Test RMSE",
        "Test MAE",
    ]

    missing_result_columns = [
        column
        for column in required_result_columns
        if column not in benchmark_results.columns
    ]

    if missing_result_columns:
        raise ValueError(
            f"Missing benchmark columns: "
            f"{missing_result_columns}"
        )

    if model_name not in predictions:
        raise KeyError(
            f"Model '{model_name}' not found in predictions."
        )

    if model_name not in fitted_models:
        raise KeyError(
            f"Model '{model_name}' not found in fitted_models."
        )

    # ---------------------------------------------------------
    # Prepare train and test predictions
    # ---------------------------------------------------------
    model = fitted_models[model_name]

    train_observed = pd.to_numeric(
        development_df[target],
        errors="coerce",
    )

    train_predicted = pd.Series(
        model.predict(
            development_df[features]
        ),
        index=development_df.index,
    )

    train_valid = (
        train_observed.notna()
        & train_predicted.notna()
    )

    train_observed = (
        train_observed[train_valid]
        .to_numpy()
    )

    train_predicted = (
        train_predicted[train_valid]
        .to_numpy()
    )

    test_df = predictions[model_name].copy()

    required_prediction_columns = [
        "Observed",
        "Predicted",
    ]

    missing_prediction_columns = [
        column
        for column in required_prediction_columns
        if column not in test_df.columns
    ]

    if missing_prediction_columns:
        raise ValueError(
            f"Prediction data for '{model_name}' are missing "
            f"columns: {missing_prediction_columns}"
        )

    test_observed = pd.to_numeric(
        test_df["Observed"],
        errors="coerce",
    )

    test_predicted = pd.to_numeric(
        test_df["Predicted"],
        errors="coerce",
    )

    test_valid = (
        test_observed.notna()
        & test_predicted.notna()
    )

    test_observed = (
        test_observed[test_valid]
        .to_numpy()
    )

    test_predicted = (
        test_predicted[test_valid]
        .to_numpy()
    )

    if len(test_observed) < 2:
        raise ValueError(
            "At least two valid held-out test observations "
            "are required."
        )

    train_residuals = (
        train_observed
        - train_predicted
    )

    test_residuals = (
        test_observed
        - test_predicted
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    result_row = (
        benchmark_results
        .set_index("Model")
        .loc[model_name]
    )

    test_r2 = float(
        result_row["Test R2"]
    )

    test_rmse = float(
        result_row["Test RMSE"]
    )

    test_mae = float(
        result_row["Test MAE"]
    )

    # ---------------------------------------------------------
    # Fixed axis limits
    # ---------------------------------------------------------
    x_min = 0
    x_max = 60

    y_min = 0
    y_max = 60

    # ---------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------
    fig = plt.figure(
        figsize=figure_size,
    )

    # Outer layout:
    # upper row = marginal distributions + main scatter
    # lower row = residual plot
    #
    # Two columns ensure that the residual plot has exactly
    # the same width as the main scatter panel.
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[
            5.9,
            2.8,
        ],
        width_ratios=[
            5.4,
            1.15,
        ],
        hspace=0.14,
        wspace=0.0,
    )

    # Left part of the upper block:
    # top marginal + main scatter
    upper_left_grid = (
        outer_grid[0, 0]
        .subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[
                0.9,
                5.0,
            ],
            hspace=0.0,
        )
    )

    # Right part of the upper block:
    # empty top-right cell + right marginal
    upper_right_grid = (
        outer_grid[0, 1]
        .subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=[
                0.9,
                5.0,
            ],
            hspace=0.0,
        )
    )

    # Create main panel first so marginal panels can share axes
    ax_main = fig.add_subplot(
        upper_left_grid[1, 0]
    )

    ax_top = fig.add_subplot(
        upper_left_grid[0, 0],
        sharex=ax_main,
    )

    ax_right = fig.add_subplot(
        upper_right_grid[1, 0],
        sharey=ax_main,
    )

    ax_empty_top_right = fig.add_subplot(
        upper_right_grid[0, 0]
    )

    ax_residual = fig.add_subplot(
        outer_grid[1, 0]
    )

    ax_empty_bottom_right = fig.add_subplot(
        outer_grid[1, 1]
    )

    ax_empty_top_right.axis("off")
    ax_empty_bottom_right.axis("off")

    # ---------------------------------------------------------
    # Configure marginal axes
    # ---------------------------------------------------------
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    for spine in ax_right.spines.values():
        spine.set_visible(False)

    ax_top.set_xlabel("")
    ax_top.set_ylabel("")

    ax_top.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    ax_top.grid(False)

    ax_right.set_xlabel("")
    ax_right.set_ylabel("")

    ax_right.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    ax_right.grid(False)

    # Prevent automatic margins at the shared edges
    ax_top.margins(
        x=0,
    )

    ax_right.margins(
        y=0,
    )

    # ---------------------------------------------------------
    # Top marginal distribution: observed values
    # ---------------------------------------------------------
    bins_observed = np.linspace(
        x_min,
        x_max,
        34,
    )

    ax_top.hist(
        train_observed,
        bins=bins_observed,
        density=True,
        color=train_color,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
    )

    ax_top.hist(
        test_observed,
        bins=bins_observed,
        density=True,
        color=test_color,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
    )

    observed_grid = np.linspace(
        x_min,
        x_max,
        400,
    )

    if len(train_observed) > 1:
        train_kde = gaussian_kde(
            train_observed
        )

        ax_top.plot(
            observed_grid,
            train_kde(
                observed_grid
            ),
            color=train_color,
            linewidth=2.0,
        )

    if len(test_observed) > 1:
        test_kde = gaussian_kde(
            test_observed
        )

        ax_top.plot(
            observed_grid,
            test_kde(
                observed_grid
            ),
            color=test_color,
            linewidth=2.0,
        )

    ax_top.set_xlim(
        x_min,
        x_max,
    )

    # Model name above the marginal panel
    ax_top.set_title(
        model_name,
        fontsize=13,
        fontweight="semibold",
        pad=3,
        color="black",
    )

    # Small density label inside the panel
    ax_top.text(
        -0.075,
        0.50,
        "Density",
        transform=ax_top.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        color="black",
        clip_on=False,
    )

    # ---------------------------------------------------------
    # Main scatter plot
    # ---------------------------------------------------------
    ax_main.scatter(
        train_observed,
        train_predicted,
        s=14,
        color=train_color,
        alpha=train_alpha,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    ax_main.scatter(
        test_observed,
        test_predicted,
        s=28,
        color=test_color,
        alpha=test_alpha,
        edgecolors="white",
        linewidth=0.35,
        rasterized=True,
        zorder=2,
    )

    # 1:1 reference line
    identity_min = max(
        x_min,
        y_min,
    )

    identity_max = min(
        x_max,
        y_max,
    )

    ax_main.plot(
        [
            identity_min,
            identity_max,
        ],
        [
            identity_min,
            identity_max,
        ],
        linestyle="--",
        color=identity_color,
        linewidth=1.5,
        zorder=3,
    )

    # Regression line based on held-out test observations
    slope, intercept, _, _, _ = linregress(
        test_observed,
        test_predicted,
    )

    fit_x = np.linspace(
        x_min,
        x_max,
        300,
    )

    fit_y = (
        slope * fit_x
        + intercept
    )

    ax_main.plot(
        fit_x,
        fit_y,
        color=fitted_color,
        linewidth=2.0,
        zorder=4,
    )

    ax_main.set_xlim(
        x_min,
        x_max,
    )

    ax_main.set_ylim(
        y_min,
        y_max,
    )

    ax_main.set_xticks(
        np.arange(
            0,
            61,
            10,
        )
    )

    ax_main.set_yticks(
        np.arange(
            0,
            61,
            10,
        )
    )

    ax_main.set_xlabel(
        "Observed PM2.5 (µg/m³)",
        fontsize=11,
        fontweight="semibold",
        labelpad=7,
    )

    ax_main.set_ylabel(
        "Predicted PM2.5 (µg/m³)",
        fontsize=11,
        fontweight="semibold",
        labelpad=7,
    )

    ax_main.grid(
        linestyle="--",
        linewidth=0.7,
        color="#E0E0E0",
        alpha=0.45,
    )

    ax_main.set_axisbelow(
        True
    )

    # Test metric annotation
    ax_main.text(
        0.04,
        0.96,
        (
            f"Test R² = {test_r2:.3f}\n"
            f"RMSE = {test_rmse:.2f} µg/m³\n"
            f"MAE = {test_mae:.2f} µg/m³"
        ),
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "#D5D5D5",
            "linewidth": 0.5,
            "alpha": 0.88,
            "pad": 3.5,
        },
    )

    # Shared legend for the main panel
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=train_color,
            markeredgecolor="none",
            markersize=7,
            label="Train data",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=test_color,
            markeredgecolor="white",
            markersize=8,
            label="Test data",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            linewidth=1.5,
            color=identity_color,
            label="1:1 line",
        ),
        Line2D(
            [0],
            [0],
            linestyle="-",
            linewidth=2.0,
            color=fitted_color,
            label="Regression line",
        ),
    ]

    ax_main.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=10,
        labelspacing=0.55,
        handlelength=2.0,
    )

    # ---------------------------------------------------------
    # Right marginal distribution: predicted values
    # ---------------------------------------------------------
    bins_predicted = np.linspace(
        y_min,
        y_max,
        32,
    )

    ax_right.hist(
        train_predicted,
        bins=bins_predicted,
        density=True,
        orientation="horizontal",
        color=train_color,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
    )

    ax_right.hist(
        test_predicted,
        bins=bins_predicted,
        density=True,
        orientation="horizontal",
        color=test_color,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
    )

    predicted_grid = np.linspace(
        y_min,
        y_max,
        400,
    )

    if len(train_predicted) > 1:
        train_pred_kde = gaussian_kde(
            train_predicted
        )

        ax_right.plot(
            train_pred_kde(
                predicted_grid
            ),
            predicted_grid,
            color=train_color,
            linewidth=2.0,
        )

    if len(test_predicted) > 1:
        test_pred_kde = gaussian_kde(
            test_predicted
        )

        ax_right.plot(
            test_pred_kde(
                predicted_grid
            ),
            predicted_grid,
            color=test_color,
            linewidth=2.0,
        )

    ax_right.set_ylim(
        y_min,
        y_max,
    )

    ax_right.text(
        0.5,
        -0.035,
        "Density",
        transform=ax_right.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="black",
        clip_on=False,
    )

    # ---------------------------------------------------------
    # Residual plot
    # ---------------------------------------------------------
    ax_residual.scatter(
        train_predicted,
        train_residuals,
        s=12,
        color=train_color,
        alpha=train_alpha,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    ax_residual.scatter(
        test_predicted,
        test_residuals,
        s=24,
        color=test_color,
        alpha=test_alpha,
        edgecolors="white",
        linewidth=0.30,
        rasterized=True,
        zorder=2,
    )

    ax_residual.axhline(
        y=0,
        color=zero_line_color,
        linestyle="-",
        linewidth=1.25,
        zorder=3,
    )

    residual_limit = max(
        np.abs(
            train_residuals
        ).max(),
        np.abs(
            test_residuals
        ).max(),
    )

    residual_limit = max(
        5,
        residual_limit * 1.08,
    )

    ax_residual.set_ylim(
        -residual_limit,
        residual_limit,
    )

    ax_residual.set_xlim(
        x_min,
        x_max,
    )

    ax_residual.set_xticks(
        np.arange(
            0,
            61,
            10,
        )
    )

    ax_residual.set_xlabel(
        "Predicted PM2.5 (µg/m³)",
        fontsize=11,
        fontweight="semibold",
        labelpad=7,
    )

    ax_residual.set_ylabel(
        "Residual (Observed − Predicted)",
        fontsize=10.5,
        fontweight="semibold",
        labelpad=7,
    )

    ax_residual.grid(
        linestyle="--",
        linewidth=0.65,
        color="#E0E0E0",
        alpha=0.45,
    )

    ax_residual.set_axisbelow(
        True
    )

    ax_residual.text(
        0.98,
        0.92,
        (
            f"Mean residual (Train) = "
            f"{train_residuals.mean():.2f}\n"
            f"Mean residual (Test) = "
            f"{test_residuals.mean():.2f}"
        ),
        transform=ax_residual.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "#D5D5D5",
            "linewidth": 0.5,
            "alpha": 0.85,
            "pad": 3.0,
        },
    )

    # ---------------------------------------------------------
    # Borders and tick formatting
    # ---------------------------------------------------------
    for ax in [
        ax_main,
        ax_residual,
    ]:
        for spine in ax.spines.values():
            spine.set_visible(
                True
            )

            spine.set_color(
                "#B0B0B0"
            )

            spine.set_linewidth(
                0.8
            )

        ax.tick_params(
            axis="both",
            which="both",
            labelcolor="black",
            color="#B0B0B0",
            width=0.8,
            length=4,
            labelsize=9,
        )

    for ax in [
        ax_top,
        ax_right,
    ]:
        for spine in ax.spines.values():
            spine.set_visible(
                False
            )

    # Align the y-axis labels of the main and residual panels
    fig.align_ylabels(
        [
            ax_main,
            ax_residual,
        ]
    )

    # Final figure margins
    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.07,
        top=0.97,
    )

    # ---------------------------------------------------------
    # Save and return
    # ---------------------------------------------------------
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print(
        f"Figure saved to: {output_path}"
    )

    return fig, {
        "top": ax_top,
        "main": ax_main,
        "right": ax_right,
        "residual": ax_residual,
    }

# =============================================================================

# Figure 6：residual distribution of a selected benchmark model.

# =============================================================================

def plot_residual_distribution(
    predictions,
    model_name,
    output_path,
    figure_size=(7.5, 5.5),
    histogram_color="#A9C5DF",
    density_color="#355C9A",
    zero_line_color="#666666",
    mean_line_color="#D55E00",
    bins=28,
    dpi=300,
):
    """
    Plot the residual distribution of a selected benchmark model.

    Parameters
    ----------
    predictions : dict
        Dictionary keyed by model name. Each DataFrame must contain
        either:
        - Residual
        or:
        - Observed and Predicted

    model_name : str
        Name of the model to analyse.

    output_path : str or pathlib.Path
        Path used to save the figure.

    figure_size : tuple
        Figure size in inches.

    histogram_color : str
        Histogram fill color.

    density_color : str
        KDE curve color.

    zero_line_color : str
        Color of the zero-residual reference line.

    mean_line_color : str
        Color of the mean-residual reference line.

    bins : int
        Number of histogram bins.

    dpi : int
        Figure output resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    ax : matplotlib.axes.Axes
        Figure axis.
    """

    if model_name not in predictions:
        raise KeyError(
            f"Model '{model_name}' was not found in predictions. "
            f"Available models: {list(predictions.keys())}"
        )

    prediction_df = predictions[model_name].copy()

    # Use existing residual column when available
    if "Residual" in prediction_df.columns:
        residuals = pd.to_numeric(
            prediction_df["Residual"],
            errors="coerce",
        )

    # Otherwise calculate residuals as observed minus predicted
    elif {
        "Observed",
        "Predicted",
    }.issubset(prediction_df.columns):
        observed = pd.to_numeric(
            prediction_df["Observed"],
            errors="coerce",
        )

        predicted = pd.to_numeric(
            prediction_df["Predicted"],
            errors="coerce",
        )

        residuals = observed - predicted

    else:
        raise ValueError(
            "Prediction data must contain either 'Residual' "
            "or both 'Observed' and 'Predicted' columns."
        )

    residuals = residuals.dropna().to_numpy(dtype=float)

    if len(residuals) < 2:
        raise ValueError(
            "At least two valid residual values are required."
        )

    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals, ddof=1))
    residual_median = float(np.median(residuals))

    # Symmetric x-axis limits around zero
    residual_abs_max = np.max(
        np.abs(residuals)
    )

    x_limit = max(
        5.0,
        residual_abs_max * 1.08,
    )

    residual_mean = float(
        np.mean(residuals)
    )

    residual_std = float(
        np.std(
            residuals,
            ddof=1,
        )
    )

    residual_median = float(
        np.median(residuals)
    )

    # Symmetric residual range around zero
    residual_abs_max = float(
        np.max(
            np.abs(residuals)
        )
    )

    x_limit = max(
        5.0,
        residual_abs_max * 1.08,
    )

    # Symmetric histogram bin edges
    bin_edges = np.linspace(
        -x_limit,
        x_limit,
        bins + 1,
    )

    fig, ax = plt.subplots(
        figsize=figure_size,
    )

    # Histogram shown as density
    counts, bin_edges, patches = ax.hist(
        residuals,
        bins=bin_edges,
        density=True,
        color=histogram_color,
        edgecolor="white",
        linewidth=0.8,
        label="Residuals",
        zorder=2,
    )

    # Kernel density estimate
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(residuals)

        x_values = np.linspace(
            -x_limit,
            x_limit,
            400,
        )

        density_values = kde(x_values)

        ax.plot(
            x_values,
            density_values,
            color=density_color,
            linewidth=1.6,
            label="KDE",
            zorder=4,
        )

    except ImportError:
        x_min = residuals.min()
        x_max = residuals.max()

        print(
            "scipy is not installed; the KDE curve was skipped."
        )

    # Zero residual reference line
    ax.axvline(
        x=0,
        color=zero_line_color,
        linestyle="--",
        linewidth=1.5,
        label="Zero residual",
        zorder=5,
    )

    # Mean residual line
    ax.axvline(
        x=residual_mean,
        color=mean_line_color,
        linestyle="-",
        linewidth=1.6,
        label=f"Mean residual",
        zorder=5,
    )

    ax.set_xlabel(
        "Residual (µg/m³)",
        fontweight="semibold",
        fontsize=11,
    )

    ax.set_ylabel(
        "Density",
        fontweight="semibold",
        fontsize=11,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
        labelcolor="black",
        color="#B0B0B0",
        width=0.8,
        length=4,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        color="#E0E0E0",
        alpha=0.45,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Add compact residual statistics
    statistics_text = (
        f"n = {len(residuals):,}\n"
        f"Mean = {residual_mean:.2f}\n"
        f"Median = {residual_median:.2f}\n"
        f"SD = {residual_std:.2f}"
    )

    ax.set_xlim(
        -x_limit,
        x_limit,
    )

    ax.text(
        0.97,
        0.95,
        statistics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "#D8D8D8",
            "linewidth": 0.5,
            "alpha": 0.86,
            "pad": 4,
        },
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print(f"Figure saved to: {output_path}")

    return fig, ax


# =============================================================================

# Figure 7: Compare random-split and chronological-split test performance.

# =============================================================================

def plot_random_chronological_comparison(
    comparison_results,
    output_path,
    model_order=None,
    figure_size=(11, 5.8),
    random_r2_color="#7A4EA3",
    random_rmse_color="#A98AC7",
    chronological_r2_color="#245A9A",
    chronological_rmse_color="#7FA1C8",
    grid_color="#E0E0E0",
    r2_marker_size=7,
    rmse_marker_size=7,
    line_width=1.8,
    dpi=300,
):
    """
    Compare random-split and chronological-split performance
    using a dual-axis line chart.

    The figure displays four series:

    - Random split Test R²
    - Random split Test RMSE
    - Chronological split Test R²
    - Chronological split Test RMSE

    Parameters
    ----------
    comparison_results : pandas.DataFrame
        DataFrame containing the following columns:

        - Model
        - Random R2
        - Chronological R2
        - Random RMSE
        - Chronological RMSE

    output_path : str or pathlib.Path
        Output path used to save the figure.

    model_order : list of str, optional
        Display order of benchmark models.

    figure_size : tuple, default=(11, 5.8)
        Figure dimensions.

    random_r2_color : str
        Color for random-split R².

    random_rmse_color : str
        Color for random-split RMSE.

    chronological_r2_color : str
        Color for chronological-split R².

    chronological_rmse_color : str
        Color for chronological-split RMSE.

    grid_color : str
        Color of the background grid.

    r2_marker_size : float
        Marker size for R² series.

    rmse_marker_size : float
        Marker size for RMSE series.

    line_width : float
        Width of all lines.

    dpi : int
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.

    axes : tuple
        Tuple containing the R² axis and RMSE axis.
    """

    required_columns = [
        "Model",
        "Random R2",
        "Chronological R2",
        "Random RMSE",
        "Chronological RMSE",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in comparison_results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    plot_df = comparison_results.loc[
        :,
        required_columns,
    ].copy()

    # ---------------------------------------------------------
    # Validate model order
    # ---------------------------------------------------------

    if model_order is not None:
        missing_models = [
            model
            for model in model_order
            if model not in plot_df["Model"].values
        ]

        if missing_models:
            raise ValueError(
                "Models missing from comparison results: "
                f"{missing_models}"
            )

        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=model_order,
            ordered=True,
        )

        plot_df = (
            plot_df
            .sort_values("Model")
            .reset_index(drop=True)
        )

    # Check numeric columns
    numeric_columns = [
        "Random R2",
        "Chronological R2",
        "Random RMSE",
        "Chronological RMSE",
    ]

    if plot_df[numeric_columns].isna().any().any():
        raise ValueError(
            "Comparison results contain missing metric values."
        )

    models = plot_df["Model"].astype(str).tolist()
    x = np.arange(len(models))

    random_r2 = (
        plot_df["Random R2"]
        .astype(float)
        .to_numpy()
    )

    chronological_r2 = (
        plot_df["Chronological R2"]
        .astype(float)
        .to_numpy()
    )

    random_rmse = (
        plot_df["Random RMSE"]
        .astype(float)
        .to_numpy()
    )

    chronological_rmse = (
        plot_df["Chronological RMSE"]
        .astype(float)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Create figure and dual axes
    # ---------------------------------------------------------

    fig, ax_r2 = plt.subplots(
        figsize=figure_size
    )

    ax_rmse = ax_r2.twinx()

    # ---------------------------------------------------------
    # Plot R² series
    # ---------------------------------------------------------

    line_random_r2, = ax_r2.plot(
        x,
        random_r2,
        color=random_r2_color,
        linestyle="-",
        linewidth=line_width,
        marker="o",
        markersize=r2_marker_size,
        markerfacecolor=random_r2_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Random split R²",
        zorder=4,
    )

    line_chronological_r2, = ax_r2.plot(
        x,
        chronological_r2,
        color=chronological_r2_color,
        linestyle="-",
        linewidth=line_width,
        marker="o",
        markersize=r2_marker_size,
        markerfacecolor=chronological_r2_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Chronological split R²",
        zorder=4,
    )

    # ---------------------------------------------------------
    # Plot RMSE series
    # ---------------------------------------------------------

    line_random_rmse, = ax_rmse.plot(
        x,
        random_rmse,
        color=random_rmse_color,
        linestyle="--",
        linewidth=line_width,
        marker="s",
        markersize=rmse_marker_size,
        markerfacecolor=random_rmse_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Random split RMSE",
        zorder=3,
    )

    line_chronological_rmse, = ax_rmse.plot(
        x,
        chronological_rmse,
        color=chronological_rmse_color,
        linestyle="--",
        linewidth=line_width,
        marker="s",
        markersize=rmse_marker_size,
        markerfacecolor=chronological_rmse_color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Chronological split RMSE",
        zorder=3,
    )

    # ---------------------------------------------------------
    # Axis limits
    # ---------------------------------------------------------

    r2_min = min(
        random_r2.min(),
        chronological_r2.min(),
    )

    r2_max = max(
        random_r2.max(),
        chronological_r2.max(),
    )

    r2_range = max(
        r2_max - r2_min,
        1.0,
    )

    ax_r2.set_ylim(
        r2_min - 0.20 * r2_range,
        r2_max + 0.30 * r2_range,
    )

    rmse_min = min(
        random_rmse.min(),
        chronological_rmse.min(),
    )

    rmse_max = max(
        random_rmse.max(),
        chronological_rmse.max(),
    )

    rmse_range = max(
        rmse_max - rmse_min,
        1.0,
    )

    ax_rmse.set_ylim(
        max(0, rmse_min - 0.45 * rmse_range),
        rmse_max + 0.60 * rmse_range,
    )

    # ---------------------------------------------------------
    # Reference line and grid
    # ---------------------------------------------------------


    # Remove background grid
    ax_r2.grid(False)
    ax_rmse.grid(False)

    ax_rmse.grid(False)

    # ---------------------------------------------------------
    # Labels and titles
    # ---------------------------------------------------------

    ax_r2.set_ylabel(
        "Test R² (higher is better)",
        fontweight="bold",
        labelpad=12,
    )

    ax_rmse.set_ylabel(
        "Test RMSE (µg/m³) (lower is better)",
        fontweight="bold",
        labelpad=12,
    )

    ax_r2.set_xticks(x)

    ax_r2.set_xticklabels(
        models,
        rotation=0,
        ha="center",
    )

    ax_r2.tick_params(
        axis="y",
        colors="black",
        length=0,
    )

    ax_rmse.tick_params(
        axis="y",
        colors="black",
        length=0,
    )

    # ---------------------------------------------------------
    # Value annotations
    # ---------------------------------------------------------

    r2_vertical_offset = 0.05 * (
        ax_r2.get_ylim()[1] - ax_r2.get_ylim()[0]
    )

    rmse_vertical_offset = 0.035 * (
        ax_rmse.get_ylim()[1] - ax_rmse.get_ylim()[0]
     )

    # Random R² labels
    for x_value, value in zip(
        x,
        random_r2,
    ):
        ax_r2.text(
            x_value,
            value + r2_vertical_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    # Chronological R² labels
    for x_value, value in zip(
        x,
        chronological_r2,
    ):
        ax_r2.text(
            x_value,
            value - r2_vertical_offset,
            f"{value:.3f}",
            ha="center",
            va="top",
            fontsize=9,
            color="black",
        )

    # Random RMSE labels
    for x_value, value in zip(
        x,
        random_rmse,
    ):
        ax_rmse.text(
            x_value,
            value - rmse_vertical_offset,
            f"{value:.2f}",
            ha="center",
            va="top",
            fontsize=9,
            color="black",
        )

    # Chronological RMSE labels
    for x_value, value in zip(
        x,
        chronological_rmse,
    ):
        ax_rmse.text(
            x_value,
            value + rmse_vertical_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    # ---------------------------------------------------------
    # Shared legend
    # ---------------------------------------------------------

    legend_handles = [
        line_random_r2,
        line_random_rmse,
        line_chronological_r2,
        line_chronological_rmse,
    ]

    legend_labels = [
        "Random split R²",
        "Random split RMSE",
        "Chronological split R²",
        "Chronological split RMSE",
    ]

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.4,
    )

    # ---------------------------------------------------------
    # Spine styling
    # ---------------------------------------------------------

    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)

    ax_rmse.spines["top"].set_visible(False)
    ax_rmse.spines["left"].set_visible(False)

    ax_r2.spines["left"].set_color("#B5B5B5")
    ax_r2.spines["bottom"].set_color("#B5B5B5")
    ax_rmse.spines["right"].set_color("#B5B5B5")

    ax_r2.set_axisbelow(True)

    # ---------------------------------------------------------
    # Figure note
    # ---------------------------------------------------------

    fig.text(
        0.5,
        0.055,
        (
            "R²: higher values indicate better performance; "
            "RMSE: lower values indicate better performance."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.90,
        top=0.98,
        bottom=0.14,
    )

    # ---------------------------------------------------------
    # Save and return
    # ---------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.show()

    return fig, (ax_r2, ax_rmse)

# =============================================================================

# Figure 8: Computational efficiency of benchmark models

# =============================================================================

def plot_computational_efficiency(
    benchmark_results,
    output_path,
    model_order=None,
    figure_size=(11, 5.5),
    train_color="#BFA9D1",
    predict_color="#8397B7",
    separator_color="#E0E0E0",
    center_line_color="#B5B5B5",
    bar_height=0.48,
    dpi=300,
):
    """
    Plot computational efficiency using a mirrored horizontal layout.

    Training times are shown on the left using a logarithmic scale.
    Prediction times are shown on the right using a linear scale.
    Model names are displayed along a central vertical axis.

    Parameters
    ----------
    benchmark_results : pandas.DataFrame
        Benchmark results containing:
        - Model
        - Train Time (s)
        - Predict Time (s)

    output_path : str or pathlib.Path
        Path used to save the figure.

    model_order : list of str, optional
        Display order of benchmark models.

    figure_size : tuple, default=(11, 5.5)
        Figure dimensions.

    train_color : str, default="#BFA9D1"
        Colour used for training-time bars.

    predict_color : str, default="#8397B7"
        Colour used for prediction-time bars.

    separator_color : str, default="#E0E0E0"
        Colour used for vertical grid lines.

    center_line_color : str, default="#B5B5B5"
        Colour used for the central model axis.

    bar_height : float, default=0.48
        Height of each horizontal bar.

    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.

    axes : tuple
        Training-time, model-label, and prediction-time axes.
    """

    required_columns = [
        "Model",
        "Train Time (s)",
        "Predict Time (s)",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in benchmark_results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    plot_df = benchmark_results.loc[
        :,
        required_columns,
    ].copy()

    # Validate timing values
    if plot_df["Train Time (s)"].isna().any():
        raise ValueError(
            "Training-time values contain missing data."
        )

    if plot_df["Predict Time (s)"].isna().any():
        raise ValueError(
            "Prediction-time values contain missing data."
        )

    if (plot_df["Train Time (s)"] <= 0).any():
        raise ValueError(
            "Training-time values must be greater than zero "
            "for logarithmic scaling."
        )

    if (plot_df["Predict Time (s)"] < 0).any():
        raise ValueError(
            "Prediction-time values cannot be negative."
        )

    # Apply requested model order
    if model_order is not None:
        missing_models = [
            model
            for model in model_order
            if model not in plot_df["Model"].values
        ]

        if missing_models:
            raise ValueError(
                "Models missing from benchmark results: "
                f"{missing_models}"
            )

        plot_df["Model"] = pd.Categorical(
            plot_df["Model"],
            categories=model_order,
            ordered=True,
        )

        plot_df = (
            plot_df
            .sort_values("Model")
            .reset_index(drop=True)
        )

    models = plot_df["Model"].astype(str).tolist()

    train_times = (
        plot_df["Train Time (s)"]
        .astype(float)
        .to_numpy()
    )

    predict_times = (
        plot_df["Predict Time (s)"]
        .astype(float)
        .to_numpy()
    )

    y_positions = np.arange(len(models))

    # ---------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=figure_size,
        constrained_layout=False,
    )

    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[
            1.0,
            0.34,
            1.0,
        ],
        wspace=0.02,
    )

    ax_train = fig.add_subplot(grid[0, 0])
    ax_model = fig.add_subplot(
        grid[0, 1],
        sharey=ax_train,
    )
    ax_predict = fig.add_subplot(
        grid[0, 2],
        sharey=ax_train,
    )

    # ---------------------------------------------------------
    # Left panel: Training time
    # ---------------------------------------------------------

    minimum_train_time = train_times.min()
    maximum_train_time = train_times.max()

    # Use a lower plotting baseline one decade below the
    # smallest observed training time.
    train_floor = 10 ** (
        np.floor(np.log10(minimum_train_time)) - 1
    )

    train_ceiling = 10 ** (
        np.ceil(np.log10(maximum_train_time))
    )

    # Ensure sufficient room for labels at the largest value
    if train_ceiling <= maximum_train_time:
        train_ceiling *= 10

    ax_train.barh(
        y=y_positions,
        width=train_times - train_floor,
        left=train_floor,
        height=bar_height,
        color=train_color,
        edgecolor="none",
        zorder=3,
    )

    ax_train.set_xscale("log")
    from matplotlib.ticker import FuncFormatter

    ax_train.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:g}")
    )

    # Reverse the training-time axis so bars extend leftward
    ax_train.set_xlim(
        train_ceiling,
        train_floor,
    )

    ax_train.set_title(
        "Training time",
        fontweight="bold",
        pad=14,
    )

    ax_train.set_xlabel(
        "Time (s) [log scale]",
        fontweight="bold",
        labelpad=10,
    )

    ax_train.set_yticks(y_positions)
    ax_train.set_yticklabels([])

    # Keep only vertical grid lines
    ax_train.grid(
        visible=False,
    )

    ax_train.grid(
        axis="x",
        color=separator_color,
        linestyle="--",
        linewidth=0.8,
        alpha=0.75,
        zorder=0,
    )

    # Training-time labels
    for y_position, value in zip(
        y_positions,
        train_times,
    ):
        ax_train.text(
            value * 1.12,
            y_position,
            f"{value:.2f}",
            ha="right",
            va="center",
            fontsize=9,
            color="black",
            clip_on=False,
        )

    # ---------------------------------------------------------
    # Centre panel: Model names
    # ---------------------------------------------------------

    ax_model.set_xlim(0, 1)
    ax_model.set_ylim(
        -0.55,
        len(models) - 0.45,
    )

    ax_model.axvline(
        x=0.5,
        color=center_line_color,
        linewidth=1.0,
        zorder=1,
    )

    for y_position, model in zip(
        y_positions,
        models,
    ):
        ax_model.text(
            0.5,
            y_position,
            model,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "pad": 2.5,
            },
            zorder=3,
        )

    ax_model.set_title(
        "Model",
        fontweight="bold",
        pad=14,
    )

    ax_model.set_xticks([])
    ax_model.set_yticks([])
    ax_model.grid(visible=False)

    for spine in ax_model.spines.values():
        spine.set_visible(False)

    # ---------------------------------------------------------
    # Right panel: Prediction time
    # ---------------------------------------------------------

    prediction_maximum = predict_times.max()

    if prediction_maximum == 0:
        prediction_axis_maximum = 1.0
    else:
        prediction_axis_maximum = (
            prediction_maximum * 1.27
        )

    ax_predict.barh(
        y=y_positions,
        width=predict_times,
        height=bar_height,
        color=predict_color,
        edgecolor="none",
        zorder=3,
    )

    ax_predict.set_xlim(
        0,
        prediction_axis_maximum,
    )

    ax_predict.set_title(
        "Prediction time",
        fontweight="bold",
        pad=14,
    )

    ax_predict.set_xlabel(
        "Time (s) [linear scale]",
        fontweight="bold",
        labelpad=10,
    )

    ax_predict.set_yticks(y_positions)
    ax_predict.set_yticklabels([])

    # Keep only vertical grid lines
    ax_predict.grid(
        visible=False,
    )

    ax_predict.grid(
        axis="x",
        color=separator_color,
        linestyle="--",
        linewidth=0.8,
        alpha=0.75,
        zorder=0,
    )

    prediction_label_offset = (
        prediction_axis_maximum * 0.015
    )

    # Prediction-time labels
    for y_position, value in zip(
        y_positions,
        predict_times,
    ):
        ax_predict.text(
            value + prediction_label_offset,
            y_position,
            f"{value:.4f}",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            clip_on=False,
        )

    # ---------------------------------------------------------
    # Shared formatting
    # ---------------------------------------------------------

    for ax in [
        ax_train,
        ax_model,
        ax_predict,
    ]:
        ax.set_ylim(
            -0.55,
            len(models) - 0.45,
        )

        ax.invert_yaxis()

    for ax in [
        ax_train,
        ax_predict,
    ]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.tick_params(
            axis="y",
            left=False,
        )

        ax.tick_params(
            axis="x",
            colors="black",
        )

        ax.set_axisbelow(True)

    # Keep only the bottom axis lines
    ax_train.spines["bottom"].set_color(
        "#B5B5B5"
    )
    ax_predict.spines["bottom"].set_color(
        "#B5B5B5"
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.87,
        bottom=0.17,
        wspace=0.02,
    )

    # ---------------------------------------------------------
    # Save and return
    # ---------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.show()

    return (
        fig,
        (
            ax_train,
            ax_model,
            ax_predict,
        ),
    )
    
