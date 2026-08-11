"""
Visualisation utilities for feature evaluation.

This module provides reusable functions for styling summary tables
and plotting feature distributions used in Notebook 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_ranking(
    correlation_results,
    output_path,
    figure_size=(8, 6.5),
    positive_color="#4C78A8",
    negative_color="#D55E00",
    dpi=300,
):

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Times New Roman"

    ranking = (
        correlation_results
        .assign(abs_r=lambda x: x["Pearson r"].abs())
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=figure_size)

    y = np.arange(len(ranking))

    for i, value in enumerate(ranking["Pearson r"]):

        color = (
            positive_color
            if value >= 0
            else negative_color
        )

        ax.hlines(
            y=i,
            xmin=0,
            xmax=value,
            color=color,
            linewidth=2.2,
        )

        ax.scatter(
            value,
            i,
            s=70,
            color=color,
            zorder=3,
        )

        ax.text(
            value + (0.02 if value >= 0 else -0.02),
            i,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
        )

    ax.axvline(
        0,
        color="0.75",
        linewidth=1,
        linestyle="--",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        ranking["Feature"],
        fontsize=11,
    )

    ax.invert_yaxis()

    max_abs = ranking["Pearson r"].abs().max()

    ax.set_xlim(
        -(max_abs + 0.08),
        max_abs + 0.08,
    )

    ax.set_xlabel(
        "Pearson correlation coefficient (r)",
        fontsize=12,
    )

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig


def plot_image_feature_group_performance(
    image_group_results,
    output_path=None,
    metric="Test R2",
    figure_size=(8, 5.5),
    ridge_color="#A8C5DD",
    rf_color="#4C78A8",
    dpi=300,
):
    """
    Compare development cross-validation performance of image feature groups
    using Ridge regression and Random Forest.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    required_columns = {
        "Model",
        "Feature Set",
        metric,
    }

    missing_columns = (
        required_columns
        - set(image_group_results.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    plt.rcParams["font.family"] = "Times New Roman"

    feature_order = [
        "Color",
        "Contrast",
        "Visibility",
        "All Image Features",
    ]

    model_order = [
        "Ridge",
        "Random Forest",
    ]

    plot_df = image_group_results.copy()

    pivot_df = (
        plot_df
        .pivot(
            index="Feature Set",
            columns="Model",
            values=metric,
        )
        .reindex(feature_order)
    )

    missing_models = [
        model
        for model in model_order
        if model not in pivot_df.columns
    ]

    if missing_models:
        raise ValueError(
            f"Missing model results: {missing_models}"
        )

    x = np.arange(len(feature_order))
    width = 0.34

    fig, ax = plt.subplots(
        figsize=figure_size,
    )

    ridge_values = pivot_df["Ridge"].to_numpy()
    rf_values = pivot_df["Random Forest"].to_numpy()

    ridge_bars = ax.bar(
        x - width / 2,
        ridge_values,
        width=width,
        label="Ridge",
        color=ridge_color,
        edgecolor="white",
        linewidth=0.8,
    )

    rf_bars = ax.bar(
        x + width / 2,
        rf_values,
        width=width,
        label="Random Forest",
        color=rf_color,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.bar_label(
        ridge_bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )

    ax.bar_label(
        rf_bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        feature_order,
        fontsize=11,
    )

    ax.set_ylabel(
        "Development CV R²" if metric == "CV R2" else "Independent test R²",
        fontsize=12,
    )

    ax.set_xlabel("")

    ax.tick_params(
        axis="y",
        labelsize=11,
    )

    ax.grid(False)

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.18,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.legend(
        frameon=False,
        fontsize=10,
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_values = np.concatenate(
        [
            ridge_values,
            rf_values,
        ]
    )

    y_min = max(
        0,
        all_values.min() - 0.08,
    )

    y_max = min(
        1,
        all_values.max() + 0.08,
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig


def plot_progressive_feature_fusion(
    fusion_results,
    output_path=None,
    model_name="Random Forest",
    metric="Test R2",
    figure_size=(9.5, 5),
    line_color="#4C78A8",
    point_color="#355C9A",
    dpi=300,
):
    """
    Plot progressive multisource feature fusion performance.

    Parameters
    ----------
    fusion_results : pandas.DataFrame
        Evaluation results containing Model, Feature Set,
        and the selected performance metric.

    output_path : str or pathlib.Path, optional
        Path used to save the figure.

    model_name : str, default="Random Forest"
        Model whose fusion results are displayed.

    metric : str, default="Test R2"
        Performance metric shown on the y-axis.

    figure_size : tuple, default=(9.5, 5)
        Figure dimensions.

    line_color : str
        Color of the connecting line.

    point_color : str
        Color of the feature-set nodes.

    dpi : int, default=300
        Output resolution.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "Times New Roman"

    required_columns = {
        "Model",
        "Feature Set",
        metric,
    }

    missing_columns = (
        required_columns
        - set(fusion_results.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    feature_order = [
        "Image",
        "Image + ERA5",
        "Image + ERA5 + ARPA",
        "All Features",
    ]

    plot_df = (
        fusion_results[
            fusion_results["Model"] == model_name
        ]
        .copy()
        .set_index("Feature Set")
        .reindex(feature_order)
        .reset_index()
    )

    if plot_df[metric].isna().any():
        missing_sets = plot_df.loc[
            plot_df[metric].isna(),
            "Feature Set",
        ].tolist()

        raise ValueError(
            f"Missing results for feature sets: {missing_sets}"
        )

    plot_df["R2 Gain"] = (
        plot_df[metric]
        .diff()
    )

    x_positions = np.arange(
        len(plot_df)
    )

    metric_values = (
        plot_df[metric]
        .to_numpy()
    )

    metric_gains = (
        plot_df["R2 Gain"]
        .to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=figure_size,
    )

    ax.plot(
        x_positions,
        metric_values,
        color=line_color,
        linewidth=2.2,
        marker="o",
        markersize=9,
        markerfacecolor=point_color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )

    value_offset = max(
        0.012,
        0.04 * (
            metric_values.max()
            - metric_values.min()
        ),
    )

    for i, (
        x_position,
        metric_value,
    ) in enumerate(
        zip(
            x_positions,
            metric_values,
        )
    ):

        ax.text(
            x_position,
            metric_value + value_offset,
            f"R² = {metric_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

        if i > 0:

            midpoint_x = (
                x_positions[i - 1]
                + x_positions[i]
            ) / 2

            midpoint_y = (
                metric_values[i - 1]
                + metric_values[i]
            ) / 2

            gain = metric_gains[i]

            gain_label = (
                f"ΔR² = {gain:+.3f}"
            )

            ax.text(
                midpoint_x,
                midpoint_y + value_offset * 0.65,
                gain_label,
                ha="center",
                va="bottom",
                fontsize=9.5,
            )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        feature_order,
        fontsize=10.5,
    )

    ax.set_ylabel(
        "Development CV R²" if metric == "CV R2" else "Independent test R²",
        fontsize=12,
    )

    ax.set_xlabel("")

    y_range = max(
        0.10,
        metric_values.max()
        - metric_values.min(),
    )

    ax.set_ylim(
        metric_values.min()
        - 0.30 * y_range,
        metric_values.max()
        + 0.50 * y_range,
    )

    ax.tick_params(
        axis="y",
        labelsize=11,
    )

    ax.grid(False)

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.18,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig
