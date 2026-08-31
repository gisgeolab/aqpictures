"""Publication and supplementary figures for the pretrained-CNN study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.plot_style import COLORS, style_axes


def _save(fig, output_path, dpi=300):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def plot_configuration_comparison(image_folds, fusion_folds, output_path, dpi=300):
    """Show fold-wise relative improvement from image-only to Fusion."""
    required = {"Fold", "RMSE", "R2"}
    for name, data in (("image_folds", image_folds), ("fusion_folds", fusion_folds)):
        if missing := required - set(data.columns):
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    paired = image_folds[["Fold", "RMSE", "R2"]].merge(
        fusion_folds[["Fold", "RMSE", "R2"]],
        on="Fold", suffixes=(" Image only", " Fusion"),
        validate="one_to_one",
    ).sort_values("Fold")

    rmse_improvement = (
        (paired["RMSE Image only"] - paired["RMSE Fusion"])
        / paired["RMSE Image only"] * 100
    ).to_numpy(float)
    r2_improvement = (
        (paired["R2 Fusion"] - paired["R2 Image only"])
        / paired["R2 Image only"].abs() * 100
    ).to_numpy(float)
    mean_rmse_improvement = rmse_improvement.mean()
    mean_r2_improvement = r2_improvement.mean()

    labels = [f"Fold {int(value)}" for value in paired["Fold"]]
    x = np.arange(len(labels))
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.2, 4.3))
    rmse_bars = axis.bar(
        x - width / 2, rmse_improvement, width,
        color=COLORS["primary"], alpha=0.88,
        label="RMSE reduction",
    )
    r2_bars = axis.bar(
        x + width / 2, r2_improvement, width,
        color=COLORS["secondary"], alpha=0.88,
        label="R² gain",
    )
    axis.axhline(0, color=COLORS["reference"], linewidth=1.0)
    axis.axhline(
        mean_rmse_improvement,
        color=COLORS["tertiary"],
        linewidth=1.5,
        linestyle="--",
        label=f"Mean RMSE reduction ({mean_rmse_improvement:.1f}%)",
        zorder=0.5,
    )
    axis.axhline(
        mean_r2_improvement,
        color=COLORS["moderate"],
        linewidth=1.5,
        linestyle=":",
        label=f"Mean R² gain ({mean_r2_improvement:.1f}%)",
        zorder=0.5,
    )

    for bars in (rmse_bars, r2_bars):
        for bar in bars:
            value = bar.get_height()
            axis.annotate(
                f"{value:.1f}%",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 8 if value >= 0 else -8),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                color=COLORS["text"],
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 0.6,
                },
            )

    axis.set_xticks(x, labels)
    axis.set_ylabel("Relative improvement with Fusion (%)")
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=1.6,
    )
    style_axes(axis, grid_axis="y")
    axis.margins(y=0.15)
    fig.tight_layout()
    _save(fig, output_path, dpi)
    return fig, axis


def plot_holdout_diagnostics(predictions, metrics, development_maximum,
                             output_path, dpi=300):
    """Show CNN hold-out agreement with concentration-stratified observations."""
    data = predictions.copy()
    observed = data["Observed"].to_numpy(float)
    predicted = data["Predicted"].to_numpy(float)
    residual = predicted - observed
    limit = max(observed.max(), predicted.max()) * 1.04
    slope, intercept, *_ = linregress(observed, predicted)

    fig, axis = plt.subplots(figsize=(7.4, 5.5))
    concentration_groups = (
        (observed <= 20, "≤20", COLORS["primary"]),
        ((observed > 20) & (observed <= 35), "20–35", COLORS["moderate"]),
        (observed > 35, ">35", COLORS["high"]),
    )
    for mask, label, color in concentration_groups:
        axis.scatter(
            observed[mask], predicted[mask],
            s=25, color=color, alpha=0.48,
            edgecolor="white", linewidth=0.35,
            label=label, zorder=2,
        )

    axis.plot(
        [0, limit], [0, limit], "--",
        color=COLORS["reference"], linewidth=1.3,
        label="1:1 line", zorder=1,
    )
    axis.plot(
        [0, limit], intercept + slope * np.asarray([0, limit]),
        color=COLORS["secondary"], linewidth=1.8,
        label="Calibration line", zorder=3,
    )
    if development_maximum < limit:
        axis.axvspan(
            development_maximum, limit,
            color=COLORS["tertiary"], alpha=0.07, linewidth=0,
        )

    axis.set(
        xlim=(0, limit), ylim=(0, limit),
        xlabel=r"Observed PM$_{2.5}$ ($\mu$g m$^{-3}$)",
        ylabel=r"Predicted PM$_{2.5}$ ($\mu$g m$^{-3}$)",
    )
    axis.text(
        0.03, 0.97,
        f"Common hold-out\n$R^2$ = {metrics['R2']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.2f} $\\mu$g m$^{{-3}}$\n"
        f"MAE = {metrics['MAE']:.2f} $\\mu$g m$^{{-3}}$",
        transform=axis.transAxes, ha="left", va="top", fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.92},
    )
    axis.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13),
        ncol=5, columnspacing=1.1, handletextpad=0.4,
    )
    style_axes(axis, grid_axis="both")

    residual_axis = axis.inset_axes([0.62, 0.17, 0.33, 0.25])
    residual_axis.set_facecolor("white")
    residual_axis.hist(
        residual, bins=18,
        color=COLORS["tertiary_light"],
        edgecolor="white", linewidth=0.45,
    )
    residual_axis.axvline(0, color=COLORS["reference"], linewidth=1.0)
    residual_axis.axvline(
        residual.mean(), color=COLORS["tertiary"],
        linewidth=1.4, linestyle="--",
    )
    residual_axis.set_xlabel(
        r"Residual ($\mu$g m$^{-3}$)",
        fontsize=8,
        labelpad=1,
    )
    residual_axis.set_ylabel("Count", fontsize=8)
    residual_axis.tick_params(labelsize=7)
    style_axes(residual_axis, grid_axis="y")
    for spine in residual_axis.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["reference"])
        spine.set_linewidth(0.7)

    fig.tight_layout()
    _save(fig, output_path, dpi)
    return fig, axis


def plot_all_model_comparison(results, output_path, dpi=300):
    """Compare CV and common hold-out RMSE and R² for all models."""
    data = results.sort_values("CV RMSE").reset_index(drop=True)
    data["Plot label"] = data["Model"].replace(
        {"EfficientNet-B0 Fusion": "EfficientNet Fusion"}
    )
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for panel, (axis, cv_metric, test_metric, title, ylabel) in enumerate((
        (axes[0], "CV RMSE", "Test RMSE", "RMSE", r"RMSE ($\mu$g m$^{-3}$)"),
        (axes[1], "CV R2", "Test R2", r"$R^2$", r"$R^2$"),
    )):
        std = data.get(f"{cv_metric} Std", pd.Series(np.nan, index=data.index))
        axis.errorbar(
            x - 0.07, data[cv_metric], yerr=std,
            fmt="o", markersize=6,
            color=COLORS["primary"],
            ecolor=COLORS["primary"],
            elinewidth=1.2, capsize=3,
            label="Development CV mean ± SD",
            zorder=2,
        )
        axis.scatter(
            x + 0.07, data[test_metric],
            marker="D", s=52,
            color=COLORS["secondary"],
            edgecolor="white", linewidth=0.6,
            label="Common hold-out", zorder=3,
        )
        axis.set_xticks(
            x,
            [name.replace(" ", "\n") for name in data["Plot label"]],
            rotation=0,
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.text(-0.10, 1.02, f"({chr(97 + panel)})", transform=axis.transAxes,
                  fontweight="bold", ha="left", va="bottom")
        style_axes(axis, grid_axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.01), ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.5)
    _save(fig, output_path, dpi)
    return fig, axes


def plot_cv_candidate_comparison(summary, label_column, output_path, dpi=300):
    """Compare development-CV candidates without using hold-out results."""
    data = summary.sort_values("CV RMSE").reset_index(drop=True)
    x = np.arange(len(data))
    colors = [COLORS["primary"]] * len(data)
    colors[0] = COLORS["secondary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for panel, (axis, metric, std_metric, title, ylabel, direction) in enumerate((
        (axes[0], "CV RMSE", "CV RMSE Std", "RMSE", "CV RMSE (µg m$^{-3}$)", "Lower is better"),
        (axes[1], "CV R2", "CV R2 Std", "R²", "CV R²", "Higher is better"),
    )):
        axis.errorbar(x, data[metric], yerr=data[std_metric], fmt="none",
                      ecolor=COLORS["reference"], elinewidth=1.0, capsize=3)
        axis.scatter(x, data[metric], c=colors, s=58, edgecolor="white",
                     linewidth=0.6, zorder=3)
        axis.set_xticks(x, [str(value).replace("; ", "\n") for value in data[label_column]],
                        rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.text(0.99, 0.98, direction, transform=axis.transAxes,
                  ha="right", va="top", fontsize=8.5, color=COLORS["reference"])
        axis.text(-0.10, 1.02, f"({chr(97 + panel)})", transform=axis.transAxes,
                  fontweight="bold", ha="left", va="bottom")
        style_axes(axis, grid_axis="y")
    fig.tight_layout(w_pad=2.5)
    _save(fig, output_path, dpi)
    return fig, axes
