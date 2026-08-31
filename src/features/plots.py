"""Publication plotting utilities for Notebook 3 feature evaluation."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.plot_style import COLORS, style_axes


FEATURE_LABELS = {
    "R_roi": "ROI red",
    "G_roi": "ROI green",
    "B_roi": "ROI blue",
    "R_std": "Red SD",
    "G_std": "Green SD",
    "B_std": "Blue SD",
    "S_mean": "Mean saturation",
    "V_mean": "Mean value",
    "colorfulness": "Colourfulness",
    "sky_brightness": "Sky brightness",
    "contrast": "Contrast",
    "gray_entropy": "Gray entropy",
    "laplacian_variance": "Laplacian variance",
    "mean_gradient_magnitude": "Mean gradient",
    "local_contrast": "Local contrast",
    "B_R_ratio": "Blue/red ratio",
    "dark_pixel_ratio": "Dark-pixel ratio",
    "sky_luminance_gradient": "Sky luminance gradient",
}


def _save_figure(fig, output_path, dpi=300):
    """Save a figure using the shared publication defaults."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")


def plot_pm25_distribution_and_temporal_coverage(
    data,
    target,
    output_path=None,
    time_column="time",
    thresholds=(20, 35),
    figure_size=(11.5, 4.3),
    dpi=300,
):
    """Combine the PM2.5 distribution and daily coverage in two panels."""
    daily = (
        data.set_index(time_column)[target]
        .resample("D")
        .agg(["mean", "min", "max", "count"])
    )
    daily.loc[daily["count"].eq(0), ["mean", "min", "max"]] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=figure_size)

    distribution_ax = axes[0]
    sns.histplot(
        data=data,
        x=target,
        bins=30,
        stat="density",
        color=COLORS["primary_light"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        ax=distribution_ax,
    )
    sns.kdeplot(
        data=data,
        x=target,
        color=COLORS["primary"],
        linewidth=2.0,
        ax=distribution_ax,
    )
    median = data[target].median()
    distribution_ax.axvline(
        median,
        color=COLORS["reference"],
        linestyle="-.",
        linewidth=1.1,
        label=f"Median = {median:.1f}",
    )
    distribution_ax.axvline(
        thresholds[1],
        color=COLORS["high"],
        linestyle=":",
        linewidth=1.2,
        label=rf"{thresholds[1]} $\mu$g m$^{{-3}}$",
    )
    distribution_ax.set_xlabel(
        r"PM$_{2.5}$ concentration ($\mu$g m$^{-3}$)"
    )
    distribution_ax.set_ylabel("Density")
    distribution_ax.legend(loc="upper right")
    style_axes(distribution_ax, grid_axis="y")
    distribution_ax.text(
        -0.12,
        1.03,
        "(a)",
        transform=distribution_ax.transAxes,
        fontweight="bold",
    )

    temporal_ax = axes[1]
    temporal_ax.fill_between(
        daily.index,
        daily["min"],
        daily["max"],
        color=COLORS["primary_light"],
        alpha=0.55,
        linewidth=0,
        label="Daily range",
    )
    temporal_ax.plot(
        daily.index,
        daily["mean"],
        color=COLORS["primary"],
        linewidth=1.2,
        label="Daily mean",
    )
    for threshold, color in zip(
        thresholds,
        (COLORS["moderate"], COLORS["high"]),
    ):
        temporal_ax.axhline(
            threshold,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=rf"{threshold} $\mu$g m$^{{-3}}$",
        )
    temporal_ax.set_xlabel("Date")
    temporal_ax.set_ylabel(
        r"PM$_{2.5}$ concentration ($\mu$g m$^{-3}$)"
    )
    temporal_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    temporal_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    temporal_ax.tick_params(axis="x", rotation=0)
    temporal_ax.legend(loc="upper right", ncol=2)
    style_axes(temporal_ax, grid_axis="y")
    temporal_ax.text(
        -0.12,
        1.03,
        "(b)",
        transform=temporal_ax.transAxes,
        fontweight="bold",
    )

    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, axes


def plot_monthly_pm25_distributions(
    data,
    target,
    output_path=None,
    month_column="year_month",
    figure_size=(9.5, 4.8),
    dpi=300,
):
    """Plot chronological monthly PM2.5 boxplots with sample counts."""
    month_order = data[month_column].drop_duplicates().tolist()
    month_counts = data.groupby(month_column).size()
    month_labels = [
        f"{month}\n(n={month_counts.loc[month]:,})" for month in month_order
    ]

    fig, ax = plt.subplots(figsize=figure_size)
    sns.boxplot(
        data=data,
        x=month_column,
        y=target,
        order=month_order,
        showfliers=False,
        width=0.62,
        color=COLORS["primary_light"],
        boxprops={"edgecolor": COLORS["reference"], "linewidth": 0.9},
        whiskerprops={"color": COLORS["reference"], "linewidth": 0.9},
        capprops={"color": COLORS["reference"], "linewidth": 0.9},
        medianprops={"color": COLORS["primary"], "linewidth": 1.6},
        ax=ax,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel(r"PM$_{2.5}$ concentration ($\mu$g m$^{-3}$)")
    ax.set_xticks(range(len(month_order)))
    ax.set_xticklabels(month_labels, rotation=45, ha="right")
    style_axes(ax, grid_axis="y")
    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, ax


def plot_pm25_distribution_and_monthly_distributions(
    data,
    target,
    output_path=None,
    month_column="year_month",
    figure_size=(12.5, 4.8),
    dpi=300,
):
    """Combine the overall and monthly PM2.5 distributions in two panels."""
    month_order = data[month_column].drop_duplicates().tolist()
    month_counts = data.groupby(month_column).size()
    month_labels = [
        f"{month}\n(n={month_counts.loc[month]:,})" for month in month_order
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size,
        gridspec_kw={"width_ratios": (0.8, 2.0)},
    )

    distribution_ax = axes[0]
    sns.histplot(
        data=data,
        x=target,
        bins=30,
        stat="density",
        color=COLORS["primary_light"],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        ax=distribution_ax,
    )
    sns.kdeplot(
        data=data,
        x=target,
        color=COLORS["primary"],
        linewidth=2.0,
        ax=distribution_ax,
    )
    median = data[target].median()
    distribution_ax.axvline(
        median,
        color=COLORS["reference"],
        linestyle="-.",
        linewidth=1.1,
        label=f"Median = {median:.1f}",
    )
    distribution_ax.set_xlabel(
        r"PM$_{2.5}$ concentration ($\mu$g m$^{-3}$)"
    )
    distribution_ax.set_ylabel("Density")
    distribution_ax.legend(loc="upper right")
    style_axes(distribution_ax, grid_axis="y")
    distribution_ax.text(
        -0.16,
        1.03,
        "(a)",
        transform=distribution_ax.transAxes,
        fontweight="bold",
    )

    monthly_ax = axes[1]
    sns.boxplot(
        data=data,
        x=month_column,
        y=target,
        order=month_order,
        showfliers=False,
        width=0.62,
        color=COLORS["primary_light"],
        boxprops={"edgecolor": COLORS["reference"], "linewidth": 0.9},
        whiskerprops={"color": COLORS["reference"], "linewidth": 0.9},
        capprops={"color": COLORS["reference"], "linewidth": 0.9},
        medianprops={"color": COLORS["primary"], "linewidth": 1.6},
        ax=monthly_ax,
    )
    monthly_ax.set_xlabel("Month")
    monthly_ax.set_ylabel(
        r"PM$_{2.5}$ concentration ($\mu$g m$^{-3}$)"
    )
    monthly_ax.set_xticks(range(len(month_order)))
    monthly_ax.set_xticklabels(month_labels, rotation=45, ha="right")
    style_axes(monthly_ax, grid_axis="y")
    monthly_ax.text(
        -0.08,
        1.03,
        "(b)",
        transform=monthly_ax.transAxes,
        fontweight="bold",
    )

    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, axes


def plot_standardized_feature_distributions(
    feature_data,
    feature_groups,
    output_path=None,
    figure_size=(10, 5.2),
    dpi=300,
):
    """Plot standardized ROI-feature distributions by semantic group."""

    feature_order = [
        feature
        for features in feature_groups.values()
        for feature in features
    ]
    missing = set(feature_order) - set(feature_data.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    standardized = (
        feature_data[feature_order] - feature_data[feature_order].mean()
    ) / feature_data[feature_order].std()

    label_order = [FEATURE_LABELS.get(feature, feature) for feature in feature_order]

    plot_data = standardized.melt(
        var_name="Feature",
        value_name="Standardized value",
    )
    plot_data["Label"] = plot_data["Feature"].map(FEATURE_LABELS).fillna(
        plot_data["Feature"]
    )
    group_lookup = {
        feature: group
        for group, features in feature_groups.items()
        for feature in features
    }
    plot_data["Group"] = plot_data["Feature"].map(group_lookup)

    feature_group_palette = {
        "Colour": COLORS["primary_light"],
        "Contrast and texture": COLORS["secondary_light"],
        "Haze and visibility": COLORS["tertiary_light"],
    }
    fig, ax = plt.subplots(figsize=figure_size)
    sns.boxplot(
        data=plot_data,
        x="Label",
        y="Standardized value",
        hue="Group",
        order=label_order,
        hue_order=list(feature_groups),
        palette=feature_group_palette,
        saturation=1,
        dodge=False,
        legend=False,
        showfliers=False,
        width=0.62,
        linewidth=0.8,
        boxprops={
            "edgecolor": COLORS["reference"],
            "linewidth": 0.8,
        },
        whiskerprops={
            "color": COLORS["reference"],
            "linewidth": 0.8,
        },
        capprops={
            "color": COLORS["reference"],
            "linewidth": 0.8,
        },
        medianprops={"color": COLORS["reference"], "linewidth": 1.2},
        ax=ax,
    )

    ax.axhline(
        0,
        color=COLORS["reference"],
        linewidth=1.0,
        linestyle="--",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Standardized feature value")
    ax.tick_params(axis="x", rotation=50)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    boundary = 0
    for group_index, (group, features) in enumerate(feature_groups.items()):
        start = boundary
        boundary += len(features)
        center = (start + boundary - 1) / 2
        ax.text(
            center,
            1.02,
            group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["text"],
        )
        if group_index < len(feature_groups) - 1:
            ax.axvline(
                boundary - 0.5,
                color=COLORS["grid"],
                linestyle=":",
                linewidth=1.0,
                zorder=0,
            )

    style_axes(ax, grid_axis="y")
    fig.tight_layout()

    _save_figure(fig, output_path, dpi)

    return fig, ax

def plot_correlation_comparison(
    correlation_results,
    output_path=None,
    figure_size=(9.2, 7.0),
    top_n=None,
    dpi=300,
):
    """Compare Pearson and Spearman correlations for all image features."""
    required = {"Feature", "Pearson r", "Spearman rho"}
    missing = required - set(correlation_results.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    feature_groups = {
        **{feature: "Colour and brightness" for feature in (
            "R_roi", "G_roi", "B_roi", "R_std", "G_std", "B_std",
            "S_mean", "V_mean", "colorfulness", "sky_brightness",
        )},
        **{feature: "Contrast and texture" for feature in (
            "contrast", "gray_entropy", "laplacian_variance",
            "mean_gradient_magnitude", "local_contrast",
        )},
        **{feature: "Haze and visibility" for feature in (
            "B_R_ratio", "dark_pixel_ratio", "sky_luminance_gradient",
        )},
    }
    group_colors = {
        "Colour and brightness": COLORS["primary"],
        "Contrast and texture": COLORS["secondary"],
        "Haze and visibility": COLORS["tertiary"],
    }

    ranking = (
        correlation_results
        .assign(
            rank_value=lambda data: (
                data["Pearson r"].abs() + data["Spearman rho"].abs()
            ) / 2
        )
        .sort_values("rank_value", ascending=False)
    )
    if top_n is not None:
        ranking = ranking.head(top_n)
    ranking = ranking.reset_index(drop=True)

    labels = ranking["Feature"].map(FEATURE_LABELS).fillna(ranking["Feature"])
    y = np.arange(len(ranking))

    bar_colors = [
        group_colors[feature_groups[feature]] for feature in ranking["Feature"]
    ]
    largest = ranking[["Pearson r", "Spearman rho"]].abs().to_numpy().max()
    axis_limit = max(0.6, np.ceil((largest + 0.04) * 10) / 10)

    fig, axes = plt.subplots(
        1, 2, figsize=figure_size, sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    panels = (
        (axes[0], "Pearson r", r"(a) Pearson $r$"),
        (axes[1], "Spearman rho", r"(b) Spearman $\rho$"),
    )

    for ax, column, title in panels:
        values = ranking[column].to_numpy()
        ax.barh(
            y, values, height=0.62, color=bar_colors, alpha=0.82,
            edgecolor="white", linewidth=0.5, zorder=2,
        )
        ax.axvline(0, color=COLORS["reference"], linewidth=0.9, zorder=3)
        for position, value in zip(y, values):
            ax.text(
                value + (0.012 if value >= 0 else -0.012),
                position,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8.2,
                color=COLORS["text"],
            )
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_xlabel("Correlation with PM$_{2.5}$")
        ax.set_title(title, loc="left", pad=8)
        style_axes(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("")

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=color, edgecolor="none", alpha=0.82, label=group)
        for group, color in group_colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.94, bottom=0.12)
    _save_figure(fig, output_path, dpi)
    return fig, axes


def plot_image_feature_group_cv(
    image_group_results,
    output_path=None,
    metric="CV RMSE",
    baseline_value=None,
    figure_size=(8.5, 5.0),
    dpi=300,
):
    """Compare image-feature groups with model-specific CV error bars."""
    std_column = f"{metric} Std"
    required = {"Model", "Feature Set", metric, std_column}
    missing = required - set(image_group_results.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    feature_order = [
        "Color",
        "Contrast",
        "Visibility",
        "All Image Features",
    ]
    feature_labels = [
        "Colour",
        "Contrast and\ntexture",
        "Haze and\nvisibility",
        "All image\nfeatures",
    ]
    model_colors = {
        "Ridge": COLORS["primary"],
        "Random Forest": COLORS["secondary"],
    }
    x = np.arange(len(feature_order))
    model_offsets = {
        "Ridge": -0.08,
        "Random Forest": 0.08,
    }

    fig, ax = plt.subplots(figsize=figure_size)
    for model_name in ("Ridge", "Random Forest"):
        model_results = (
            image_group_results.loc[
                image_group_results["Model"].eq(model_name)
            ]
            .set_index("Feature Set")
            .reindex(feature_order)
        )
        if model_results[metric].isna().any():
            raise ValueError(f"Incomplete results for {model_name}.")
        ax.errorbar(
            x + model_offsets[model_name],
            model_results[metric],
            yerr=model_results[std_column],
            color=model_colors[model_name],
            marker="o",
            markersize=6.5,
            linestyle="none",
            elinewidth=1.3,
            capsize=3,
            label=model_name,
        )

    if baseline_value is not None:
        ax.axhline(
            baseline_value,
            color=COLORS["reference"],
            linestyle="--",
            linewidth=1.1,
            label=f"Mean baseline = {baseline_value:.2f}",
            zorder=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels)
    ax.set_xlabel("")
    ax.set_ylabel(
        r"CV RMSE ($\mu$g m$^{-3}$)"
        if metric == "CV RMSE"
        else metric
    )
    ax.margins(y=0.08)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        borderaxespad=0,
    )
    style_axes(ax, grid_axis="y")

    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, ax


def plot_progressive_fusion_cv(
    fusion_results,
    output_path=None,
    figure_size=(11, 4.6),
    dpi=300,
):
    """Plot progressive-fusion CV RMSE and R2 for both analytical models."""
    metrics = ("CV RMSE", "CV R2")
    required = {
        "Model",
        "Feature Set",
        *metrics,
        *(f"{metric} Std" for metric in metrics),
    }
    missing = required - set(fusion_results.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    feature_order = [
        "Image",
        "Image + ERA5",
        "Image + ERA5 + ARPA",
        "All Features",
    ]
    feature_labels = [
        "Image only",
        "+ ERA5",
        "+ ARPA",
        "+ Temporal",
    ]
    model_colors = {
        "Ridge": COLORS["primary"],
        "Random Forest": COLORS["secondary"],
    }
    model_offsets = {
        "Ridge": -0.035,
        "Random Forest": 0.035,
    }
    x = np.arange(len(feature_order))

    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    for ax, metric in zip(axes, metrics):
        for model_name in ("Ridge", "Random Forest"):
            model_results = (
                fusion_results.loc[
                    fusion_results["Model"].eq(model_name)
                ]
                .set_index("Feature Set")
                .reindex(feature_order)
            )
            if model_results[metric].isna().any():
                raise ValueError(f"Incomplete fusion results for {model_name}.")
            ax.errorbar(
                x + model_offsets[model_name],
                model_results[metric],
                yerr=model_results[f"{metric} Std"],
                color=model_colors[model_name],
                marker="o",
                markersize=6.5,
                linewidth=1.5,
                elinewidth=1.2,
                capsize=3,
                label=model_name,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels)
        ax.set_xlabel("")
        style_axes(ax, grid_axis="y")

    axes[0].set_ylabel(r"CV RMSE ($\mu$g m$^{-3}$)")
    axes[1].set_ylabel(r"CV R$^2$")
    axes[0].text(
        -0.12, 1.03, "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
    )
    axes[1].text(
        -0.12, 1.03, "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
    )
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, axes


def plot_source_ablation_delta(
    ablation_fold_changes,
    output_path=None,
    figure_size=(6.8, 4.4),
    dpi=300,
):
    """Plot fold-level and mean RMSE changes after removing each source."""
    required = {
        "Removed Source",
        "Fold",
        "RMSE Change",
    }
    missing = required - set(ablation_fold_changes.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    plot_data = ablation_fold_changes.copy()
    preferred_order = ["Image", "ERA5", "Temporal", "ARPA"]
    available_sources = set(plot_data["Removed Source"])
    source_order = [
        source for source in preferred_order if source in available_sources
    ]
    source_order.extend(sorted(available_sources - set(source_order)))
    x_positions = {
        source: position for position, source in enumerate(source_order)
    }
    folds = sorted(plot_data["Fold"].unique())
    fold_offsets = dict(zip(folds, np.linspace(-0.12, 0.12, len(folds))))

    fig, ax = plt.subplots(figsize=figure_size)
    for source in source_order:
        source_data = plot_data.loc[
            plot_data["Removed Source"].eq(source)
        ]
        jittered_x = [
            x_positions[source] + fold_offsets[fold]
            for fold in source_data["Fold"]
        ]
        ax.scatter(
            jittered_x,
            source_data["RMSE Change"],
            color=COLORS["primary_light"],
            edgecolor=COLORS["primary"],
            linewidth=0.6,
            s=34,
            alpha=0.9,
            label="Individual fold" if source == source_order[0] else None,
            zorder=2,
        )
        ax.scatter(
            x_positions[source],
            source_data["RMSE Change"].mean(),
            color=COLORS["secondary"],
            marker="D",
            s=62,
            label="Fold mean" if source == source_order[0] else None,
            zorder=3,
        )
    ax.axhline(
        0,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xticks(range(len(source_order)))
    ax.set_xticklabels(source_order)
    ax.set_xlabel("")
    ax.set_ylabel(
        r"Change in CV RMSE after source removal ($\mu$g m$^{-3}$)"
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0,
    )
    style_axes(ax, grid_axis="y")

    fig.tight_layout()
    _save_figure(fig, output_path, dpi)
    return fig, ax
