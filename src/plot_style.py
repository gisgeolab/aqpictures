"""Shared publication style for figures used in the thesis.

The PoliMi thesis template currently uses LaTeX's default Computer Modern
family.  STIX is used for Unicode text because Matplotlib's legacy ``cmr10``
font corrupts characters such as superscript two, delta, and underscores.
Mathematical expressions retain the Computer Modern math set.
"""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler


# Stable semantic colours used throughout Notebooks 01--05.
COLORS = {
    "primary": "#4C78A8",
    "primary_dark": "#355C7D",
    "primary_light": "#A9C5DF",
    "primary_pale": "#D5E3EF",
    "secondary": "#4C956C",
    "secondary_light": "#A8CDB7",
    "tertiary": "#8F77B5",
    "tertiary_light": "#C5B7D9",
    "accent": "#D55E00",
    "low": "#4C78A8",
    "moderate": "#E3A14A",
    "high": "#C44E52",
    "text": "#202020",
    "reference": "#4A4A4A",
    "grid": "#D9D9D9",
}

CONCENTRATION_COLORS = {
    "≤20": COLORS["low"],
    "20–35": COLORS["moderate"],
    ">35": COLORS["high"],
}

MODEL_FAMILY_COLORS = {
    "Conventional ML": COLORS["primary"],
    "Feature-based DNN": COLORS["tertiary"],
    "Pretrained CNN extension": COLORS["secondary"],
}

def apply_thesis_style() -> None:
    """Apply the shared Computer Modern publication theme globally."""

    # Reset stale notebook state before applying the publication defaults.
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "font.size": 11,
            "font.weight": "normal",
            "mathtext.fontset": "cm",
            "mathtext.default": "regular",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "axes.labelsize": 11,
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": "#B0B0B0",
            "axes.linewidth": 0.8,
            "axes.prop_cycle": cycler(
                color=[
                    COLORS["primary"],
                    COLORS["secondary"],
                    COLORS["tertiary"],
                    COLORS["accent"],
                    COLORS["high"],
                ]
            ),
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "legend.fontsize": 10,
            "legend.frameon": False,
            "figure.titlesize": 13,
            "figure.titleweight": "normal",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
        }
    )


def style_axes(ax, *, grid_axis: str = "y"):
    """Apply the common light-grid and open-frame treatment to one axis."""

    ax.grid(False)
    ax.grid(
        axis=grid_axis,
        color=COLORS["grid"],
        linestyle="--",
        linewidth=0.6,
        alpha=0.45,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
