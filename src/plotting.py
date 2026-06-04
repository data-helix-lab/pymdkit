"""
plotting.py
-----------
Centralised Matplotlib style and colour definitions for the MD toolkit.
Import `apply_style()` at the top of every notebook or analysis script
so all figures share a consistent look.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Optional


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

PALETTE = {
    "rmsd":   "#2E86AB",   # steel blue
    "rmsf":   "#27AE60",   # forest green
    "rg":     "#C0392B",   # crimson
    "helix":  "#C0392B",
    "sheet":  "#E67E22",   # orange
    "coil":   "#7F8C8D",   # grey
    "hbond":  "#8E44AD",   # purple
    "sasa":   "#16A085",   # teal
    "pc1":    "#2C3E50",
    "pc2":    "#E74C3C",
}

CMAPS = {
    "fel":     "inferno",
    "fel_2d":  "jet",
    "scatter": "viridis",
    "contact": "Blues",
    "bfactor": "RdYlBu_r",
}


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

RC_BASE: dict = {
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.weight":          "bold",
    "axes.labelweight":     "bold",
    "axes.titleweight":     "bold",
    "axes.linewidth":       2.0,
    "xtick.major.width":    1.5,
    "ytick.major.width":    1.5,
    "xtick.minor.width":    1.0,
    "ytick.minor.width":    1.0,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "legend.framealpha":    0.85,
    "legend.edgecolor":     "0.8",
    "savefig.dpi":          300,
    "savefig.format":       "png",
    "savefig.bbox":         "tight",
    "figure.dpi":           150,
}


def apply_style(font_size: int = 13) -> None:
    """Apply global rcParams. Call once at the top of a notebook/script."""
    rc = RC_BASE.copy()
    rc.update(
        {
            "font.size":        font_size,
            "axes.labelsize":   font_size,
            "axes.titlesize":   font_size + 1,
            "xtick.labelsize":  font_size - 1,
            "ytick.labelsize":  font_size - 1,
            "legend.fontsize":  font_size - 1,
        }
    )
    plt.rcParams.update(rc)

    # Remove top / right spines globally via style context
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False

    print(f"[plotting] Style applied — Times New Roman, bold, {font_size}pt.")


# ---------------------------------------------------------------------------
# Shared helper: add colorbar to an existing axes
# ---------------------------------------------------------------------------

def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    cmap: str,
    vmin: float,
    vmax: float,
    label: str,
    orientation: str = "vertical",
) -> plt.colorbar:
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    return fig.colorbar(sm, ax=ax, orientation=orientation, label=label, pad=0.02)


# ---------------------------------------------------------------------------
# Shared helper: annotate figure with a watermark / metadata line
# ---------------------------------------------------------------------------

def add_caption(fig: plt.Figure, text: str, fontsize: int = 8) -> None:
    """Add a small grey caption at the bottom of a figure."""
    fig.text(
        0.5, -0.01, text,
        ha="center", va="top",
        fontsize=fontsize, color="0.55",
        style="italic",
    )
