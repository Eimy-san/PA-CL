"""Publication-quality matplotlib style for PA-CL paper figures.

Usage:
    from figstyle import apply_style, METHOD_STYLE, FIG_WIDTH_1COL, FIG_WIDTH_2COL
    apply_style()
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


FIG_WIDTH_1COL = 3.35
FIG_WIDTH_2COL = 6.9

_PALETTE = {
    "erm": "#6B7280",
    "ewc": "#7C3AED",
    "agem": "#D4A400",
    "er": "#2563EB",
    "derpp": "#DC2626",
    "er_ace": "#0891B2",
    "cls_er": "#BE185D",
    "xder": "#92400E",
    "pacl": "#059669",
    "pacl_er": "#2563EB",
    "pacl_derpp": "#DC2626",
    "pacl_er_ace": "#0891B2",
    "pacl_cls_er": "#BE185D",
    "pacl_xder": "#92400E",
    "cbp": "#F59E0B",
}

_LABELS = {
    "erm": "ERM",
    "ewc": "EWC",
    "agem": "A-GEM",
    "er": "ER",
    "derpp": "DER++",
    "er_ace": "ER-ACE",
    "cls_er": "CLS-ER",
    "xder": "X-DER",
    "pacl": "PA-CL",
    "pacl_er": "PA-CL+ER",
    "pacl_derpp": "PA-CL+DER++",
    "pacl_er_ace": "PA-CL+ER-ACE",
    "pacl_cls_er": "PA-CL+CLS-ER",
    "pacl_xder": "PA-CL+X-DER",
    "cbp": "Cont. BP",
}

_MARKERS = {
    "erm": "o",
    "ewc": "s",
    "agem": "D",
    "er": "^",
    "derpp": "v",
    "er_ace": "h",
    "cls_er": "p",
    "xder": "X",
    "pacl": "*",
    "pacl_er": "^",
    "pacl_derpp": "v",
    "pacl_er_ace": "h",
    "pacl_cls_er": "p",
    "pacl_xder": "X",
    "cbp": "P",
}

_LINESTYLES = {
    "erm": "-",
    "ewc": "-",
    "agem": "-",
    "er": "-",
    "derpp": "-",
    "er_ace": "-",
    "cls_er": "-",
    "xder": "-",
    "pacl": "-",
    "pacl_er": "--",
    "pacl_derpp": "--",
    "pacl_er_ace": "--",
    "pacl_cls_er": "--",
    "pacl_xder": "--",
    "cbp": "-",
}

_METHOD_ORDER = [
    "erm",
    "cbp",
    "ewc",
    "agem",
    "er",
    "derpp",
    "er_ace",
    "cls_er",
    "xder",
    "pacl",
    "pacl_er",
    "pacl_derpp",
    "pacl_er_ace",
    "pacl_cls_er",
    "pacl_xder",
]

Z_ORDER = {
    "erm": 2,
    "ewc": 2,
    "agem": 2,
    "er": 2,
    "derpp": 2,
    "er_ace": 2,
    "cls_er": 2,
    "xder": 2,
    "pacl": 3,
    "pacl_er": 3,
    "pacl_derpp": 3,
    "pacl_er_ace": 3,
    "pacl_cls_er": 3,
    "pacl_xder": 3,
    "cbp": 2,
}

LW_BASELINE = 1.6
LW_PACL = 2.2
MARKER_SIZE = 4.5
MARKER_SIZE_PACL = 6.0
BAND_ALPHA = 0.15
GRID_ALPHA = 0.25
GRID_LS = "--"
DPI = 300


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#D1D5DB",
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#374151",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": GRID_LS,
            "grid.linewidth": 0.5,
            "grid.color": "#9CA3AF",
            "lines.linewidth": LW_BASELINE,
            "lines.markersize": MARKER_SIZE,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 100,
            "figure.autolayout": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "cm",
        }
    )


def get_color(method: str) -> str:
    return _PALETTE.get(method, "#333333")


def get_label(method: str) -> str:
    return _LABELS.get(method, method.upper())


def get_marker(method: str) -> str:
    return _MARKERS.get(method, "o")


def get_linestyle(method: str) -> str:
    return _LINESTYLES.get(method, "-")


def get_linewidth(method: str) -> float:
    if method.startswith("pacl"):
        return LW_PACL
    return LW_BASELINE


def get_markersize(method: str) -> float:
    if method.startswith("pacl"):
        return MARKER_SIZE_PACL
    return MARKER_SIZE


def get_zorder(method: str) -> int:
    return Z_ORDER.get(method, 2)


def method_order(methods: list[str]) -> list[str]:
    return [m for m in _METHOD_ORDER if m in methods]


def place_legend_above(
    ax,
    n_entries: int,
    fontsize: float = 7.5,
    max_per_row: int = 7,
    title: "str | None" = None,
):
    """Place the legend fully above the axes, outside the plot area.

    The legend's bottom edge is anchored just above the axes top
    spine, so the box never dips into the plot region.  With
    savefig.bbox = 'tight' the legend is included in the saved
    figure.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    ncols = min(n_entries, max_per_row) if n_entries > 5 else n_entries
    ncols = max(1, min(ncols, len(handles)))
    kwargs = dict(
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        framealpha=0.85,
        edgecolor="#D1D5DB",
        fancybox=False,
        borderpad=0.0,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.75,
    )
    if title is not None:
        kwargs["title"] = title
        kwargs["title_fontsize"] = 9
        kwargs["frameon"] = False
    try:
        return ax.legend(handles, labels, ncols=ncols, **kwargs)
    except TypeError:
        return ax.legend(handles, labels, ncol=ncols, **kwargs)
