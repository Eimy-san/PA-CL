"""Shared matplotlib styling for all PA-CL paper figures.

Centralises colours, line styles, markers and figure sizes so that every
figure in the paper uses a consistent visual language.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIG_WIDTH_1COL = 3.35
FIG_WIDTH_2COL = 6.9
BAND_ALPHA = 0.15

_PALETTE = {
    "erm": "#6B7280",
    "cbp": "#F59E0B",
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
}

_MARKERS = {
    "erm": "o", "cbp": "s", "ewc": "s", "agem": "D", "er": "o",
    "derpp": "s", "er_ace": "H", "cls_er": "p", "xder": "X",
    "pacl": "*", "pacl_er": "o", "pacl_derpp": "s", "pacl_er_ace": "H",
    "pacl_cls_er": "p", "pacl_xder": "X",
}

_LINESTYLES = {
    "erm": "--", "cbp": "-", "ewc": ":", "agem": "--", "er": "-",
    "derpp": "-", "er_ace": "-", "cls_er": "-", "xder": "-",
    "pacl": "-", "pacl_er": "--", "pacl_derpp": "--", "pacl_er_ace": "--",
    "pacl_cls_er": "--", "pacl_xder": "--",
}

_LINewidths = {
    "erm": 1.2, "cbp": 1.2, "ewc": 1.2, "agem": 1.2, "er": 1.4,
    "derpp": 1.4, "er_ace": 1.4, "cls_er": 1.4, "xder": 1.4,
    "pacl": 1.8, "pacl_er": 1.6, "pacl_derpp": 1.6, "pacl_er_ace": 1.6,
    "pacl_cls_er": 1.6, "pacl_xder": 1.6,
}

_ZORDER = {
    "erm": 1, "cbp": 1, "ewc": 1, "agem": 1, "er": 2, "derpp": 2,
    "er_ace": 2, "cls_er": 2, "xder": 2, "pacl": 3, "pacl_er": 3,
    "pacl_derpp": 3, "pacl_er_ace": 3, "pacl_cls_er": 3, "pacl_xder": 3,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.fallback_font_format": "pdf",
            "font.family": "sans-serif",
            "font.sans-serif": ["Times", "Arial", "DejaVu Sans"],
            "text.usetex": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.tick_label_size": 7.5,
            "tick.fontsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "tick.direction": "in",
            "tick.style": "academic",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.dpi": 300,
            "dpi": 300,
            "figure.dpi": 300,
            "pdf.compression": True,
        }
    )


def get_color(method: str) -> str:
    return _PALETTE.get(method, "#6B7280")


def get_marker(method: str) -> str:
    return _MARKERS.get(method, "o")


def get_linestyle(method: str) -> str:
    return _LINESTYLES.get(method, "-")


def get_linewidth(method: str) -> float:
    return _LINewidths.get(method, 1.2)


def get_zorder(method: str) -> int:
    return _ZORDER.get(method, 1)


def get_label(method: str) -> str:
    labels = {
        "erm": "ERM", "cbp": "Cont. BP", "ewc": "EWC", "agem": "A-GEM",
        "er": "ER", "derpp": "DER++", "er_ace": "ER-ACE",
        "cls_er": "CLS-ER", "xder": "X-DER", "pacl": "PA-CL",
        "pacl_er": "PA-CL+ER", "pacl_derpp": "PA-CL+DER++",
        "pacl_er_ace": "PA-CL+ER-ACE", "pacl_cls_er": "PA-CL+CLS-ER",
        "pacl_xder": "PA-CL+X-DER",
    }
    return labels.get(method, method)


def method_order(methods: list[str]) -> list[str]:
    order = ["erm", "cbp", "ewc", "agem", "er", "derpp", "er_ace", "cls_er",
             "xder", "pacl", "pacl_er", "pacl_derpp", "pacl_er_ace",
             "pacl_cls_er", "pacl_xder"]
    return [m for m in order if m in methods]


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
