"""Figures made by the analysis tools: one QuantAccelerator style, saved as PNG (for the vision model) and SVG (for notebooks and
the page) under <run_dir>/figures/<run_id>.{png,svg}.

Uses matplotlib with the Agg backend (no display needed).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from quantaccelerator.tools.registry import CTX  # noqa: E402

# the page's categorical slots (validated for CVD), a neutral ink and one alert colour
BLUE, ORANGE, GREEN, AMBER = "#3987e5", "#d95926", "#199e70", "#c98500"
SERIES = [BLUE, ORANGE, GREEN, AMBER, "#7a5cd6", "#c2447a"]
INK, MUTED, GRID, ALERT = "#1f2937", "#6b7280", "#e5e7eb", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 1.6,
})


def new(nrows: int = 1, ncols: int = 1, *, width: float = 7.5, height: float = 2.6, sharex: bool = False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height * nrows), sharex=sharex, squeeze=False)
    return fig, axes.ravel()


def date_axis(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def mark(ax, x, label: str | None = None, color: str = ALERT):
    """Mark a finding (e.g. a candidate break). Marks are left out of the plain copy the vision model reads."""
    marks = ax.figure.__dict__.setdefault("_qa_marks", [])
    marks.append(ax.axvline(x, color=color, linewidth=1, linestyle="--"))
    if label:
        n = getattr(ax, "_qa_nlabels", 0)  # stagger labels so nearby marks stay readable
        ax._qa_nlabels = n + 1
        marks.append(ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(3, -10 - 11 * (n % 3)),
                    textcoords="offset points", fontsize=8, color=color,
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8}))


def save(fig, title: str | None = None) -> str:
    """Save under the current run's figures/ dir named by the tool call's run_id; return the project-relative path."""
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    run_dir = CTX.run_dir or Path("runs/_adhoc")
    out = Path(run_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = CTX.current_run_id or "figure"
    fig.savefig(out / f"{stem}.png", dpi=110)
    fig.savefig(out / f"{stem}.svg")
    for artist in fig.__dict__.get("_qa_marks", []):  # plain copy: the data only, none of our own findings
        artist.remove()
    fig.savefig(out / f"{stem}.plain.png", dpi=110)
    plt.close(fig)
    return f"figures/{stem}.png"
