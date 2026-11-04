"""Pronoun-prediction figures: stacked bars of the probabilities of she, he and they for the most gendered occupations.

The pronoun the occupations are sorted by is the solid bottom segment; the two others are stacked above it in a
lighter tint. One panel per pronoun.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import VERSIONS  # noqa: E402
from .pronoun_prediction import PRONOUNS  # noqa: E402

COLORS = {"male": "#3f9b6b", "female": "#3a76c4", "neutral": "#a3785a"}  # the thesis' green, blue, brown
WORDS = {"en": {"male": "he", "female": "she", "neutral": "they"}, "zh": {"male": "他", "female": "她", "neutral": "TA"}}
PANEL_TITLE = {"male": "Masculine", "female": "Feminine", "neutral": "Neutral"}
LANGUAGE_NAME = {"en": "English", "zh": "Chinese"}
VERSION_NAME = {"base": "base", "finetuned": "finetuned", "adapter": "adapter"}
INK, MUTED, GRID = "#22252b", "#6b7280", "#e5e7eb"
CJK_FONTS = ["Hiragino Sans GB", "PingFang SC", "PingFang HK", "Heiti SC", "Heiti TC", "STHeiti", "Songti SC",
             "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]


def setup():
    """A clean style; Latin text in Helvetica/Arial, Chinese in the first CJK font installed."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    latin = [f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed]
    cjk = [f for f in CJK_FONTS if f in installed]
    plt.rcParams.update({
        "font.family": latin + cjk + ["DejaVu Sans"], "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": MUTED, "ytick.left": False, "xtick.major.size": 0,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return cjk


def tint(color, alpha):
    r, g, b, _ = to_rgba(color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def stacked(ax, x, rows, pronoun, width=0.78, labels=True):
    """One stacked bar per row: `pronoun` at the bottom (solid), the two others above (light)."""
    order = [pronoun] + [p for p in PRONOUNS if p != pronoun]
    bottom = [0.0] * len(rows)
    for i, p in enumerate(order):
        values = (rows[f"{p}_prob"] * 100).tolist()
        ax.bar(x, values, width, bottom=bottom, color=COLORS[p] if i == 0 else tint(COLORS[p], 0.38),
               edgecolor="white", linewidth=0.8, zorder=2)
        if labels:
            for xi, b, v in zip(x, bottom, values):
                if v >= (3.5 if i == 0 else 6):
                    ax.text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center", zorder=3,
                            fontsize=8 if i == 0 else 7, fontweight="bold" if i == 0 else "normal",
                            color="white" if i == 0 else INK)
        bottom = [b + v for b, v in zip(bottom, values)]
    return bottom


def legend(fig, language):
    """Top right, on the line of the title."""
    handles = [Patch(color=COLORS[p], label=WORDS[language][p]) for p in PRONOUNS]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=3, frameon=False,
               fontsize=10.5, handlelength=1.1, handleheight=1.1, columnspacing=1.4)


def header(fig, title, subtitle):
    fig.text(0.02, 0.975, title, fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.915, subtitle, fontsize=10, color=MUTED, ha="left", va="top")


def base_model_figure(rows, scores, language, model, n, path):
    """Top-n occupations per pronoun of one base model."""
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharey=True)
    top_of_bars = []
    for ax, pronoun in zip(axes, PRONOUNS):
        part = rows[rows["pronoun"] == pronoun]
        top_of_bars += stacked(ax, range(len(part)), part, pronoun)
        ax.set_xticks(range(len(part)), part["label"], rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{PANEL_TITLE[pronoun]}  ·  bias {scores[pronoun]:.2f}", fontsize=11, loc="left",
                     color=COLORS[pronoun], fontweight="bold")
    axes[0].set_ylabel("Probability (%)", fontsize=10)
    axes[0].set_ylim(0, max(top_of_bars) * 1.08)
    header(fig, f"Gendered occupations according to {model}",
           f"Top {n} occupations per pronoun · {LANGUAGE_NAME[language]} alternate single-entity sentences · "
           f"bias over all {3 * n}: {scores['all']:.2f}")
    legend(fig, language)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def versions_figure(rows, language, model, n, path):
    """Top-n occupations per pronoun of base, finetuned and adapter."""
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True)
    gap = 0.9
    top_of_bars = []
    for ax, pronoun in zip(axes, PRONOUNS):
        ticks, labels = [], []
        for g, version in enumerate(VERSIONS):
            part = rows[(rows["pronoun"] == pronoun) & (rows["version"] == version)]
            x = [g * (n + gap) + i for i in range(len(part))]
            top_of_bars += stacked(ax, x, part, pronoun, labels=False)
            ticks += x
            labels += part["label"].tolist()
            ax.text(sum(x) / len(x), -0.02, VERSION_NAME[version],
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=MUTED)
            if g:
                ax.axvline(g * (n + gap) - (gap + 1) / 2, color=GRID, linewidth=1, zorder=1)
        ax.set_xticks(ticks, labels, rotation=55, ha="right", fontsize=8.5)
        ax.tick_params(axis="x", pad=20)
        ax.set_title(PANEL_TITLE[pronoun], fontsize=11, loc="left", color=COLORS[pronoun], fontweight="bold")
    axes[0].set_ylabel("Probability (%)", fontsize=10)
    axes[0].set_ylim(0, max(top_of_bars) * 1.08)
    header(fig, f"Gendered occupations according to {model}: base, finetuned and adapter",
           f"Top {n} occupations per pronoun and version · {LANGUAGE_NAME[language]} alternate single-entity "
           f"sentences")
    legend(fig, language)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)
