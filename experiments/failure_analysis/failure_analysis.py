import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fer_rag.evaluation.analysis import (RETRIEVAL_CASES, TABLE_COLUMNS, TABLE_LEVELS, split_by_retrieval_case,
                                        table_override_rescue)
from fer_rag.evaluation.vis_results import validate_results

parser = argparse.ArgumentParser()
parser.add_argument("--rag_path", help="Path to the RAG predictions CSV",
                    default="/truenas/home/sdolev/FacialExpressionsDolev/experiments/generator_comparison/runs/llava_next_34b/rag_llava_next_34b_8704.csv")
parser.add_argument("--zs_path", help="Path to the zero-shot predictions CSV",
                    default="/truenas/home/sdolev/FacialExpressionsDolev/experiments/generator_comparison/runs/llava_next_34b/zero_shot_llava_next_34b_8730.csv")
args = parser.parse_args()

rag_df = validate_results(pd.read_csv(args.rag_path))
zs_df = validate_results(pd.read_csv(args.zs_path)) if args.zs_path else None

# x-axis names follow the thesis terms: medium cases = conflicting evidence, low = erroneous evidence
CASE_LABELS = {
    "High": "(both correct)",
    "Only_top1_correct": "(top-1 only)",
    "Only_top2_correct": "(top-2 only)",
    "Low": "(both wrong)",
}
# (x position, name): "Conflicting evidence" is written once, centred over its two cases
EVIDENCE_GROUPS = [(0, "Correct evidence"), (1.5, "Conflicting evidence"), (3, "Erroneous evidence")]
OVERRIDE_COLOR = "#e07b73"  # soft red: RAG breaks a correct zero-shot answer
RESCUE_COLOR = "#5b9bd5"    # soft blue: RAG fixes a wrong zero-shot answer
FIG_SIZE = (11, 6.2)  # inches; LaTeX shrinks it to \textwidth (~6.3 in), i.e. to ~57%
TEXT_SIZE = 14        # axis titles, x labels, legend: ~8 pt after shrinking
VALUE_SIZE = 12       # numbers above / under bars: ~7 pt after shrinking


def plot_override_rescue(df, zs_df, true_col="true_label", pred_col="prediction", id_col="file_path"):
    """Per retrieval case, plot:
    override = P(RAG wrong | ZS right, case) and rescue = P(RAG right | ZS wrong, case).
    Above each bar: rate and numerator count. Below each bar: the denominator count."""
    merged = df.merge(zs_df[[id_col, pred_col]], on=id_col, suffixes=("", "_zs"))
    rag_right = merged[pred_col] == merged[true_col]
    zs_right = merged[pred_col + "_zs"] == merged[true_col]

    series = [
        ("Override: RAG wrong, out of ZS-right samples", zs_right, ~rag_right, OVERRIDE_COLOR, -0.2),
        ("Rescue: RAG right, out of ZS-wrong samples", ~zs_right, rag_right, RESCUE_COLOR, 0.2),
    ]

    # fonts are sized so they stay readable after LaTeX shrinks the figure to \textwidth
    fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")
    x = np.arange(len(RETRIEVAL_CASES))

    for label, given, event, color, offset in series:
        for i, case in enumerate(RETRIEVAL_CASES):
            denominator = (given & (merged["retrieval_case"] == case)).sum()
            numerator = (given & event & (merged["retrieval_case"] == case)).sum()
            rate = 100 * numerator / denominator if denominator else 0

            # same-colour edge with round joins softens the bar corners
            ax.bar(x[i] + offset, rate, width=0.36, color=color, edgecolor=color, linewidth=2.4, joinstyle="round",
                   label=label if i == 0 else None)
            ax.annotate(f"{rate:.1f}%\n({numerator:,})", (x[i] + offset, rate), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_SIZE, linespacing=1.1)
            ax.annotate(f"{denominator:,}", (x[i] + offset, 0), xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top", fontsize=VALUE_SIZE, color="#444444")

    # x-axis text: evidence group names on one row, case details on the row below
    ax.set_xticks(x, [""] * len(x))
    for x_pos, name in EVIDENCE_GROUPS:
        ax.annotate(name, (x_pos, 0), xytext=(0, -24), textcoords="offset points", ha="center", va="top", fontsize=TEXT_SIZE)
    for i, case in enumerate(RETRIEVAL_CASES):
        ax.annotate(CASE_LABELS[case], (x[i], 0), xytext=(0, -42), textcoords="offset points", ha="center", va="top", fontsize=TEXT_SIZE)

    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=TEXT_SIZE)
    ax.set_xlim(-0.65, 3.65)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("rate (%)", fontweight="bold", fontsize=TEXT_SIZE)
    ax.set_xlabel("Retrieval case", fontweight="bold", fontsize=TEXT_SIZE, labelpad=64)
    ax.grid(axis="y", color="#e5e5e5", linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=TEXT_SIZE,
              handlelength=1.5, columnspacing=1.5)

    # save only pdf
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "Override and rescue by retrieval case.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)

    print(f"Saved: {out_path}")
    plt.close(fig)


def table_within_retrieval_case(df, zs_df, true_col="true_label", pred_col="prediction", id_col="file_path"):
    """Same table as table_override_rescue, but the four ZS / RAG right-wrong combinations are
    percentages out of the retrieval case's own samples (e.g. ZS right & RAG right in Both correct / Both correct).
    'Share of test set' stays out of the whole test set."""
    merged = df.merge(zs_df[[id_col, pred_col]], on=id_col, suffixes=("", "_zs"))
    rag_right = merged[pred_col] == merged[true_col]
    zs_right = merged[pred_col + "_zs"] == merged[true_col]
    total = len(merged)

    def share_cell(mask, denominator):
        return f"{100 * mask.sum() / denominator:.1f}% ({mask.sum():,})" if denominator else "–"

    rows = []
    for level, in_level in [(TABLE_LEVELS[case], merged["retrieval_case"] == case) for case in RETRIEVAL_CASES] + \
                           [("All", pd.Series(True, index=merged.index))]:
        level_size = in_level.sum()
        rows.append([
            level,
            share_cell(in_level, total),
            share_cell(in_level & zs_right & rag_right, level_size),
            share_cell(in_level & zs_right & ~rag_right, level_size),
            share_cell(in_level & ~zs_right & rag_right, level_size),
            share_cell(in_level & ~zs_right & ~rag_right, level_size),
        ])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=TABLE_COLUMNS, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(TEXT_SIZE)
    table.auto_set_column_width(col=list(range(len(TABLE_COLUMNS))))
    table.scale(1, 2.4)
    for (row, col), cell in table.get_celld().items():
        # only horizontal lines between rows, like the reference table
        cell.visible_edges = "BT" if row == 0 else "B"
        cell.set_edgecolor("#d9d9d9")
        cell.PAD = 0.08

    # save only pdf
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "percentages out of retrieval case.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)

    print(f"Saved: {out_path}")
    plt.close(fig)


df = split_by_retrieval_case(rag_df)

if zs_df is not None:
    plot_override_rescue(df, zs_df)
    table_override_rescue(df, zs_df, figures_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    table_within_retrieval_case(df, zs_df)


