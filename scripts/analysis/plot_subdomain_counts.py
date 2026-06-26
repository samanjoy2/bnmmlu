#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Subdomain Value Counts (Log Scale)

Reads a CSV (default: merged_all_questions_with_subdomains_renamed.csv),
counts the values of a subdomain column, and outputs a log-scale bar chart
with 45-degree rotated x-axis labels. Also writes the counts to a CSV.

Usage examples:
  python plot_subdomain_counts.py \
    --csv merged_all_questions_with_subdomains_renamed.csv \
    --column subdomain_name \
    --out subdomain_counts.png

  python plot_subdomain_counts.py --top 100 --dpi 200
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional


def detect_subdomain_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Return the column name to use for subdomain counts.
    Tries a preferred name, else a list of common candidates.
    Raises ValueError if none is found.
    """
    if preferred and preferred in df.columns:
        return preferred
    candidates = [
        "subdomain_name",
        "subdomain",
        "Subdomain",
        "sub_domain",
        "subDomain",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a subdomain column. Tried: {[preferred] + candidates}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Count subdomain values and plot log-scale bar chart"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="merged_all_questions_with_subdomains_renamed.csv",
        help="Path to input CSV (ignored if --counts-csv is provided)",
    )
    parser.add_argument(
        "--csv2",
        type=str,
        default=None,
        help="Optional second CSV to compare side-by-side (ignored if --counts-csv2 is provided)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Column name to use for subdomains (auto-detect if omitted)",
    )
    parser.add_argument(
        "--column2",
        type=str,
        default=None,
        help="Column for second CSV (auto-detect if omitted)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default auto-generated)",
    )
    parser.add_argument(
        "--counts-out",
        type=str,
        default="subdomain_value_counts.csv",
        help="Where to save the counts CSV",
    )
    parser.add_argument(
        "--counts-out2",
        type=str,
        default="subdomain_value_counts_2.csv",
        help="Where to save the second counts CSV (if --csv2 provided)",
    )
    parser.add_argument(
        "--combined-counts-out",
        type=str,
        default="subdomain_value_counts_combined.csv",
        help="Where to save combined counts when comparing two CSVs",
    )
    parser.add_argument(
        "--counts-csv",
        type=str,
        default=None,
        help="Use a precomputed counts CSV for dataset 1 (expects columns: subdomain/subdomain_name and count/n_questions)",
    )
    parser.add_argument(
        "--counts-csv2",
        type=str,
        default=None,
        help="Use a precomputed counts CSV for dataset 2",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=-1,
        help="Plot only the top-N subdomains (all if -1)",
    )
    parser.add_argument(
        "--figw",
        type=float,
        default=28.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--figh",
        type=float,
        default=10.0,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=45.0,
        help="X-axis label rotation in degrees",
    )
    parser.add_argument(
        "--ylabel-size",
        type=float,
        default=18.0,
        help="Font size for Y-axis label",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=12.0,
        help="Font size for tick labels",
    )
    parser.add_argument(
        "--title-size",
        type=float,
        default=16.0,
        help="Font size for the plot title",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=0.6,
        help="Bar width (0-1). Higher reduces gaps between bars",
    )
    parser.add_argument(
        "--label1",
        type=str,
        default="Dataset 1",
        help="Legend label for the first CSV",
    )
    parser.add_argument(
        "--label2",
        type=str,
        default="Dataset 2",
        help="Legend label for the second CSV",
    )
    args = parser.parse_args()

    # Load or compute counts for dataset 1
    if args.counts_csv:
        if not os.path.exists(args.counts_csv):
            raise FileNotFoundError(f"Counts CSV not found: {args.counts_csv}")
        cdf = pd.read_csv(args.counts_csv, encoding="utf-8")
        # Flexible column detection
        name_col = None
        for cand in ["subdomain", "subdomain_name", "Subdomain", "name", "category"]:
            if cand in cdf.columns:
                name_col = cand
                break
        count_col = None
        for cand in ["count", "n_questions", "n", "freq", "value", "values"]:
            if cand in cdf.columns:
                count_col = cand
                break
        if not name_col or not count_col:
            raise ValueError("Counts CSV must include subdomain name and count columns")
        counts = (
            cdf[[name_col, count_col]]
            .dropna()
            .groupby(name_col)[count_col]
            .sum()
            .sort_values(ascending=False)
        )
        col = name_col
    else:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"Input CSV not found: {args.csv}")
        df = pd.read_csv(args.csv, encoding="utf-8")
        col = detect_subdomain_column(df, args.column)

        # Clean and count
        series = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA})
            .dropna()
        )
        counts = series.value_counts().sort_values(ascending=False)

    counts2 = None
    if args.counts_csv2:
        if not os.path.exists(args.counts_csv2):
            raise FileNotFoundError(f"Second counts CSV not found: {args.counts_csv2}")
        cdf2 = pd.read_csv(args.counts_csv2, encoding="utf-8")
        name_col2 = None
        for cand in ["subdomain", "subdomain_name", "Subdomain", "name", "category"]:
            if cand in cdf2.columns:
                name_col2 = cand
                break
        count_col2 = None
        for cand in ["count", "n_questions", "n", "freq", "value", "values"]:
            if cand in cdf2.columns:
                count_col2 = cand
                break
        if not name_col2 or not count_col2:
            raise ValueError("Second counts CSV must include subdomain name and count columns")
        counts2 = (
            cdf2[[name_col2, count_col2]]
            .dropna()
            .groupby(name_col2)[count_col2]
            .sum()
            .sort_values(ascending=False)
        )
    elif args.csv2:
        if not os.path.exists(args.csv2):
            raise FileNotFoundError(f"Second CSV not found: {args.csv2}")
        df2 = pd.read_csv(args.csv2, encoding="utf-8")
        col2 = detect_subdomain_column(df2, args.column2)
        series2 = (
            df2[col2].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA}).dropna()
        )
        counts2 = series2.value_counts().sort_values(ascending=False)

    # Save counts to CSV(s)
    counts.to_csv(args.counts_out, header=["count"])
    if counts2 is not None:
        counts2.to_csv(args.counts_out2, header=["count"])
        # Combined CSV (union of subdomains)
        combined = pd.DataFrame({args.label1: counts, args.label2: counts2}).fillna(0).astype(int)
        combined.index.name = "subdomain"
        combined.to_csv(args.combined_counts_out)

    # Optionally limit to top-N for plotting clarity
    if counts2 is None:
        plot_counts = counts if args.top is None or args.top < 0 else counts.head(args.top)
    else:
        # Use union of subdomains, ordered by first dataset unless top is set
        combined = pd.DataFrame({"c1": counts, "c2": counts2}).fillna(0)
        if args.top is not None and args.top > 0:
            # determine top by total across both
            combined["total"] = combined["c1"] + combined["c2"]
            combined = combined.sort_values("total", ascending=False).head(args.top)
        else:
            combined = combined.sort_values("c1", ascending=False)
        plot_counts = combined

    # Figure output path
    if args.out:
        out_path = args.out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"subdomain_counts_log_{ts}.png"

    # Plot
    plt.figure(figsize=(args.figw, args.figh), dpi=args.dpi)
    ax = plt.gca()
    if counts2 is None:
        x = plot_counts.index.tolist()
        y = plot_counts.values.tolist()
        # Bars and edges; width controls gaps
        ax.bar(x, y, color="#1BA39C", edgecolor="#0e6f6a", linewidth=0.5, width=args.bar_width)
    else:
        cats = plot_counts.index.tolist()
        y1 = plot_counts["c1"].to_numpy()
        y2 = plot_counts["c2"].to_numpy()
        # numeric x positions for side-by-side bars
        pos = np.arange(len(cats), dtype=float)
        group_w = min(max(args.bar_width, 0.1), 0.95)
        bar_w = group_w / 2.0 * 0.92  # small inner gap
        ax.bar(pos - bar_w / 2.0, y1, width=bar_w, color="#1BA39C", edgecolor="#0e6f6a", linewidth=0.5, label=args.label1)
        ax.bar(pos + bar_w / 2.0, y2, width=bar_w, color="#2C82C9", edgecolor="#1f5a8a", linewidth=0.5, label=args.label2)
        ax.set_xticks(pos, cats)
        ax.legend(frameon=False)
    ax.set_yscale("log")

    ax.set_ylabel("# Instances (Log-Scale)", fontsize=args.ylabel_size)
    ax.set_xlabel("")
    # No mention of total questions in the figure
    ax.set_title("Subdomain Distribution", fontsize=args.title_size)

    # More dotted gridlines on log scale (major + many minor ticks)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), numticks=100))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # Stronger, more visible dotted lines
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=1.0, color="#777", alpha=0.9)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.7, color="#999", alpha=0.7)
    # Vertical dotted lines behind bars at category centers
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.6, color="#bbb", alpha=0.6)

    # Rotate x labels and increase tick font size
    plt.xticks(rotation=args.rotate, ha="right", fontsize=args.tick_size)
    ax.tick_params(axis="y", labelsize=args.tick_size)

    # Minimize outer margins (reduces empty space at plot ends)
    ax.margins(x=0.005)

    # Draw vertical dotted guide lines over the bars at category centers (more visible)
    ymin, ymax = ax.get_ylim()
    for xc in ax.get_xticks():
        ax.vlines(xc, ymin, ymax, linestyles=":", colors="#9a9a9a", linewidth=0.6, alpha=0.7, zorder=3)

    # Invisible count labels on top of bars (alpha close to 0 to keep selectable text in PDF)
    try:
        inv_alpha = max(0.0, min(1.0, getattr(args, 'inv_alpha', 0.03)))
        # Find bar containers if present
        for cont in ax.containers:
            # Only annotate bar containers
            if hasattr(cont, 'datavalues') or len(getattr(cont, 'patches', [])) > 0:
                for rect in getattr(cont, 'patches', []):
                    h = rect.get_height()
                    if h > 0:
                        ax.text(
                            rect.get_x() + rect.get_width() / 2.0,
                            h * 1.02,
                            f"{int(h)}",
                            ha="center",
                            va="bottom",
                            fontsize=max(8, int(args.tick_size * 0.8)),
                            alpha=inv_alpha,
                            color="#000000",
                        )
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_path)
    # Also write a vector PDF with selectable text
    base, _ = os.path.splitext(out_path)
    pdf_path = args.pdf_out if getattr(args, 'pdf_out', None) else base + ".pdf"
    try:
        plt.savefig(pdf_path, format='pdf')
    except Exception:
        pass

    print(f"Saved bar chart: {out_path}")
    print(f"Saved counts CSV: {args.counts_out}")
    print(f"Saved PDF chart: {pdf_path}")
    print(f"Column used: {col}")
    print(f"Unique subdomains: {len(counts):,}")


if __name__ == "__main__":
    main()
