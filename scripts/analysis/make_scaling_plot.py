#!/usr/bin/env python3
"""
Make a scaling-style plot (ExaFLOP vs Average Performance) from usage averages.

This script reads a per-model averages CSV (the one produced by your
aggregate_usage_averages.py) and a metadata CSV that provides factual compute
inputs per model (parameter count and training tokens). It then plots
ExaFLOP = 6 * (#params in billions) * (#train tokens in billions)
on a log-scaled x‑axis against the average accuracy on the y‑axis.

The script intentionally ignores model families that do not have at least two
distinct parameter sizes (to avoid drawing single-point trends).

Inputs (CSV files):
  1) usage_csv: e.g.,
     "Usage Proper Documentation/llm_usage_averages_by_model_0-shot Direct (Non-Reasoning).csv"
     Columns expected (minimum):
       - model_name
       - accruaC Y AVERAGE   (floating accuracy in [0,1])
     If the accuracy column is absent, you may add it with your aggregators.

  2) metadata_csv: e.g., "Usage Proper Documentation/model_compute_metadata.csv"
     Columns (required):
       - model_name   (must match the model_name in the usage CSV exactly)
       - family       (e.g., "Qwen3", "Meta-Llama-3", "Gemma-3")
       - params_b     (model parameter count in billions, float)
       - train_tokens_b (training tokens in billions, float)

No assumptions are made by this script. Models that lack factual entries in the
metadata are ignored. Families with fewer than two matched sizes are ignored.

Usage:
  python make_scaling_plot.py \
      --usage-csv "Usage Proper Documentation/llm_usage_averages_by_model_0-shot Direct (Non-Reasoning).csv" \
      --metadata-csv "Usage Proper Documentation/model_compute_metadata.csv" \
      --out "Usage Proper Documentation/scaling_plot_0shot_direct_nonreasoning.png"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def read_usage(usage_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(usage_csv)
    if "model_name" not in df.columns:
        raise ValueError("usage_csv is missing required column 'model_name'.")
    # Accept a few accuracy header variants, prefer the aggregators' column
    acc_cols: List[str] = [
        "accruaC Y AVERAGE",  # from our aggregator updates
        "accuracy_avg",
        "avg_accuracy",
    ]
    acc_col = next((c for c in acc_cols if c in df.columns), None)
    if acc_col is None:
        raise ValueError(
            "usage_csv does not contain an accuracy average column. Add 'accruaC Y AVERAGE' via the aggregator."
        )
    out = df[["model_name", acc_col]].copy()
    out.rename(columns={acc_col: "accuracy"}, inplace=True)
    # Ensure accuracy is numeric and between 0 and 1; drop invalid
    out["accuracy"] = pd.to_numeric(out["accuracy"], errors="coerce")
    out = out.dropna(subset=["accuracy"])
    # If accuracy appears to be percentage > 1, scale it to [0,1]
    if out["accuracy"].max() > 1.0:
        out["accuracy"] = out["accuracy"].clip(0, 100) / 100.0
    return out


def read_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"model_name", "family", "params_b", "train_tokens_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata_csv missing columns: {', '.join(sorted(missing))}")
    # Enforce numeric types
    df["params_b"] = pd.to_numeric(df["params_b"], errors="coerce")
    df["train_tokens_b"] = pd.to_numeric(df["train_tokens_b"], errors="coerce")
    df = df.dropna(subset=["params_b", "train_tokens_b"]).copy()
    # ExaFLOP per Kaplan et al. (2020): 6 * params * tokens (both in billions)
    df["exaflop"] = 6.0 * df["params_b"] * df["train_tokens_b"]
    return df


def _save_all_formats(out_png: Path | None, out_pdf: Path | None, out_pdf_selectable: Path | None) -> None:
    # Save PNG (raster)
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        print(f"[done] Saved PNG: {out_png}")

    # Save standard PDF (vector). Keep default fonttype (usually 3)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"[done] Saved PDF: {out_pdf}")

    # Save selectable/searchable PDF with TrueType fonts (fonttype=42)
    if out_pdf_selectable is not None:
        out_pdf_selectable.parent.mkdir(parents=True, exist_ok=True)
        orig_pdf_fonttype = mpl.rcParams.get("pdf.fonttype", 3)
        orig_ps_fonttype = mpl.rcParams.get("ps.fonttype", 3)
        try:
            mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType, keeps text selectable/searchable
            mpl.rcParams["ps.fonttype"] = 42
            plt.tight_layout()
            plt.savefig(out_pdf_selectable, format="pdf", bbox_inches="tight")
            print(f"[done] Saved selectable PDF: {out_pdf_selectable}")
        finally:
            mpl.rcParams["pdf.fonttype"] = orig_pdf_fonttype
            mpl.rcParams["ps.fonttype"] = orig_ps_fonttype


def build_plot(df: pd.DataFrame, out_png: Path | None, out_pdf: Path | None, out_pdf_selectable: Path | None, title: str = "Scaling by Family") -> None:
    # Expect columns: model_name, family, exaflop, accuracy
    families = (
        df.groupby("family")["model_name"].count().reset_index(name="count").query("count >= 2")["family"].tolist()
    )
    df = df[df["family"].isin(families)].copy()
    if df.empty:
        raise SystemExit("No families with multiple parameter sizes after matching usage and metadata.")

    plt.figure(figsize=(8, 5))
    for fam, sub in df.sort_values(["family", "exaflop"]).groupby("family"):
        x = sub["exaflop"].values
        y = sub["accuracy"].values
        # Plot line with markers (log x-axis later)
        plt.plot(x, y, marker="o", linestyle="-", label=fam)

    plt.xscale("log")
    plt.xlabel("ExaFLOP (log scale)")
    plt.ylabel("Average Performance (accuracy)")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(title="Family", loc="best")
    _save_all_formats(out_png, out_pdf, out_pdf_selectable)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make ExaFLOP vs Accuracy scaling plot.")
    ap.add_argument("--usage-csv", type=Path, required=True, help="Path to usage averages CSV with accuracy column.")
    ap.add_argument("--metadata-csv", type=Path, required=True, help="Path to model compute metadata CSV.")
    ap.add_argument("--out", type=Path, required=False, default=None, help="Output PNG path.")
    ap.add_argument("--out-pdf", type=Path, required=False, default=None, help="Output PDF path (vector).")
    ap.add_argument(
        "--out-pdf-selectable",
        type=Path,
        required=False,
        default=None,
        help="Output PDF with selectable/searchable text (embeds TrueType fonts).",
    )
    ap.add_argument("--title", type=str, default="0-shot Direct (Non-Reasoning) – Scaling", help="Plot title")
    args = ap.parse_args()

    usage = read_usage(args.usage_csv)
    meta = read_metadata(args.metadata_csv)

    # Exact join on model_name to remain factual (no fuzzy mappings)
    merged = usage.merge(meta, on="model_name", how="inner")
    if merged.empty:
        raise SystemExit(
            "No overlapping models between usage and metadata. Ensure 'model_name' values match exactly."
        )

    build_plot(
        merged[["model_name", "family", "exaflop", "accuracy"]],
        args.out,
        args.out_pdf,
        args.out_pdf_selectable,
        title=args.title,
    )


if __name__ == "__main__":
    main()
