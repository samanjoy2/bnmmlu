#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a KMMLU-style grid figure from a questions CSV.

- Lets you pick row/column fields (e.g., subject vs. subdomain_name)
- Allows optional explicit row/column ordering via CLI (JSON list)
- Samples one example question per cell
- Renders a compact table to PNG/PDF using matplotlib

Example usage (PowerShell):
  python export_kmmlu_style_grid.py \
    --csv finished/verified_dataset_overlap_with_hard_subset.csv \
    --row-field subject \
    --col-field subdomain_name \
    --row-order '["STEM","Applied Science","HUMSS","Other"]' \
    --title "Required Type of Knowledge"

If you don't supply orders, the script chooses the most frequent values
up to --max-rows / --max-cols.
"""

import argparse
import json
import os
import random
import textwrap
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


def parse_order_list(order_json: Optional[str]) -> Optional[List[str]]:
    if not order_json:
        return None
    try:
        data = json.loads(order_json)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return None


def choose_values_by_frequency(series: pd.Series, limit: int) -> List[str]:
    value_counts = series.astype(str).value_counts()
    return [v for v in value_counts.index[: max(0, limit)]]


def sample_question_cell(df: pd.DataFrame, question_field: str) -> Optional[str]:
    if df.empty:
        return None
    # Prefer a deterministic but varied sample: first try a mid index
    try:
        idx = len(df) // 2
        q = str(df.iloc[idx][question_field])
        if q:
            return q
    except Exception:
        pass
    # Fallback: random
    try:
        row = df.sample(1, random_state=random.randint(0, 1_000_000)).iloc[0]
        return str(row[question_field])
    except Exception:
        return None


def wrap_text(text: str, width: int) -> str:
    try:
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))
    except Exception:
        return text


def build_grid_cells(
    df: pd.DataFrame,
    row_field: str,
    col_field: str,
    row_values: List[str],
    col_values: List[str],
    question_field: str = "question",
    cell_wrap: int = 36,
    mode: str = "sample",  # sample | count
    per_cell: int = 1,
) -> List[List[str]]:
    # Header row: first empty cell + column headers
    grid: List[List[str]] = []
    header_row = ["Category"] + [str(c) for c in col_values]
    grid.append(header_row)

    for r in row_values:
        row_label = str(r)
        row_cells: List[str] = [row_label]
        for c in col_values:
            sub = df[(df[row_field].astype(str) == str(r)) & (df[col_field].astype(str) == str(c))]
            if mode == "count":
                content = f"{len(sub)} items" if len(sub) else "-"
            else:
                # sample mode: up to per_cell examples
                if sub.empty:
                    content = "-"
                else:
                    samples = []
                    try:
                        if per_cell <= 1:
                            q = sample_question_cell(sub, question_field)
                            samples = [q] if q else []
                        else:
                            sampled = sub.sample(min(per_cell, len(sub)), random_state=42)
                            samples = [str(v) for v in sampled[question_field].tolist()]
                    except Exception:
                        q = sample_question_cell(sub, question_field)
                        samples = [q] if q else []
                    # bullet list
                    bullet_lines = [f"• {s}" for s in samples if s]
                    content = "\n".join(bullet_lines) if bullet_lines else "-"
            content = wrap_text(content, width=cell_wrap)
            row_cells.append(content)
        grid.append(row_cells)
    return grid


def render_table_image(
    grid: List[List[str]],
    title: str,
    out_png: Optional[str],
    out_pdf: Optional[str],
    font_name: Optional[str] = None,
    font_size: int = 9,
    header_facecolor: str = "#e8e8e8",
    row_header_facecolor: str = "#f5f5f5",
    cell_facecolor: str = "#ffffff",
    edge_color: str = "#cccccc",
    figsize: Optional[Tuple[float, float]] = None,
    table_scale: Tuple[float, float] = (1.1, 1.3),
):
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Empty grid")

    # Estimate figure size if not provided
    if figsize is None:
        # Wider if many columns; taller if many rows
        width = max(8.0, min(22.0, 1.8 * n_cols))
        height = max(4.5, min(20.0, 0.9 * n_rows))
        figsize = (width, height)

    plt.rcParams["font.size"] = font_size
    if font_name:
        plt.rcParams["font.family"] = font_name

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Build cell text matrix for matplotlib.table
    cell_text = grid

    table = ax.table(
        cellText=cell_text,
        cellLoc='left',
        loc='center',
    )

    # Style headers
    for col in range(n_cols):
        hdr = table[(0, col)]
        hdr.set_facecolor(header_facecolor)
        hdr.set_edgecolor(edge_color)
        hdr.set_height(0.06)
        hdr.set_fontsize(font_size + 1)
        hdr.set_text_props(weight='bold')

    for row in range(1, n_rows):
        # Row header cell (first column)
        rh = table[(row, 0)]
        rh.set_facecolor(row_header_facecolor)
        rh.set_edgecolor(edge_color)
        rh.set_height(0.07)
        rh.set_fontsize(font_size)
        rh.set_text_props(weight='bold')
        rh._loc = 'left'

        for col in range(1, n_cols):
            cell = table[(row, col)]
            cell.set_facecolor(cell_facecolor)
            cell.set_edgecolor(edge_color)
            cell.set_height(0.18)
            cell.set_fontsize(font_size)
            cell._loc = 'left'

    # Auto size columns; avoid relying on backend-specific attributes
    try:
        table.auto_set_column_width(col=list(range(n_cols)))
    except Exception:
        pass

    # Slightly upscale table for readability
    try:
        sx, sy = table_scale
        table.scale(float(sx), float(sy))
    except Exception:
        pass

    # Title
    ax.set_title(title, fontsize=font_size + 3, pad=12)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
    if out_pdf:
        plt.savefig(out_pdf, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export KMMLU-style grid from CSV")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--row-field", default="subject", help="Row grouping field name")
    parser.add_argument("--col-field", default="subdomain_name", help="Column grouping field name")
    parser.add_argument("--question-field", default="question", help="Question text field")
    parser.add_argument("--row-order", default=None, help="JSON list for explicit row order")
    parser.add_argument("--col-order", default=None, help="JSON list for explicit col order")
    parser.add_argument("--max-rows", type=int, default=4, help="Auto-pick up to N rows if no order provided")
    parser.add_argument("--max-cols", type=int, default=4, help="Auto-pick up to N cols if no order provided")
    parser.add_argument("--title", default="KMMLU-style Overview", help="Figure title")
    parser.add_argument("--out-png", default="subdomain_analysis_clean.png", help="Output PNG path")
    parser.add_argument("--out-pdf", default=None, help="Optional output PDF path")
    parser.add_argument("--cell-wrap", type=int, default=36, help="Wrap width for text in cells")
    parser.add_argument("--font", default=None, help="Matplotlib font family (e.g., 'DejaVu Sans', 'Malgun Gothic')")
    parser.add_argument("--font-size", type=int, default=11, help="Base font size")
    parser.add_argument("--fig-width", type=float, default=None, help="Figure width in inches (override auto)")
    parser.add_argument("--fig-height", type=float, default=None, help="Figure height in inches (override auto)")
    parser.add_argument("--table-scale", type=str, default="1.1,1.3", help="Scale factors for table as 'sx,sy'")
    parser.add_argument("--mode", type=str, default="sample", choices=["sample","count"], help="Cell content: sample a question or show counts")
    parser.add_argument("--per-cell", type=int, default=1, help="Samples per cell when mode=sample")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding='utf-8')
    if args.row_field not in df.columns:
        raise ValueError(f"Row field '{args.row_field}' not in CSV columns: {list(df.columns)}")
    if args.col_field not in df.columns:
        raise ValueError(f"Column field '{args.col_field}' not in CSV columns: {list(df.columns)}")
    if args.question_field not in df.columns:
        raise ValueError(f"Question field '{args.question_field}' not in CSV columns: {list(df.columns)}")

    row_order = parse_order_list(args.row_order)
    col_order = parse_order_list(args.col_order)

    if row_order is None:
        row_order = choose_values_by_frequency(df[args.row_field], args.max_rows)
    if col_order is None:
        col_order = choose_values_by_frequency(df[args.col_field], args.max_cols)

    grid = build_grid_cells(
        df=df,
        row_field=args.row_field,
        col_field=args.col_field,
        row_values=row_order,
        col_values=col_order,
        question_field=args.question_field,
        cell_wrap=args.cell_wrap,
        mode=args.mode,
        per_cell=max(1, args.per_cell),
    )

    # parse table scale
    try:
        sx_str, sy_str = (args.table_scale.split(",", 1) if "," in args.table_scale else args.table_scale.split(" ", 1))
        table_scale = (float(sx_str), float(sy_str))
    except Exception:
        table_scale = (1.1, 1.3)

    # override figsize if provided
    figsize = None
    if args.fig_width or args.fig_height:
        fw = float(args.fig_width) if args.fig_width else None
        fh = float(args.fig_height) if args.fig_height else None
        if fw and fh:
            figsize = (fw, fh)
        elif fw:
            figsize = (fw, 6.0)
        elif fh:
            figsize = (10.0, fh)

    render_table_image(
        grid=grid,
        title=args.title,
        out_png=args.out_png,
        out_pdf=args.out_pdf,
        font_name=args.font,
        font_size=args.font_size,
        figsize=figsize,
        table_scale=table_scale,
    )

    print(f"✅ Exported grid to: {args.out_png}{' and ' + args.out_pdf if args.out_pdf else ''}")


if __name__ == "__main__":
    main()


