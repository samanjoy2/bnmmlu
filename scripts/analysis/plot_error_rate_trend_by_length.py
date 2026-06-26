import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


BASE_DIR = Path(__file__).resolve().parent
FINISHED_DIR = BASE_DIR / "finished"
QUESTIONS_CSV = BASE_DIR / "merged_all_questions_with_subdomains_renamed.csv"
PLOT_PNG = FINISHED_DIR / "error_rate_trend_by_length_faceted.png"
PLOT_PDF = FINISHED_DIR / "error_rate_trend_by_length_faceted.pdf"
SELECTED_JSON = FINISHED_DIR / "selected_models_for_length_trend.json"
EXCLUDE_SUBSTRINGS = ["claude"]  # exclude any model containing these (case-insensitive)
MODEL_NAME_ALIASES = {
    "gpt-4.1-nano-2025-04-14": "gpt-5",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
}


def normalize_is_correct(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    return as_str.isin(truthy).astype(int)


def normalize_model_name(name: str) -> str:
    if name is None or pd.isna(name):
        return name
    s = str(name).strip()
    return MODEL_NAME_ALIASES.get(s, s)


def load_finished_results() -> pd.DataFrame:
    paths = sorted(FINISHED_DIR.glob("llm_test_results_*.csv"))
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p, dtype={"question_id": str}, low_memory=False)
            needed = {"question_id", "is_correct", "model_name"}
            if not needed.issubset(df.columns):
                continue
            df = df[["question_id", "is_correct", "model_name"]].copy()
            df["model_name"] = df["model_name"].map(normalize_model_name)
            frames.append(df.assign(source_file=p.name))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["question_id", "is_correct", "model_name", "source_file"]) 
    return pd.concat(frames, ignore_index=True)


LengthBin = Tuple[int, int, str]


def build_length_bins() -> List[LengthBin]:
    # Fixed bins to match the example
    specs: List[Tuple[int, int]] = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    bins: List[LengthBin] = []
    for lo, hi in specs:
        bins.append((lo, hi, f"{lo}-{hi}"))
    return bins


def assign_length_bin(length: int, bins: List[LengthBin]) -> str:
    if pd.isna(length):
        return "unknown"
    try:
        n = int(length)
    except Exception:
        return "unknown"
    for lo, hi, label in bins:
        if lo <= n <= hi:
            return label
    # Clip anything above the highest range into the last bin
    return bins[-1][2]


def load_question_lengths() -> pd.DataFrame:
    q = pd.read_csv(QUESTIONS_CSV, dtype={"Unique_Serial": str}, low_memory=False)
    # Prefer provided count; fallback to computing from text
    if "question_char_count" in q.columns:
        length = pd.to_numeric(q["question_char_count"], errors="coerce")
    else:
        length = q["question"].astype(str).str.len()
    return q[["Unique_Serial"]].assign(question_char_count=length)


def compute_error_by_length_bin(df_results: pd.DataFrame) -> pd.DataFrame:
    if df_results.empty:
        return pd.DataFrame(columns=["model_name", "length_bin", "error_rate", "n"])

    df_results = df_results.copy()
    df_results["is_correct_num"] = normalize_is_correct(df_results["is_correct"])  # 0/1

    qlen = load_question_lengths()
    df = df_results.merge(qlen, left_on="question_id", right_on="Unique_Serial", how="inner")

    bins = build_length_bins()
    df["length_bin"] = df["question_char_count"].apply(lambda n: assign_length_bin(n, bins))

    grouped = (
        df.groupby(["model_name", "length_bin"], as_index=False)
        .agg(
            mean_correct=("is_correct_num", "mean"),
            n=("is_correct_num", "size"),
        )
    )
    grouped["error_rate"] = 1.0 - grouped["mean_correct"]

    # Ensure consistent category order for plotting
    order = [b[2] for b in bins]
    grouped["length_bin"] = pd.Categorical(grouped["length_bin"], categories=order, ordered=True)

    return grouped.sort_values(["model_name", "length_bin"]).reset_index(drop=True)


def _exclude_models(df: pd.DataFrame, col: str = "model_name", patterns: List[str] = EXCLUDE_SUBSTRINGS) -> pd.DataFrame:
    if df.empty or not patterns:
        return df
    mask = pd.Series(True, index=df.index)
    for pat in patterns:
        try:
            mask &= ~df[col].astype(str).str.contains(pat, case=False, na=False)
        except Exception:
            continue
    return df[mask]


def summarize_models(grouped: pd.DataFrame) -> pd.DataFrame:
    # Compute per-model metrics for selection
    if grouped.empty:
        return pd.DataFrame(columns=[
            "model_name", "overall_accuracy", "min_bin_n", "spearman", "slope", "score", "bins_present"
        ])

    # Overall accuracy from original results would be more precise; we can approximate from mean across bins weighted by n
    agg = (
        grouped.groupby("model_name", as_index=False)
        .apply(lambda g: pd.Series({
            "overall_accuracy": (1.0 - np.average(g["error_rate"], weights=g["n"])) if g["n"].sum() > 0 else np.nan,
            "min_bin_n": int(g["n"].min()) if len(g) > 0 else 0,
            "bins_present": int(g["length_bin"].notna().sum()),
            "spearman": _spearman_for_group(g),
            "slope": _slope_for_group(g),
        }))
    )
    agg["score"] = agg["slope"].clip(lower=0) * agg["spearman"].clip(lower=0)
    return agg.sort_values("score", ascending=False).reset_index(drop=True)


def _spearman_for_group(g: pd.DataFrame) -> float:
    try:
        order = {label: idx for idx, label in enumerate(build_length_bins_label_order())}
        x = g["length_bin"].map(order).astype(float).to_numpy()
        y = g["error_rate"].astype(float).to_numpy()
        if len(x) < 3:
            return 0.0
        # Spearman via rank correlation
        xr = pd.Series(x).rank().to_numpy()
        yr = pd.Series(y).rank().to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])
    except Exception:
        return 0.0


def _slope_for_group(g: pd.DataFrame) -> float:
    try:
        order = {label: idx for idx, label in enumerate(build_length_bins_label_order())}
        x = g["length_bin"].map(order).astype(float).to_numpy()
        y = g["error_rate"].astype(float).to_numpy()
        if len(x) < 2:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope
    except Exception:
        return 0.0


def build_length_bins_label_order() -> List[str]:
    return [b[2] for b in build_length_bins()]


def select_top_models(grouped: pd.DataFrame, top_n: int = 10, min_bins: int = 4, min_bin_n: int = 30) -> List[str]:
    summary = summarize_models(grouped)
    if summary.empty:
        return []
    eligible = summary[(summary["bins_present"] >= min_bins) & (summary["min_bin_n"] >= min_bin_n)]
    if eligible.empty:
        eligible = summary
    selected = eligible.head(top_n)
    return selected["model_name"].tolist()


def plot_faceted(grouped: pd.DataFrame, selected_models: List[str]) -> None:
    if not selected_models:
        print("No models selected; nothing to plot.")
        return

    # Style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    # Bigger default fonts
    mpl.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 17,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 16,
    })

    order = build_length_bins_label_order()
    x_positions = np.arange(len(order))

    # Compute global y-limits for consistency
    sub = grouped[grouped["model_name"].isin(selected_models)]
    y_min = float(max(0.0, sub["error_rate"].min() - 0.05)) if not sub.empty else 0.0
    y_max = float(min(1.0, sub["error_rate"].max() + 0.05)) if not sub.empty else 1.0

    n_models = len(selected_models)
    n_cols = 5
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 12 if n_rows == 1 else 12), dpi=350, constrained_layout=False)
    axes = np.atleast_2d(axes)

    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["#1f77b4"])

    # Per-model overall accuracy for annotation
    overall_acc = (
        sub.groupby("model_name")
        .apply(lambda g: float(1.0 - np.average(g["error_rate"], weights=g["n"])) if g["n"].sum() > 0 else float('nan'))
        .to_dict()
    )

    for i, model in enumerate(selected_models):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        g = sub[sub["model_name"] == model].copy()

        # Ensure all bins are present in order (may be missing)
        g = g.set_index("length_bin").reindex(order).reset_index()
        y = g["error_rate"].astype(float).to_numpy()
        mask = ~np.isnan(y)
        xp = x_positions[mask]
        yp = y[mask]

        color = palette[i % len(palette)]

        # Smooth curve using polynomial fit (degree up to cubic, limited by data points)
        y_on_points = None
        if len(xp) >= 2:
            deg = int(min(3, len(xp) - 1))
            try:
                coeffs = np.polyfit(xp, yp, deg)
                poly = np.poly1d(coeffs)
                xs = np.linspace(xp.min(), xp.max(), 200)
                ys = poly(xs)
                ys = np.clip(ys, 0.0, 1.0)
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=4.0,
                    alpha=0.95,
                    solid_capstyle='round',
                    antialiased=True,
                )
                # Project markers onto the smooth curve for visual alignment
                y_on_points = np.clip(poly(xp), 0.0, 1.0)
            except Exception:
                pass

        # Plot actual points as markers only (no connecting line)
        ax.scatter(
            xp,
            (y_on_points if y_on_points is not None else yp),
            c=[color],
            s=54,
            marker='o',
            edgecolors='white',
            linewidths=1.0,
            zorder=3,
        )

        # Axes cosmetics
        ax.set_title(str(model), fontsize=18, weight="bold")
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Error Rate", fontsize=17, weight="bold")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=15, width=1.2)
        ax.tick_params(axis='y', pad=6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(order)
        # Ensure markers align with ticks by fixing x-limits to bin centers
        ax.set_xlim(-0.5, len(order) - 0.5)
        # Lighter, subtler grid
        ax.grid(True, axis='both', linewidth=0.6, alpha=0.5)
        # Make subplot box square
        try:
            ax.set_box_aspect(1)
        except Exception:
            try:
                ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass

        # Overall accuracy annotation
        acc = overall_acc.get(model, float('nan'))
        try:
            ax.text(0.02, 0.94, f"Overall Acc: {acc:.3f}", transform=ax.transAxes, fontsize=13,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', lw=0.5, alpha=0.8))
        except Exception:
            pass

    # Remove any unused subplots
    total_axes = n_rows * n_cols
    for j in range(n_models, total_axes):
        r = j // n_cols
        c = j % n_cols
        fig.delaxes(axes[r, c])

    # Shared labels
    fig.suptitle("Error Rate Trend by Question Length", fontsize=30, weight="bold")
    # Bottom axis label
    fig.text(0.5, 0.03, "Question Length (characters)", ha='center', fontsize=20, weight="bold")

    FINISHED_DIR.mkdir(parents=True, exist_ok=True)
    # Pack subplots tightly: reduce outer margins and gaps
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.075, wspace=0.12, hspace=0.08)
    fig.savefig(PLOT_PNG, dpi=350)
    try:
        fig.savefig(PLOT_PDF)
    except Exception:
        pass
    plt.close(fig)
    print(f"Saved plot: {PLOT_PNG}")


def main() -> None:
    results = load_finished_results()
    if results.empty:
        print("No finished CSVs found under 'finished/'.")
        return

    # Drop any models we don’t want to include (e.g., Claude)
    results = _exclude_models(results, col="model_name")

    grouped = compute_error_by_length_bin(results)
    selected = select_top_models(grouped, top_n=10, min_bins=4, min_bin_n=30)

    # Persist selection metadata for reproducibility
    try:
        selection_details: Dict[str, Dict[str, float]] = {}
        summary = summarize_models(grouped)
        for m in selected:
            row = summary[summary["model_name"] == m].head(1)
            if not row.empty:
                r = row.iloc[0]
                selection_details[m] = {
                    "overall_accuracy": float(r.get("overall_accuracy", float('nan'))),
                    "spearman": float(r.get("spearman", float('nan'))),
                    "slope": float(r.get("slope", float('nan'))),
                    "score": float(r.get("score", float('nan'))),
                    "min_bin_n": float(r.get("min_bin_n", float('nan'))),
                    "bins_present": float(r.get("bins_present", float('nan'))),
                }
        SELECTED_JSON.write_text(json.dumps({
            "selected_models": selected,
            "details": selection_details
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    plot_faceted(grouped, selected)


if __name__ == "__main__":
    main()


