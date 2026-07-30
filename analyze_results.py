"""
Analyze evaluation results: generate tables, plots, and statistical tests.

Reads the CSV output from ``evaluate_models.py`` and produces:
  - LaTeX tables for single-paradigm and hybrid (Seq/Sim) results.
  - Quantization robustness and stability box plots.
  - Per-subject accuracy distribution plots.
  - Statistical test results (Wilcoxon signed-rank tests).

Usage:
    python analyze_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import CONFIG
import math
from scipy.optimize import fsolve, brentq
from scipy.stats import friedmanchisquare, wilcoxon
import itertools
import matplotlib

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def load_and_preprocess(in_path):
    """Load evaluation results CSV and preprocess for analysis.

    Renames columns, converts bit-widths to ordered categoricals, computes
    latency/ITR columns split by Seq/Sim mode.

    Args:
        in_path: Path to the ``evaluation_results.csv`` file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(in_path)

    df["bits"] = df["bits"].replace({32: "32", 16: "16", 8: "8", 4: "4"})
    precision_order = ["32", "16", "8", "4"]
    df["bits"] = pd.Categorical(df["bits"].astype(str), categories=precision_order, ordered=True)

    df = df.rename(columns={
        "combination": "Paradigm",
        "model": "Model",
        "bits": "Bits",
        "mode": "Mode",
        "test_acc": "Acc",
        "f1": "F1",
        "kappa": "Kappa",
        "itr": "ITR",
        "latency_ms": "Lat",
        "n_classes": "N",
        "subject": "Subject"
    })

    df["Paradigm"] = df["Paradigm"].str.replace("+", "-", regex=False).str.upper()
    df["Lat"] = df["Lat"] / 1000.0  # Convert to seconds

    # Split latency and ITR by hybrid mode for plotting
    df['Lat-Sim'] = np.where(df['Mode'] == 'Sim', df['Lat'], np.nan)
    df['Lat-Seq'] = np.where(df['Mode'] == 'Seq', df['Lat'], np.nan)
    df['ITR-Sim'] = np.where(df['Mode'] == 'Sim', df['ITR'], np.nan)
    df['ITR-Seq'] = np.where(df['Mode'] == 'Seq', df['ITR'], np.nan)

    return df


def generate_subject_boxplots(df, out_dir):
    """Generate boxplots showing inter-subject variability.

    Plots accuracy distribution across subjects for 8-bit models on selected
    paradigms (MI, SSVEP, and hybrid combinations).

    Args:
        df: Preprocessed DataFrame (subject-level summary).
        out_dir: Output directory for the plot PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_8 = df[(df["Bits"] == "8") & (df["Subject"] != "All")].copy()

    # Filter to key paradigms
    desired_order = ["MI", "SSVEP", "MI-MI", "SSVEP-SSVEP", "MI-SSVEP"]
    df_8 = df_8[df_8["Paradigm"].isin(desired_order)].copy()
    df_8["Paradigm"] = pd.Categorical(df_8["Paradigm"], categories=desired_order, ordered=True)

    if df_8.empty:
        print("No subject-specific data available for boxplots.")
        return

    plt.figure(figsize=(12, 4), dpi=300)
    ax = sns.boxplot(data=df_8, x='Paradigm', y='Acc', hue='Model', palette='Set2', showfliers=False)
    sns.stripplot(data=df_8, x='Paradigm', y='Acc', hue='Model', dodge=True, alpha=0.4, size=4, legend=False, ax=ax)
    plt.title('Subject-Specific Accuracy Distribution (8-bit)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=0)

    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left')

    plt.tight_layout()
    plt.savefig(out_dir / 'subject_accuracy_distribution.pdf')
    plt.close()


# -------------------------------------------------------------------------
# LATEX TABLES GENERATION
# -------------------------------------------------------------------------
def _cmidrule(start: int, end: int) -> str:
    return rf"\cmidrule(lr){{{start}-{end}}}"


def build_table(df, paradigms, bits_list, mode, caption, label):
    """Build a LaTeX table comparing models across paradigms and bit-widths.

    Args:
        df: Preprocessed DataFrame (pooled data).
        paradigms: List of paradigm names to include as rows.
        bits_list: List of bit-width strings for column groups.
        mode: Hybrid mode filter (``"-"`` for single, ``"Seq"`` or ``"Sim"``).
        caption: LaTeX table caption.
        label: LaTeX table label.

    Returns:
        str: Complete LaTeX table source.
    """
    metric_keys = ["Acc", "F1", "ITR", "Kappa"]
    metric_labels = ["Acc", "F1", "ITR", "Kappa"]
    models = df['Model'].unique()
    n_metrics = len(metric_keys)
    n_bits = len(bits_list)
    n_models = len(models)

    sub = df[(df["Paradigm"].isin(paradigms)) & (df["Bits"].isin(bits_list)) & (df["Mode"] == mode)].copy()

    lookup = {}
    for _, row in sub.iterrows():
        for k in metric_keys:
            if pd.notna(row[k]):
                lookup[(str(row["Paradigm"]), str(row["Model"]), str(row["Bits"]), k)] = float(row[k])

    best = {}
    for p in paradigms:
        for k in metric_keys:
            for b in bits_list:
                vals = [lookup[(p, m, b, k)] for m in models if (p, m, b, k) in lookup]
                if vals:
                    best[(p, k, b)] = max(vals)

    col_fmt = "ll" + "c" * (n_models * n_bits)

    h1 = [("", 1), ("", 1)]
    cmidrules1 = []
    col_cursor = 3
    for model in models:
        h1.append((model, n_bits))
        cmidrules1.append(_cmidrule(col_cursor, col_cursor + n_bits - 1))
        col_cursor += n_bits

    h2 = [(r"\textbf{Paradigm}", 1), ("", 1)]
    cmidrules2 = []
    col_cursor = 3
    for _ in models:
        for b in bits_list:
            h2.append((str(b), 1))
            cmidrules2.append(_cmidrule(col_cursor, col_cursor))
            col_cursor += 1

    data_lines = []
    for p_idx, paradigm in enumerate(paradigms):
        for m_idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            paradigm_cell = rf"\multirow{{{n_metrics}}}{{*}}{{{paradigm}}}" if m_idx == 0 else ""
            row_parts = [paradigm_cell, mlabel]
            for model in models:
                for bits in bits_list:
                    val = lookup.get((paradigm, model, bits, mkey))
                    if val is None:
                        cell = "--"
                    else:
                        cell = f"{val:.2f}"
                        if mkey in ["Acc", "F1"]: cell = f"{val*100:.1f}"
                        if b := best.get((paradigm, mkey, bits)):
                            if abs(val - b) < 1e-6:
                                cell = r"\textbf{" + cell + "}"
                    row_parts.append(cell)
            data_lines.append(" & ".join(row_parts) + r" \\")
        if p_idx < len(paradigms) - 1:
            data_lines.append(r"\midrule")

    lines = [
        r"\begin{table*}", r"  \centering", rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
        r"  \small", r"  \setlength{\tabcolsep}{4pt}", rf"  \begin{{tabular}}{{{col_fmt}}}", r"    \toprule",
        "    " + " & ".join([rf"\multicolumn{{{sp}}}{{c}}{{{tx}}}" if sp > 1 else tx for tx, sp in h1]) + r" \\",
        "    " + " ".join(cmidrules1),
        "    " + " & ".join([tx for tx, sp in h2]) + r" \\", r"    \midrule"
    ] + ["    " + line for line in data_lines] + [
        r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"
    ]
    return "\n".join(lines)


def generate_all_tables(df, out_dir):
    """Generate all LaTeX tables (single, sequential hybrid, simultaneous hybrid).

    Args:
        df: Preprocessed DataFrame.
        out_dir: Output directory for ``.tex`` files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pooled = df[df["Subject"] == "All"]
    single_paradigms = [p for p in df_pooled['Paradigm'].unique() if '-' not in p]
    hybrid_paradigms = [p for p in df_pooled['Paradigm'].unique() if '-' in p]

    tables = [
        ("tab_single.tex", single_paradigms, ["32", "16", "8", "4"], "-", "Single Modality", "tab:single"),
        ("tab_hybrid_seq.tex", hybrid_paradigms, ["32", "16", "8", "4"], "Seq", "Sequential Hybrid", "tab:seq"),
        ("tab_hybrid_sim.tex", hybrid_paradigms, ["32", "16", "8", "4"], "Sim", "Simultaneous Hybrid", "tab:sim")
    ]

    for fname, paradigms, bits, mode, caption, label in tables:
        if paradigms:
            tex = build_table(df_pooled, paradigms, bits, mode, caption, label)
            (out_dir / fname).write_text(tex)


# -------------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------------
def generate_plots(df, out_dir):
    """Generate summary plots: quantization robustness and stability boxplot.

    Args:
        df: Preprocessed DataFrame.
        out_dir: Output directory for plot PDFs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pooled = df[df["Subject"] == "All"].copy()
    baseline_df = df_pooled[df_pooled['Bits'] == "32"].copy()

    plt.figure(figsize=(8, 6), dpi=300)
    sns.lineplot(data=df_pooled, x='Bits', y='Acc', hue='Model', linewidth=2, marker='o')
    plt.title('Quantization Robustness: Accuracy vs Bit Depth')
    plt.savefig(out_dir / 'quantization_robustness.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6), dpi=300)
    sns.boxplot(data=baseline_df, x='Model', y='Acc', hue='Model', palette='Set2', showfliers=False, legend=False)
    sns.stripplot(data=baseline_df, x='Model', y='Acc', hue='Paradigm', dodge=True, alpha=0.6, size=7)
    plt.title('Performance Stability Across Models and Paradigms (Pooled)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(out_dir / 'stability_boxplot.pdf', bbox_inches='tight')
    plt.close()


# -------------------------------------------------------------------------
# STATISTICAL TESTING
# -------------------------------------------------------------------------
def run_statistical_tests(df, out_dir):
    """Run Wilcoxon signed-rank tests and save results.

    Tests performed:
      1. Sequential vs Simultaneous (on ITR, per hybrid paradigm).
      2. 32-bit vs 8-bit (on Accuracy, per model).
      3. Model vs Model (on 32-bit Accuracy, pairwise).

    Args:
        df: Preprocessed DataFrame.
        out_dir: Output directory for the results CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pooled = df[df["Subject"] == "All"].copy()
    df_test = df_pooled.copy()

    res = []

    # 1. Wilcoxon: Seq vs Sim (on ITR)
    shared = sorted(set(df_test[df_test["Mode"] == "Seq"]["Paradigm"]) & set(df_test[df_test["Mode"] == "Sim"]["Paradigm"]))
    for combo in shared:
        seq = df_test[(df_test["Paradigm"] == combo) & (df_test["Mode"] == "Seq")].sort_values(["Model", "Bits", "fold"])["ITR"].values
        sim = df_test[(df_test["Paradigm"] == combo) & (df_test["Mode"] == "Sim")].sort_values(["Model", "Bits", "fold"])["ITR"].values
        n = min(len(seq), len(sim))
        if n >= 2:
            try:
                stat, p = wilcoxon(seq[:n], sim[:n])
                res.append({"Test": "Seq_vs_Sim (ITR)", "Comparison": combo, "p-value": p, "Significant": p < 0.05})
            except ValueError:
                pass

    # 2. Wilcoxon: 32-bit vs 8-bit (on Accuracy)
    for model in df_test["Model"].unique():
        b32 = df_test[(df_test["Model"] == model) & (df_test["Bits"] == "32")].sort_values(["Paradigm", "Mode", "fold"])["Acc"].values
        b8 = df_test[(df_test["Model"] == model) & (df_test["Bits"] == "8")].sort_values(["Paradigm", "Mode", "fold"])["Acc"].values
        n = min(len(b32), len(b8))
        if n >= 2:
            try:
                stat, p = wilcoxon(b32[:n], b8[:n])
                res.append({"Test": "32bit_vs_8bit (Acc)", "Comparison": model, "p-value": p, "Significant": p < 0.05})
            except ValueError:
                pass

    # 3. Wilcoxon: Model vs Model (on 32-bit Accuracy)
    models = df_test["Model"].unique()
    for m1, m2 in itertools.combinations(models, 2):
        acc1 = df_test[(df_test["Model"] == m1) & (df_test["Bits"] == "32")].sort_values(["Paradigm", "Mode", "fold"])["Acc"].values
        acc2 = df_test[(df_test["Model"] == m2) & (df_test["Bits"] == "32")].sort_values(["Paradigm", "Mode", "fold"])["Acc"].values
        n = min(len(acc1), len(acc2))
        if n >= 2:
            try:
                stat, p = wilcoxon(acc1[:n], acc2[:n])
                res.append({"Test": "Model_vs_Model (Acc, 32-bit)", "Comparison": f"{m1}_vs_{m2}", "p-value": p, "Significant": p < 0.05})
            except ValueError:
                pass

    pd.DataFrame(res).to_csv(out_dir / "statistical_results.csv", index=False)


def main():
    results_dir = Path(CONFIG["results_dir"])
    in_path = results_dir / "evaluation_results.csv"

    if not in_path.exists():
        print(f"No results file found at {in_path}.")
        return

    df = load_and_preprocess(in_path)

    # 1. Subject-level analysis
    if "Subject" in df.columns:
        # Average folds per subject
        subj_summary = df.groupby(["Model", "Paradigm", "Mode", "Bits", "Subject"], observed=True).agg({
            "Acc": "mean", "F1": "mean", "ITR": "mean", "Kappa": "mean", "Lat": "mean"
        }).reset_index()

        # Cross-subject average (excluding pooled "All")
        overall = subj_summary[subj_summary["Subject"] != "All"].groupby(["Model", "Paradigm", "Mode", "Bits"], observed=True).agg({
            "Acc": ["mean", "std"], "F1": ["mean", "std"]
        }).reset_index()
        overall.to_csv(results_dir / "subject_average_summary.csv", index=False)

        # Subject boxplots
        generate_subject_boxplots(subj_summary, results_dir)

    # 2. LaTeX tables (pooled data)
    generate_all_tables(df, results_dir / "latex_tables")

    # 3. Plots
    generate_plots(df, results_dir / "plots")

    # 4. Statistical tests
    run_statistical_tests(df, results_dir)

    print("Analysis complete. Check the Results/ directory.")


if __name__ == "__main__":
    main()
