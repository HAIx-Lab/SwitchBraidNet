"""
Evaluate trained models across single and hybrid BCI paradigms.

Runs the hybrid BCI evaluation pipeline for all configured models and
precision levels. For each model, evaluates all paradigms (single and
Cartesian hybrid combinations) across all subjects and cross-validation
folds. Results are saved to a CSV file.

Usage:
    python evaluate_models.py
"""

import pandas as pd
from pathlib import Path
import traceback
from config import CONFIG
from utils.registry import get_model_spec
from utils.hybrid_eval import run_hbci_eval


def main():
    print("Starting Unified Evaluation Workflow...")

    all_results = []

    for model_name in CONFIG["models"]:
        try:
            spec = get_model_spec(model_name)
            # run_hbci_eval evaluates all modalities and their Cartesian
            # hybrid products simultaneously — no need to iterate over modes.
            for bits in CONFIG["precisions"]:
                results = run_hbci_eval(spec, bits, CONFIG)
                all_results.extend(results)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)

        # Enforce a specific column order for consistency
        col_order = [
            "model", "bits", "subject", "fold", "combination", "mode", "n_classes",
            "test_acc", "f1", "precision", "recall", "kappa", "itr", "latency_ms", "size_mb"
        ]

        # Ensure all columns exist before reordering
        for col in col_order:
            if col not in df.columns:
                df[col] = float('nan')

        df = df[col_order]

        results_dir = Path(CONFIG["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "evaluation_results.csv"
        df.to_csv(out_path, index=False, float_format='%.6f')
        print(f"\nEvaluation complete. Results saved to {out_path}")
    else:
        print("\nNo results were gathered.")


if __name__ == "__main__":
    main()
