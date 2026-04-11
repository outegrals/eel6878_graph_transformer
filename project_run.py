"""
Project_Run.py
--------------
Entry point for a full experiment run.

Usage:
    python Project_Run.py

What it does:
    1.  Creates  graph_project/Results/<timestamp>/  as the run directory.
    2.  Redirects all stdout/stderr to console_log.txt inside that directory
        so every training print is captured automatically.
    3.  Calls run_experiment() from train.py — all training, saving, and
        plotting happens there and writes into the run directory.
    4.  After the experiment, writes:
            hyperparameters.txt   — human-readable config summary
            config.json           — machine-readable config (for re-runs)
            final_metrics.csv     — one row per model, key scalars only

    run_summary.json is written by run_experiment() itself.

To hand this project off:
    - Recipient clones / unzips the graph_project folder.
    - Installs dependencies:  pip install torch torch-geometric scikit-learn matplotlib
    - Runs:  python Project_Run.py
    - All outputs appear in Results/<new_timestamp>/ — nothing is overwritten.
    - To try different hyperparameters, edit the `config` dict below.
"""

import contextlib
import csv
import datetime
import os
from pprint import pformat

# macOS OpenMP runtime conflicts can occur when multiple libraries bring in
# `libomp.dylib`. This workaround allows the script to continue executing
# in environments where the runtime is already initialized by another library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch and torch_geometric early to avoid import issues
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from train import run_experiment, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_directory(project_root: str) -> str:
    """Return a fresh timestamped directory under <project_root>/Results/."""
    results_root = os.path.join(project_root, "Results")
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
    run_dir = os.path.join(results_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_hyperparameters(config: dict, filepath: str) -> None:
    """Write a plain-text summary of every hyperparameter used this run."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Hyperparameters Used For This Run\n")
        f.write("=" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def write_metrics_csv(summary: dict, filepath: str) -> None:
    """Write one CSV row per model with key scalar metrics."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model",
            "Best Val Acc",
            "Best Test Acc",
            "Best Val F1",
            "Best Test F1",
            "Training Time (s)",
        ])
        for model_name, metrics in summary["models"].items():
            writer.writerow([
                model_name,
                metrics["best_val_acc"],
                metrics["best_test_acc"],
                metrics["best_val_f1"],
                metrics["best_test_f1"],
                metrics["training_time_seconds"],
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # All paths are derived from the location of this file — no hard-coding
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_dir      = make_run_directory(project_root)
    log_path     = os.path.join(run_dir, "console_log.txt")

    print(f"Run directory: {run_dir}")
    print(f"Logging to: {log_path}")
    
    # IMPORTANT: Pre-load dataset BEFORE entering redirect context
    print("Pre-loading dataset...", flush=True)
    data_root = os.path.join(project_root, "data", "Cora")
    dataset = Planetoid(root=data_root, name="Cora", transform=NormalizeFeatures())
    data = dataset[0]
    print(f"Dataset pre-loaded: {data.num_nodes} nodes, {data.num_edges} edges\n", flush=True)

    # Edit this dict to change hyperparameters — nothing else needs touching
    config = {
        "dataset_name":        "Cora",
        "epochs":              200,
        # GCN
        "gcn_hidden_channels": 16,
        "gcn_dropout":         0.5,
        "gcn_lr":              0.01,
        "gcn_weight_decay":    5e-4,
        # GAT
        "gat_hidden_channels": 8,
        "gat_heads":           8,
        "gat_dropout":         0.6,
        "gat_lr":              0.005,
        "gat_weight_decay":    5e-4,
        # Graph Transformer
        "gt_hidden_channels":  8,
        "gt_heads":            8,
        "gt_dropout":          0.6,
        "gt_lr":               0.005,
        "gt_weight_decay":     5e-4,
    }

    # ------------------------------------------------------------------
    # Run experiment with all output captured to console_log.txt
    # summary is assigned inside the block; we re-raise on failure so the
    # finally section can still write the artefacts that don't depend on it.
    # ------------------------------------------------------------------
    summary = None
    try:
        with (
            open(log_path, "w", encoding="utf-8", buffering=1) as log_file,
            contextlib.redirect_stdout(log_file),
            contextlib.redirect_stderr(log_file),
        ):
            try:
                print("Graph Project Run", flush=True)
                print("=" * 60)
                print(f"Run directory: {run_dir}")
                print("\nConfiguration:")
                print(pformat(config))
                print("\nStarting experiment...\n")

                summary = run_experiment(config, run_dir, data)

                print("\nFinal Summary")
                print("=" * 60)
                for model_name, metrics in summary["models"].items():
                    print(f"\n{model_name}")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                print("\nRun completed successfully.")
                
            except Exception as inner_exc:
                # If something fails inside the redirected context, still try to log it
                print(f"\nEXCEPTION INSIDE REDIRECT: {type(inner_exc).__name__}")
                print(f"Message: {inner_exc}")
                import traceback
                traceback.print_exc()
                raise

    except Exception as exc:
        # If training crashes, surface the error to the terminal instead of
        # swallowing it silently inside the log file redirect.
        raise RuntimeError(
            f"Experiment failed. Partial logs saved to: {log_path}"
        ) from exc

    finally:
        # These artefacts don't require a completed summary — safe to always write
        write_hyperparameters(config, os.path.join(run_dir, "hyperparameters.txt"))
        save_json(config,            os.path.join(run_dir, "config.json"))

    # Only reached if experiment succeeded and summary is populated
    write_metrics_csv(summary, os.path.join(run_dir, "final_metrics.csv"))

    print(f"Run complete. Results saved to:\n  {run_dir}")
    print("\nFiles written:")
    for fname in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, fname)
        if os.path.isdir(fpath):
            sub = sorted(os.listdir(fpath))
            print(f"  {fname}/  ({len(sub)} files)")
        else:
            print(f"  {fname}")


if __name__ == "__main__":
    main()