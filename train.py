"""
train.py
--------
Core training logic for the GCN / GAT / Graph Transformer comparison.

Public API consumed by Project_Run.py:
    run_experiment(config: dict, run_dir: str) -> dict
    save_json(obj: dict, filepath: str) -> None

The module can also be run directly (python train.py) for quick testing;
in that case it writes output to ./Results/<timestamp>/ beside this file.
"""

import json
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from gcn_model import GCN
from gat_model import GAT
from graph_transformer_model import GraphTransformer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item() / len(labels)


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds   = logits.argmax(dim=1).cpu().numpy()
    targets = labels.cpu().numpy()
    return float(f1_score(targets, preds, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    data,
) -> Tuple[float, float, float, float, float, float]:
    """Return (train_acc, val_acc, test_acc, train_f1, val_f1, test_f1)."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
        val_acc   = accuracy(out[data.val_mask],   data.y[data.val_mask])
        test_acc  = accuracy(out[data.test_mask],  data.y[data.test_mask])
        train_f1v = macro_f1(out[data.train_mask], data.y[data.train_mask])
        val_f1v   = macro_f1(out[data.val_mask],   data.y[data.val_mask])
        test_f1v  = macro_f1(out[data.test_mask],  data.y[data.test_mask])
    return train_acc, val_acc, test_acc, train_f1v, val_f1v, test_f1v


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    data,
    model_name: str,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> Dict:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history: Dict[str, List] = {
        "train_loss": [], "train_acc": [],
        "val_acc":    [], "test_acc":  [],
        "train_f1":   [], "val_f1":    [], "test_f1": [],
    }

    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_val_f1   = 0.0
    best_test_f1  = 0.0
    start_time    = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc, train_f1v, val_f1v, test_f1v = evaluate(model, data)

        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["train_f1"].append(train_f1v)
        history["val_f1"].append(val_f1v)
        history["test_f1"].append(test_f1v)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            best_val_f1   = val_f1v
            best_test_f1  = test_f1v

        if (epoch + 1) % 20 == 0:
            print(
                f"{model_name} | Epoch {epoch + 1:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Acc {val_acc:.4f} | "
                f"Test Acc {test_acc:.4f} | "
                f"Val F1 {val_f1v:.4f}"
            )

    elapsed = time.time() - start_time

    history["best_val_acc"]          = best_val_acc
    history["best_test_acc"]         = best_test_acc
    history["best_val_f1"]           = best_val_f1
    history["best_test_f1"]          = best_test_f1
    history["training_time_seconds"] = round(elapsed, 2)

    return history


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_json(obj: dict, filepath: str) -> None:
    """Serialise *obj* to *filepath*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


# ---------------------------------------------------------------------------
# Plotting helpers  (all paths supplied by caller — no hard-coding)
# ---------------------------------------------------------------------------

def _plot_curve(
    values: List[float],
    title: str,
    ylabel: str,
    filepath: str,
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def _plot_comparison(
    histories: Dict[str, Dict],
    metric: str,
    title: str,
    ylabel: str,
    filepath: str,
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.figure(figsize=(9, 5))
    for name, h in histories.items():
        plt.plot(h[metric], label=name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def generate_plots(histories: Dict[str, Dict], plots_dir: str) -> None:
    """Write all per-model and comparison plots into *plots_dir*."""
    os.makedirs(plots_dir, exist_ok=True)
    short = {"GCN": "gcn", "GAT": "gat", "GraphTransformer": "gt"}

    for name, h in histories.items():
        key = short.get(name, name.lower())
        _plot_curve(h["train_loss"], f"{name} Training Loss",         "Loss",     os.path.join(plots_dir, f"{key}_loss.png"))
        _plot_curve(h["val_acc"],    f"{name} Validation Accuracy",   "Accuracy", os.path.join(plots_dir, f"{key}_val_acc.png"))
        _plot_curve(h["val_f1"],     f"{name} Validation F1 (Macro)", "F1",       os.path.join(plots_dir, f"{key}_val_f1.png"))

    _plot_comparison(histories, "val_acc",    "Validation Accuracy Comparison",   "Accuracy", os.path.join(plots_dir, "comparison_val_acc.png"))
    _plot_comparison(histories, "test_acc",   "Test Accuracy Comparison",         "Accuracy", os.path.join(plots_dir, "comparison_test_acc.png"))
    _plot_comparison(histories, "train_loss", "Training Loss Comparison",         "Loss",     os.path.join(plots_dir, "comparison_train_loss.png"))
    _plot_comparison(histories, "val_f1",     "Validation F1 Comparison (Macro)", "F1",       os.path.join(plots_dir, "comparison_val_f1.png"))


# ---------------------------------------------------------------------------
# run_experiment — primary entry point called by Project_Run.py
# ---------------------------------------------------------------------------

def run_experiment(config: dict, run_dir: str) -> dict:
    """
    Train all three models under *config* and save artefacts into *run_dir*.

    Directory layout written by this function:
        <run_dir>/
            plots/          all PNG plots
            results/        per-model JSON histories
            run_summary.json

    Returns a summary dict with scalar metrics for every model so that
    Project_Run.py can write hyperparameters.txt, config.json, and
    final_metrics.csv without needing to re-run training.
    """
    plots_dir   = os.path.join(run_dir, "plots")
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Dataset path is always relative to the project root (where this file lives)
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root    = os.path.join(project_root, "data", "Cora")

    dataset = Planetoid(
        root=data_root,
        name=config.get("dataset_name", "Cora"),
        transform=NormalizeFeatures(),
    )
    data = dataset[0]

    # --- Build models from config ------------------------------------------
    gcn = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=config["gcn_hidden_channels"],
        out_channels=dataset.num_classes,
        dropout=config["gcn_dropout"],
    )
    gat = GAT(
        in_channels=dataset.num_node_features,
        hidden_channels=config["gat_hidden_channels"],
        out_channels=dataset.num_classes,
        heads=config["gat_heads"],
        dropout=config["gat_dropout"],
    )
    gt = GraphTransformer(
        in_channels=dataset.num_node_features,
        hidden_channels=config["gt_hidden_channels"],
        out_channels=dataset.num_classes,
        heads=config["gt_heads"],
        dropout=config["gt_dropout"],
    )

    epochs = config.get("epochs", 200)

    # --- Train -------------------------------------------------------------
    print("\nTraining GCN...")
    gcn_history = train_model(gcn, data, "GCN",
                              epochs=epochs,
                              lr=config["gcn_lr"],
                              weight_decay=config["gcn_weight_decay"])

    print("\nTraining GAT...")
    gat_history = train_model(gat, data, "GAT",
                              epochs=epochs,
                              lr=config["gat_lr"],
                              weight_decay=config["gat_weight_decay"])

    print("\nTraining Graph Transformer...")
    gt_history  = train_model(gt,  data, "GraphTransformer",
                              epochs=epochs,
                              lr=config["gt_lr"],
                              weight_decay=config["gt_weight_decay"])

    all_histories = {
        "GCN":              gcn_history,
        "GAT":              gat_history,
        "GraphTransformer": gt_history,
    }

    # --- Save per-model JSON histories -------------------------------------
    save_json(gcn_history, os.path.join(results_dir, "gcn_history.json"))
    save_json(gat_history, os.path.join(results_dir, "gat_history.json"))
    save_json(gt_history,  os.path.join(results_dir, "gt_history.json"))

    # --- Generate all plots ------------------------------------------------
    generate_plots(all_histories, plots_dir)

    # --- Build and save run_summary.json -----------------------------------
    summary = {
        "dataset": config.get("dataset_name", "Cora"),
        "epochs":  epochs,
        "models": {
            name: {
                "best_val_acc":          h["best_val_acc"],
                "best_test_acc":         h["best_test_acc"],
                "best_val_f1":           h["best_val_f1"],
                "best_test_f1":          h["best_test_f1"],
                "training_time_seconds": h["training_time_seconds"],
            }
            for name, h in all_histories.items()
        },
    }
    save_json(summary, os.path.join(run_dir, "run_summary.json"))

    return summary


# ---------------------------------------------------------------------------
# Standalone entry point  (python train.py)
# ---------------------------------------------------------------------------

def main() -> None:
    import datetime
    project_root = os.path.dirname(os.path.abspath(__file__))
    timestamp    = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
    run_dir      = os.path.join(project_root, "Results", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "dataset_name":        "Cora",
        "epochs":              200,
        "gcn_hidden_channels": 16,  "gcn_dropout": 0.5,  "gcn_lr": 0.01,   "gcn_weight_decay": 5e-4,
        "gat_hidden_channels": 8,   "gat_heads": 8,  "gat_dropout": 0.6,   "gat_lr": 0.005,  "gat_weight_decay": 5e-4,
        "gt_hidden_channels":  8,   "gt_heads":  8,  "gt_dropout":  0.6,   "gt_lr":  0.005,  "gt_weight_decay":  5e-4,
    }

    summary = run_experiment(config, run_dir)

    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'Best Val Acc':>12} {'Best Test Acc':>14} {'Best Test F1':>13} {'Time (s)':>10}")
    print("-" * 70)
    for name, m in summary["models"].items():
        print(f"{name:<22} {m['best_val_acc']:>12.4f} {m['best_test_acc']:>14.4f} "
              f"{m['best_test_f1']:>13.4f} {m['training_time_seconds']:>10.1f}")
    print("=" * 70)
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()