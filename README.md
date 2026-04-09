# Graph Project

This project compares three graph neural network models on the **Cora** dataset:

- GCN
- GAT
- Graph Transformer

The code trains each model, saves plots, and writes summary metrics for easy comparison.

## Project Files

- `project_run.py` - main script to run the full experiment
- `train.py` - training and evaluation logic
- `gcn_model.py` - GCN model
- `gat_model.py` - GAT model
- `graph_transformer_model.py` - Graph Transformer model
- `dataset_check.py` - dataset checking utility

## Environment

Recommended:

- Python 3.11
- Windows PowerShell or CMD

## Setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv env
.\env\Scripts\activate

## Notes

- Some Windows systems may encounter path length limitations during dependency installation (e.g., PyTorch).
- If this occurs, move the project to a shorter path such as:
  - `C:\graph_project`
- This is only necessary if installation errors occur.
