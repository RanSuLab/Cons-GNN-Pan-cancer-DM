import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import functional as F
from torch_geometric.loader import DataLoader


MODEL_PATH = Path(__file__).parent / "src" / "Cons-GNN.py"
spec = importlib.util.spec_from_file_location("cons_gnn_model", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
ConsGNN = model_module.ConsGNN
augment_graph = model_module.augment_graph
contrastive_loss = model_module.contrastive_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Cons-GNN with slide-grouped cross-validation.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--contrastive-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_graphs(path):
    try:
        graphs = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        graphs = torch.load(path, map_location="cpu")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"Expected a non-empty list of PyG Data objects in {path}")
    return graphs


def grouped_folds(graphs, folds, seed):
    grouped = {}
    for graph in graphs:
        slide_id = str(graph.graph_name)
        original_id = slide_id.split("+")[0]
        grouped.setdefault(original_id, []).append(graph)
    groups = list(grouped.values())
    if len(groups) < folds:
        raise ValueError(f"Need at least {folds} unique slides, found {len(groups)}")
    random.Random(seed).shuffle(groups)
    return [groups[i::folds] for i in range(folds)]


def batch_embeddings(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs.append(model.encoder(batch.x, batch.edge_index, batch.edge_weight, batch.batch))
    return outputs


def train_fold(train_graphs, valid_graphs, fold_index, args, device):
    model = ConsGNN(input_dim=train_graphs[0].x.size(1)).to(device)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False)

    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    model.train()
    for epoch in range(args.contrastive_epochs):
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            view_a, view_b = augment_graph(batch), augment_graph(batch)
            z_a = model.encoder(view_a.x, view_a.edge_index, view_a.edge_weight, view_a.batch)
            z_b = model.encoder(view_b.x, view_b.edge_index, view_b.edge_weight, view_b.batch)
            loss = contrastive_loss(z_a, z_b)
            encoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            total += loss.item()
        print(f"Fold {fold_index} contrastive epoch {epoch + 1}: loss={total:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = F.nll_loss(output, batch.y.view(-1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            probabilities.extend(output.exp()[:, 1].cpu().tolist())
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            labels.extend(batch.y.view(-1).cpu().tolist())

    metrics = {
        "fold": fold_index,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auc": roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else float("nan"),
        "aupr": average_precision_score(labels, probabilities) if len(set(labels)) > 1 else float("nan"),
    }
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_dir / f"fold_{fold_index}.pt")
    return metrics


def main():
    args = parse_args()
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("folds must be >= 2; epochs and batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    graphs = load_graphs(args.data_path)
    folds = grouped_folds(graphs, args.folds, args.seed)
    all_metrics = []
    for fold_index in range(args.folds):
        valid_graphs = folds[fold_index]
        train_graphs = [graph for index, fold in enumerate(folds) if index != fold_index for graph in fold]
        print(f"Starting fold {fold_index + 1}/{args.folds}: train={len(train_graphs)}, valid={len(valid_graphs)}")
        all_metrics.append(train_fold(train_graphs, valid_graphs, fold_index + 1, args, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(all_metrics)
    results.loc[len(results)] = {"fold": "mean", **results.drop(columns="fold").mean().to_dict()}
    results.to_csv(args.output_dir / "metrics.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
