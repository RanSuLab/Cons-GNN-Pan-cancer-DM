import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from torchvision.models import DenseNet121_Weights, densenet121


class PatchDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def gaussian_coo(features, quantile, max_edges):
    distances = torch.cdist(features, features, p=2)
    positive = distances[distances > 0]
    sigma = positive.median().clamp_min(1e-12) if positive.numel() else torch.tensor(1.0)
    similarity = torch.exp(-distances.square() / (2 * sigma.square()))
    similarity.fill_diagonal_(0)

    values = similarity.flatten()
    nonzero = values > 0
    candidates = values[nonzero]
    if candidates.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    threshold = torch.quantile(candidates, quantile)
    keep = similarity >= threshold
    keep.fill_diagonal_(False)
    indices = torch.nonzero(keep, as_tuple=False)
    weights = similarity[indices[:, 0], indices[:, 1]]
    if weights.numel() > max_edges:
        selected = torch.topk(weights, max_edges, sorted=False).indices
        indices, weights = indices[selected], weights[selected]
    return indices.t().contiguous(), weights.contiguous()


def main():
    parser = argparse.ArgumentParser(description="Extract DenseNet features and build sparse WSI graphs.")
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-edges", type=int, default=20000)
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_edges < 1 or not 0 <= args.quantile <= 1:
        parser.error("batch-size and max-edges must be positive; quantile must be in [0, 1]")

    labels_frame = pd.read_csv(args.labels_csv)
    if not {"slide", "label"}.issubset(labels_frame.columns):
        parser.error("Labels CSV must contain slide,label columns")
    labels = dict(zip(labels_frame["slide"].astype(str), labels_frame["label"].astype(int)))

    weights = DenseNet121_Weights.DEFAULT
    transform = weights.transforms()
    model = densenet121(weights=weights)
    model.classifier = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    graphs = []
    for patch_folder in sorted(path for path in args.patch_dir.iterdir() if path.is_dir()):
        slide_id = patch_folder.name
        if slide_id not in labels:
            print(f"Skip {slide_id}: no label entry")
            continue
        image_paths = sorted(p for p in patch_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        if not image_paths:
            print(f"Skip {slide_id}: no patches")
            continue

        loader = DataLoader(PatchDataset(image_paths, transform), batch_size=args.batch_size, shuffle=False)
        batches = []
        with torch.inference_mode():
            for batch in loader:
                batches.append(model(batch.to(device)).cpu())
        features = torch.cat(batches).float()
        edge_index, edge_weight = gaussian_coo(features, args.quantile, args.max_edges)
        graphs.append(Data(
            x=features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([labels[slide_id]], dtype=torch.long),
            graph_name=slide_id,
        ))
        print(f"{slide_id}: {features.size(0)} nodes, {edge_index.size(1)} directed edges")

    if not graphs:
        parser.error("No graphs were created; check patch folders and label IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, args.output)
    print(f"Saved {len(graphs)} graphs to {args.output}")


if __name__ == "__main__":
    main()
