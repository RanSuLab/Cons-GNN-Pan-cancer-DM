# Cons-GNN-Pan-cancer-DM

Cons-GNN: A contrastive graph learning framework for robust pan-cancer metastasis prediction across clinical distributions
## Project tree

```text
Cons-GNN/
├── checkpoints/                 # Model checkpoints
├── data/
│   ├── svs/                     # Input whole-slide .svs files
│   ├── patches/                 # Extracted patch images
│   ├── graphs/                  # COO graph datasets (.pt)
│   └── splits/                  # Optional split manifests
├── results/                     # Metrics and experiment outputs
├── scripts/
│   ├── extract_patch_args_multi_folders.py
│   └── get_features_graph_args_multi_folder.py
├── src/
│   └── Cons-GNN.py              # Model definitions
├── Engine.py                    # Training and evaluation
├── requirements.txt
└── README.md
```

## Setup

Use Python 3..9.7 or newer. Install PyTorch and the matching PyTorch Geometric build for your CUDA/CPU environment first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

OpenSlide system libraries and Python bindings are required for SVS reading. Keep raw slides locally; do not commit clinical data or generated graphs to a public repository.

## Data layout

Place slides under a directory containing one or more slide folders, for example:

```text
data/svs/
└── cohort_a/
    ├── case_001.svs
    └── case_002.svs
```

Patch images are grouped by slide. Labels are supplied as a CSV with columns `slide,label` where labels are `0` (no metastasis) or `1` (metastasis).

## Step 1: Extract patches

The command scans all `.svs` files recursively, tiles level 0, and filters background patches.

```bash
python scripts/extract_patch.py \
  --input-dir data/svs \
  --output-dir data/patches \
  --patch-size 512
```

## Step 2: Extract features and build graphs

DenseNet121 ImageNet weights are used as a feature extractor. Pairwise Euclidean distances are converted to Gaussian similarities, then edges below the 95% threshold are removed and each graph is capped at 20,000 directed COO edges.

```bash
python scripts/get_features_graph.py \
  --patch-dir data/patches \
  --labels-csv data/labels.csv \
  --output data/graphs/graphs.pt \
  --batch-size 32 \
  --max-edges 20000 \
  --quantile 0.95
```

The output is a list of PyTorch Geometric `Data` objects containing `x`, `edge_index`, `edge_weight`, `y`, and `graph_name`.

## Step 3: Train Cons-GNN

```bash
python Engine.py \
  --data-path data/graphs/graphs.pt \
  --output-dir results/run-01 \
  --checkpoint-dir checkpoints/run-01 \
  --epochs 200 \
  --batch-size 32 \
  --folds 5
```

Cross-validation groups augmented slide variants by the original slide name (the part before `+`) to avoid placing variants of one slide in both train and validation sets. Metrics are written to the output directory.

## Notes

- The scripts expect one patch folder per slide, named after the slide ID.
- The graph builder matches slide folder names against the `slide` column in the label CSV.
- Graph edges are stored as directed entries in COO form. The cap applies to stored directed edges.
- Large intermediate files under `data/` should normally be excluded from Git.
