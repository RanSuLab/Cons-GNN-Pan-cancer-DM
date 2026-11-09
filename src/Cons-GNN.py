import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class GCNEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        return global_add_pool(x, batch)


class ConsGNN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, embedding_dim=128, num_classes=2):
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, edge_weight, batch):
        graph_embedding = self.encoder(x, edge_index, edge_weight, batch)
        return F.log_softmax(self.classifier(graph_embedding), dim=1)


def contrastive_loss(view_a, view_b, temperature=0.5):
    view_a = F.normalize(view_a, dim=1)
    view_b = F.normalize(view_b, dim=1)
    logits = view_a @ view_b.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)


def augment_graph(data, edge_drop=0.2, feature_mask=0.2):
    view = data.clone()
    edge_mask = torch.rand(view.edge_index.size(1), device=view.edge_index.device) >= edge_drop
    view.edge_index = view.edge_index[:, edge_mask]
    if getattr(view, "edge_weight", None) is not None:
        view.edge_weight = view.edge_weight[edge_mask]
    feature_mask_tensor = torch.rand_like(view.x) >= feature_mask
    view.x = view.x * feature_mask_tensor
    return view
