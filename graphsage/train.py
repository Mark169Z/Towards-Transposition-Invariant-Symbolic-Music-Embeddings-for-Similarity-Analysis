import os
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from build_graph import build_graph_data, export_debug_tables
from model import GraphSAGE

def unique_undirected_edges(edge_index: torch.Tensor):
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    pairs = set()
    for u, v in zip(src, dst):
        if u == v:
            continue
        a, b = sorted((u, v))
        pairs.add((a, b))
    pairs = sorted(pairs)
    return torch.tensor(pairs, dtype=torch.long)


def split_edges(pos_edges: torch.Tensor, train_ratio=0.9, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(pos_edges)))
    rng.shuffle(idx)

    cut = int(len(idx) * train_ratio)
    train_idx = idx[:cut]
    val_idx = idx[cut:]

    train_edges = pos_edges[train_idx]
    val_edges = pos_edges[val_idx] if val_idx else pos_edges[:0]
    return train_edges, val_edges


def make_bidirectional(edge_pairs: torch.Tensor) -> torch.Tensor:
    rev = edge_pairs[:, [1, 0]]
    both = torch.cat([edge_pairs, rev], dim=0)
    return both.t().contiguous()


def sample_negative_edges(num_nodes: int, num_samples: int, positive_edge_set: set, device):
    neg = []
    while len(neg) < num_samples:
        u = random.randrange(num_nodes)
        v = random.randrange(num_nodes)
        if u == v:
            continue
        a, b = sorted((u, v))
        if (a, b) in positive_edge_set:
            continue
        neg.append((u, v))
    return torch.tensor(neg, dtype=torch.long, device=device)


def edge_scores(z: torch.Tensor, edges: torch.Tensor):
    return (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=1)


def evaluate_loss(model, x, edge_index, pos_edges, positive_edge_set, device):
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
        if len(pos_edges) == 0:
            return float("nan")

        neg_edges = sample_negative_edges(
            num_nodes=x.size(0),
            num_samples=len(pos_edges),
            positive_edge_set=positive_edge_set,
            device=device
        )
        pos_logits = edge_scores(z, pos_edges.to(device))
        neg_logits = edge_scores(z, neg_edges)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels) + \
               F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
        return float(loss.item())


def export_song_vectors(out_dir: Path, z: torch.Tensor, song_ids, song_node_indices, midi_files):
    out_dir.mkdir(parents=True, exist_ok=True)

    song_z = z[song_node_indices].detach().cpu().numpy().astype(np.float32)
    dim = song_z.shape[1]

    rows = []
    for i, (song_id, midi_file) in enumerate(zip(song_ids, midi_files)):
        row = {
            "song_id": song_id,
            "emb_key": song_id,
            "midi_file": midi_file,
        }
        for j in range(dim):
            row[f"v{j}"] = float(song_z[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "song_vectors.csv"
    npy_path = out_dir / "song_vectors.npy"
    node_csv_path = out_dir / "all_node_vectors.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    np.save(
        npy_path,
        {
            "vectors": song_z,
            "meta": df[["song_id", "emb_key", "midi_file"]].to_dict("records")
        }
    )

    print(f"[OK] wrote song vectors csv: {csv_path}")
    print(f"[OK] wrote song vectors npy: {npy_path}")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="edgelist_file", help="Folder containing .edgelist and names_emb.csv")
    ap.add_argument("--names", type=str, default=None, help="Optional path to names_emb.csv")
    ap.add_argument("--out_dir", type=str, default="embeddings", help="Output directory")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--out_dim", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    names_csv = args.names or os.path.join(args.input, "names_emb.csv")
    data = build_graph_data(
        edgelist_dir=args.input,
        names_csv=names_csv,
        exclude=args.exclude
    )

    export_debug_tables(data, Path(args.out_dir) / "debug_tables")
    
    x = data["x"]
    full_edge_index = data["edge_index"]
    song_ids = data["song_ids"]
    song_node_indices = data["song_node_indices"]
    midi_files = data["midi_files"]

    pos_edges = unique_undirected_edges(full_edge_index)
    train_pos, val_pos = split_edges(pos_edges, train_ratio=args.train_ratio, seed=args.seed)

    train_edge_index = make_bidirectional(train_pos)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    train_edge_index = train_edge_index.to(device)

    model = GraphSAGE(
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    positive_edge_set = {tuple(sorted(map(int, e))) for e in pos_edges.tolist()}

    print("num_nodes:", x.size(0))
    print("num_features:", x.size(1))
    print("train positive edges:", len(train_pos))
    print("val positive edges:", len(val_pos))
    print("device:", device)

    best_val = math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model(x, train_edge_index)

        neg_edges = sample_negative_edges(
            num_nodes=x.size(0),
            num_samples=len(train_pos),
            positive_edge_set=positive_edge_set,
            device=device
        )

        pos_logits = edge_scores(z, train_pos.to(device))
        neg_logits = edge_scores(z, neg_edges)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels) + \
               F.binary_cross_entropy_with_logits(neg_logits, neg_labels)

        loss.backward()
        optimizer.step()

        val_loss = evaluate_loss(model, x, train_edge_index, val_pos, positive_edge_set, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        z = model(x, full_edge_index.to(device))

    out_dir = Path(args.out_dir)
    export_song_vectors(
        out_dir=out_dir,
        z=z,
        song_ids=song_ids,
        song_node_indices=song_node_indices,
        midi_files=midi_files
    )

    model_path = out_dir / "graphsage_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": x.size(1),
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "song_ids": song_ids,
    }, model_path)
    print(f"[OK] saved model: {model_path}")


if __name__ == "__main__":
    main()