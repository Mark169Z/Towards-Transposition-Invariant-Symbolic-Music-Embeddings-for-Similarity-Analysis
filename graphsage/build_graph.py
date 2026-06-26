import os
import csv
import re
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

import torch

def export_debug_tables(data: dict, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = data["nodes"]
    x = data["x"].detach().cpu().numpy()
    edge_index = data["edge_index"].detach().cpu().numpy()

    # ---------- NODE TABLE ----------
    rows = []
    for i, node_name in enumerate(nodes):
        node_type = infer_node_type(node_name)

        row = {
            "node_idx": i,
            "node_name": node_name,
            "node_type": node_type,
        }

        # ใส่ feature vector (v0, v1, ...)
        for j in range(x.shape[1]):
            row[f"v{j}"] = float(x[i, j])

        rows.append(row)

    df_nodes = pd.DataFrame(rows)
    df_nodes.to_csv(out_dir / "nodes.csv", index=False, encoding="utf-8-sig")

    # ---------- EDGE TABLE ----------
    src = edge_index[0]
    dst = edge_index[1]

    edge_rows = []
    for s, t in zip(src, dst):
        edge_rows.append({
            "src_idx": int(s),
            "src_name": nodes[int(s)],
            "dst_idx": int(t),
            "dst_name": nodes[int(t)],
        })

    df_edges = pd.DataFrame(edge_rows)
    df_edges.to_csv(out_dir / "edges.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] Exported nodes.csv & edges.csv to: {out_dir}")

def read_names_csv(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["id"], row["filename"]))
    return rows


def load_edgelists(edgelist_dir: str, exclude=None):
    exclude = set(exclude or [])
    edges = []

    for fn in os.listdir(edgelist_dir):
        if not fn.endswith(".edgelist"):
            continue
        stem = fn.rsplit(".", 1)[0]
        if fn in exclude or stem in exclude:
            continue

        full = os.path.join(edgelist_dir, fn)
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                s, t = parts
                edges.append((s, t))

    return edges


def infer_node_type(node: str) -> str:
    if node.startswith("-") or node.startswith("C:"):
        return "song"
    if node.startswith("g"):
        return "group"
    if node.startswith("dur:"):
        return "duration"
    if node.startswith("vel:"):
        return "velocity"
    if node.startswith("tempo:"):
        return "tempo"
    if node.startswith("timesig:"):
        return "timesig"
    if "midi-ld/programs/" in node:
        return "program"
    if "midi-ld/notes_rel/" in node:
        return "rel_pitch"
    if re.fullmatch(r"\d+", node):
        return "tempo"
    return "other"


NODE_TYPES = [
    "song",
    "group",
    "duration",
    "velocity",
    "tempo",
    "timesig",
    "program",
    "rel_pitch",
    "other",
]
NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}


def parse_scalar_features(node: str):
    """
    scalar features 7 ค่า:
    [tempo, duration, velocity, rel_pitch, program, ts_num, ts_den]
    """
    vals = [0.0] * 7

    if node.startswith("tempo:"):
        try:
            vals[0] = float(node.split(":", 1)[1]) / 30.0
        except Exception:
            pass
    elif node.startswith("dur:"):
        try:
            vals[1] = float(node.split(":", 1)[1]) / 100.0
        except Exception:
            pass
    elif node.startswith("vel:"):
        try:
            vals[2] = float(node.split(":", 1)[1]) / 20.0
        except Exception:
            pass
    elif "midi-ld/notes_rel/" in node:
        try:
            pitch = float(node.rsplit("/", 1)[1])
            vals[3] = pitch / 24.0
        except Exception:
            pass
    elif "midi-ld/programs/" in node:
        try:
            prog = float(node.rsplit("/", 1)[1])
            vals[4] = prog / 127.0
        except Exception:
            pass
    elif node.startswith("timesig:"):
        try:
            frac = node.split(":", 1)[1]
            num, den = frac.split("/")
            vals[5] = float(num) / 16.0
            vals[6] = float(den) / 16.0
        except Exception:
            pass
    elif re.fullmatch(r"\d+", node):
        # backward compatibility กรณี tempo ยังเป็นเลขล้วน
        vals[0] = float(node) / 30.0

    return vals


def build_node_features(nodes: List[str]) -> torch.Tensor:
    feats = []
    for n in nodes:
        t = infer_node_type(n)
        one_hot = [0.0] * len(NODE_TYPES)
        one_hot[NODE_TYPE_TO_IDX[t]] = 1.0
        scalars = parse_scalar_features(n)
        feats.append(one_hot + scalars)
    return torch.tensor(feats, dtype=torch.float32)


def build_graph_data(edgelist_dir: str, names_csv: str, exclude=None):
    name_rows = read_names_csv(names_csv)
    song_id_to_file = dict(name_rows)

    edges = load_edgelists(edgelist_dir, exclude=exclude)

    nodes = set()
    for s, t in edges:
        nodes.add(s)
        nodes.add(t)

    # ensure every song in names.csv exists in node set
    for song_id in song_id_to_file.keys():
        nodes.add(song_id)

    nodes = sorted(nodes)
    node2idx = {n: i for i, n in enumerate(nodes)}

    # undirected graph: เพิ่มทั้งสองทิศ
    edge_pairs = []
    seen = set()
    for s, t in edges:
        u = node2idx[s]
        v = node2idx[t]
        if (u, v) not in seen:
            edge_pairs.append((u, v))
            seen.add((u, v))
        if (v, u) not in seen:
            edge_pairs.append((v, u))
            seen.add((v, u))

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    x = build_node_features(nodes)

    song_ids = []
    song_node_indices = []
    midi_files = []
    for song_id, midi_file in name_rows:
        if song_id in node2idx:
            song_ids.append(song_id)
            song_node_indices.append(node2idx[song_id])
            midi_files.append(midi_file)

    data = {
        "x": x,
        "edge_index": edge_index,
        "nodes": nodes,
        "node2idx": node2idx,
        "song_ids": song_ids,
        "song_node_indices": torch.tensor(song_node_indices, dtype=torch.long),
        "midi_files": midi_files,
    }
    return data


if __name__ == "__main__":
    base = Path("edgelist_file")
    data = build_graph_data(
        edgelist_dir=str(base),
        names_csv=str(base / "names.csv")
    )

    
    print("x shape:", data["x"].shape)
    print("edge_index shape:", data["edge_index"].shape)
    print("num songs:", len(data["song_ids"]))