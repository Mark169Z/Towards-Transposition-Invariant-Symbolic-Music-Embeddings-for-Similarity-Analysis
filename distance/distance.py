from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


# ---------------- config ----------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

EMB_ROOT = PROJECT_ROOT / "embeddings"
OUT_DIR = BASE_DIR
TOP_K = [1, 3, 5, 10]

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- load embeddings ----------------
npy_path = EMB_ROOT / "song_vectors.npy"

if not npy_path.exists():
    raise RuntimeError(f"File not found: {npy_path}")

data = np.load(npy_path, allow_pickle=True).item()

X = data["vectors"]
meta = data["meta"]

df = pd.DataFrame(meta)

df["version"] = df["song_id"]
df["song"] = (
    df["song_id"]
    .astype(str)
    .str.replace(r"^-", "", regex=True)
    .str.replace(r"_k\d+$", "", regex=True)
)

if "midi_file" in df.columns:
    df["path"] = df["midi_file"]
else:
    df["path"] = ""

songs = df["song"].values
N = len(df)

print(f"Loaded {N} embeddings from {df['song'].nunique()} songs")
print(f"Vector shape: {X.shape}")


# ==========================================================
# ------------- METRIC COMPUTATION -------------------------
# ==========================================================

S_cosine = cosine_similarity(X)
np.fill_diagonal(S_cosine, np.nan)

D_euclidean = pairwise_distances(X, metric="euclidean")
D_manhattan = pairwise_distances(X, metric="manhattan")

S_dot = X @ X.T
np.fill_diagonal(S_dot, np.nan)

X_binary = (X > 0).astype(int)
D_hamming = pairwise_distances(X_binary, metric="hamming")

D_jaccard = pairwise_distances(X_binary, metric="jaccard")
S_jaccard = 1 - D_jaccard


# ==========================================================
# ----------- Intra / Inter Helper Function ----------------
# ==========================================================

def compute_intra_inter(matrix, songs):
    intra = []
    inter = []

    for i in range(N):
        for j in range(i + 1, N):
            value = matrix[i, j]
            if songs[i] == songs[j]:
                intra.append(value)
            else:
                inter.append(value)

    intra = np.array(intra)
    inter = np.array(inter)

    return {
        "intra_mean": float(np.nanmean(intra)),
        "intra_std": float(np.nanstd(intra)),
        "inter_mean": float(np.nanmean(inter)),
        "inter_std": float(np.nanstd(inter)),
    }


# ==========================================================
# ---------------- Metric Summary --------------------------
# ==========================================================

summary = {
    "num_embeddings": N,
    "num_songs": df["song"].nunique(),
}

metrics = {
    "cosine": S_cosine,
    "dot_product": S_dot,
    "euclidean": D_euclidean,
    "manhattan": D_manhattan,
    "hamming": D_hamming,
    "jaccard": S_jaccard,
}

for name, matrix in metrics.items():
    print(f"\n=== {name.upper()} ===")
    stats = compute_intra_inter(matrix, songs)
    print("Intra mean:", stats["intra_mean"])
    print("Inter mean:", stats["inter_mean"])

    summary[f"{name}_intra_mean"] = stats["intra_mean"]
    summary[f"{name}_intra_std"] = stats["intra_std"]
    summary[f"{name}_inter_mean"] = stats["inter_mean"]
    summary[f"{name}_inter_std"] = stats["inter_std"]


# ==========================================================
# ---------------- Nearest Neighbor (Cosine) ---------------
# ==========================================================

S_nn = S_cosine.copy()
np.fill_diagonal(S_nn, -np.inf)

nn_idx = np.argmax(S_nn, axis=1)
nn_acc = (songs == songs[nn_idx]).mean()

summary["cosine_nn_top1_accuracy"] = float(nn_acc)

print(f"\nNearest Neighbor Accuracy (Cosine Top-1): {nn_acc*100:.2f}%")


# ==========================================================
# ---------------- Top-K Hit Rate (Cosine) -----------------
# ==========================================================

def topk_hit_rate(S, songs, k):
    hits = []
    idx_sorted = np.argsort(S, axis=1)[:, ::-1]

    for i in range(len(songs)):
        topk = idx_sorted[i, :k]
        hits.append(any(songs[j] == songs[i] for j in topk))

    return np.mean(hits)


for k in TOP_K:
    hit = topk_hit_rate(S_cosine, songs, k)
    summary[f"cosine_top{k}_hit_rate"] = float(hit)
    print(f"Cosine Top-{k} hit rate: {hit*100:.2f}%")


# ==========================================================
# ---------------- Song-level Matrix (Cosine) --------------
# ==========================================================

song_list = sorted(df["song"].unique())
song_to_idx = {s: i for i, s in enumerate(song_list)}

M = np.zeros((len(song_list), len(song_list)))
C = np.zeros_like(M)

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        si = song_to_idx[songs[i]]
        sj = song_to_idx[songs[j]]
        M[si, sj] += S_cosine[i, j]
        C[si, sj] += 1

M = M / np.maximum(C, 1)

song_sim_df = pd.DataFrame(M, index=song_list, columns=song_list)


# ==========================================================
# ---------------- Save Outputs ----------------------------
# ==========================================================

df.to_csv(OUT_DIR / "embedding_index.csv", index=False, encoding="utf-8-sig")
song_sim_df.to_csv(OUT_DIR / "song_similarity_matrix.csv", encoding="utf-8-sig")
pd.DataFrame([summary]).to_csv(OUT_DIR / "similarity_summary.csv", index=False, encoding="utf-8-sig")

print("\nSaved results to:", OUT_DIR.resolve())