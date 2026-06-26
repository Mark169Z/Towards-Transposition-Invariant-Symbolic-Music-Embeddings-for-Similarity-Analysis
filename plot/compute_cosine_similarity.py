from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# ---------------- config ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="song_vectors_csv", type=Path, default=Path("embeddings") / "song_vectors.csv")
ap.add_argument("--out_dir", type=Path, default=Path("plot") / "similarity_results_midi2vec")
args = ap.parse_args()

SONG_VECTORS_CSV = args.song_vectors_csv
OUT_DIR = args.out_dir
TOP_K = [1, 3, 5, 10]

OUT_DIR.mkdir(parents=True, exist_ok=True)

def base_song_from_key(key: str) -> str:
    """
    รองรับ key รูปแบบประมาณนี้:
    - Air_on_the_G_String_k0
    - Air_on_the_G_String_k0.mid
    - C:/.../Air_on_the_G_String_k0.mid
    - C:-kmutt-senior-try_embed-midi2vec-midi2vec-midi-Air_on_the_G_String-Air_on_the_G_String_k0
    """
    key = str(key).strip()

    # ตัด extension ถ้ามี
    key = re.sub(r"\.(mid|midi|abc)$", "", key, flags=re.IGNORECASE)

    # เอาชื่อท้ายสุดแบบปกติจาก path ก่อน
    tail = re.split(r"[\\/]", key)[-1]

    # กรณี flatten path เป็น - ทั้งหมด เช่น
    # C:-...-Air_on_the_G_String-Air_on_the_G_String_k0
    # ให้ดึงส่วนท้ายสุดก่อน _k\d+
    m = re.search(r"-([^-]+)_k\d+$", tail)
    if m:
        return m.group(1)

    # กรณีปกติ เช่น Air_on_the_G_String_k0
    m = re.search(r"(.+)_k\d+$", tail)
    if m:
        return m.group(1)

    # fallback
    return re.sub(r"_k\d+$", "", tail)

# ---------------- load embeddings ----------------
df = pd.read_csv(SONG_VECTORS_CSV)
df.columns = df.columns.str.strip()

KEY_COL = "emb_key" if "emb_key" in df.columns else "song_id"
vec_cols = [c for c in df.columns if re.fullmatch(r"v\d+", c)]
if not vec_cols:
    raise RuntimeError("No vector columns found (expected v0..vN).")
vec_cols = sorted(vec_cols, key=lambda x: int(x[1:]))

X = df[vec_cols].to_numpy(dtype=np.float32)

# ---------------- sanity checks ----------------
if np.isnan(X).any():
    raise RuntimeError("Found NaN in embedding vectors.")

if np.isinf(X).any():
    raise RuntimeError("Found Inf in embedding vectors.")

norms = np.linalg.norm(X, axis=1)
zero_idx = np.where(norms == 0)[0]
if len(zero_idx) > 0:
    bad_keys = df.iloc[zero_idx][KEY_COL].astype(str).tolist()
    raise RuntimeError(f"Found {len(zero_idx)} zero vectors, e.g. {bad_keys[:5]}")

df_index = pd.DataFrame({
    "key": df[KEY_COL].astype(str),
    "song": df[KEY_COL].astype(str).apply(base_song_from_key),
    "version": df[KEY_COL].astype(str).str.extract(r"_k(\d+)$", expand=False).fillna("unknown"),
})

if "midi_file" in df.columns:
    df_index["midi_file"] = df["midi_file"].astype(str)

songs = df_index["song"].values
N = len(df_index)

print(df_index[["key", "song", "version"]].head(10).to_string(index=False))
print()
print(df_index["song"].value_counts().head(10))
print()

print(f"Loaded {N} embeddings from {df_index['song'].nunique()} base songs")
print("Vector shape:", X.shape)

# ---------------- cosine similarity matrix ----------------
S = cosine_similarity(X)

# ใช้ตอนตัด self-match ออกจาก NN / Top-K
S_eval = S.copy()
np.fill_diagonal(S_eval, -np.inf)

# ---------------- intra-song vs inter-song ----------------
intra_sims = []
inter_sims = []

for i in range(N):
    for j in range(i + 1, N):
        if songs[i] == songs[j]:
            intra_sims.append(S[i, j])
        else:
            inter_sims.append(S[i, j])

intra_sims = np.array(intra_sims, dtype=np.float32)
inter_sims = np.array(inter_sims, dtype=np.float32)

summary = {
    "num_embeddings": int(N),
    "num_base_songs": int(df_index["song"].nunique()),
    "embedding_dim": int(X.shape[1]),
    "intra_mean": float(np.mean(intra_sims)) if len(intra_sims) else float("nan"),
    "intra_std": float(np.std(intra_sims)) if len(intra_sims) else float("nan"),
    "inter_mean": float(np.mean(inter_sims)) if len(inter_sims) else float("nan"),
    "inter_std": float(np.std(inter_sims)) if len(inter_sims) else float("nan"),
}

print("\n=== Intra-song Cosine Similarity (same base song, different keys) ===")
print("mean:", summary["intra_mean"])
print("std :", summary["intra_std"])

print("\n=== Inter-song Cosine Similarity (different base songs) ===")
print("mean:", summary["inter_mean"])
print("std :", summary["inter_std"])

# ---------------- nearest neighbor accuracy ----------------
nn_idx = np.argmax(S_eval, axis=1)
nn_acc = (songs == songs[nn_idx]).mean()
summary["nn_top1_accuracy"] = float(nn_acc)

print(f"\nNearest Neighbor Accuracy (Top-1): {nn_acc*100:.2f}%")

# ---------------- top-K hit rates ----------------
def topk_hit_rate(S_mat, songs_arr, k):
    idx_sorted = np.argsort(S_mat, axis=1)[:, ::-1]
    hits = []
    for i in range(len(songs_arr)):
        topk = idx_sorted[i, :k]
        hits.append(any(songs_arr[j] == songs_arr[i] for j in topk))
    return float(np.mean(hits))

for k in TOP_K:
    hit = topk_hit_rate(S_eval, songs, k)
    summary[f"top{k}_hit_rate"] = hit
    print(f"Top-{k} hit rate: {hit*100:.2f}%")

# ---------------- song-level similarity matrix ----------------
song_list = sorted(df_index["song"].unique())
song_to_idx = {s: i for i, s in enumerate(song_list)}

M = np.zeros((len(song_list), len(song_list)), dtype=np.float64)
C = np.zeros_like(M)

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        si = song_to_idx[songs[i]]
        sj = song_to_idx[songs[j]]
        M[si, sj] += S[i, j]
        C[si, sj] += 1

song_sim_df = pd.DataFrame(
    M / np.maximum(C, 1),
    index=song_list,
    columns=song_list
)

# ---------------- save outputs ----------------
df_index.to_csv(OUT_DIR / "embedding_index.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(S, index=df_index["key"], columns=df_index["key"]).to_csv(
    OUT_DIR / "embedding_similarity_matrix.csv", encoding="utf-8-sig"
)
song_sim_df.to_csv(OUT_DIR / "song_similarity_matrix.csv", encoding="utf-8-sig")
pd.DataFrame([summary]).to_csv(OUT_DIR / "similarity_summary.csv", index=False, encoding="utf-8-sig")

print("\nSaved results to:", OUT_DIR.resolve())