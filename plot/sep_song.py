import pandas as pd
import numpy as np
import os
import re
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1) โหลดข้อมูล
# =========================
df = pd.read_csv(r"C:\kmutt\senior\try_embed\midi2vec\midi2vec\embeddings\song_vectors.csv")
df.columns = df.columns.str.strip()

# =========================
# 2) ดึงชื่อเพลงหลัก
# =========================
def extract_song_name(path_str):
    fname = os.path.basename(str(path_str))   # เช่น Air_on_the_G_String_k0.mid
    fname = os.path.splitext(fname)[0]        # Air_on_the_G_String_k0
    song = re.sub(r"_k\d+$", "", fname)       # Air_on_the_G_String
    return song

df["song"] = df["song_id"].apply(extract_song_name)

# =========================
# 3) เลือกคอลัมน์ embedding
# =========================
vector_cols = [col for col in df.columns if re.fullmatch(r"v\d+", col)]
vectors = df[vector_cols].astype(float).values

# =========================
# 4) cosine similarity matrix
# =========================
cos_sim_matrix = cosine_similarity(vectors)

# =========================
# 5) สรุปผลแยกทีละเพลง
# =========================
song_results = []

for song in sorted(df["song"].unique()):
    idx = df.index[df["song"] == song].tolist()
    
    sims = []
    pair_names = []
    
    for i, j in combinations(idx, 2):
        sim = cos_sim_matrix[i, j]
        sims.append(sim)
        pair_names.append((df.loc[i, "song_id"], df.loc[j, "song_id"], sim))
    
    if len(sims) > 0:
        song_results.append({
            "song": song,
            "num_versions": len(idx),
            "num_pairs": len(sims),
            "mean_cosine_similarity": np.mean(sims),
            "min_cosine_similarity": np.min(sims),
            "max_cosine_similarity": np.max(sims),
            "std_cosine_similarity": np.std(sims)
        })
    else:
        song_results.append({
            "song": song,
            "num_versions": len(idx),
            "num_pairs": 0,
            "mean_cosine_similarity": np.nan,
            "min_cosine_similarity": np.nan,
            "max_cosine_similarity": np.nan,
            "std_cosine_similarity": np.nan
        })

song_summary_df = pd.DataFrame(song_results)

# =========================
# 6) เรียงจาก mean มากไปน้อย
# =========================
song_summary_df = song_summary_df.sort_values(
    by="mean_cosine_similarity",
    ascending=False
).reset_index(drop=True)

# =========================
# 7) แสดงผล
# =========================
print("\n=== Per-song Cosine Similarity Summary ===")
print(song_summary_df.to_string(index=False))

# =========================
# 8) บันทึกผล
# =========================
song_summary_df.to_csv("per_song_cosine_summary.csv", index=False, encoding="utf-8-sig")
print("\nSaved: per_song_cosine_summary.csv")