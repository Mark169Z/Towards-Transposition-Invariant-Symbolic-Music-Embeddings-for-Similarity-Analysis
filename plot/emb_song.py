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

# กันชื่อคอลัมน์มี space แปลก ๆ
df.columns = df.columns.str.strip()

# =========================
# 2) สร้างชื่อเพลงหลัก
# ตัวอย่าง:
# -Air_on_the_G_String/Air_on_the_G_String_k0.mid
# -> Air_on_the_G_String
# =========================
def extract_song_name(path_str):
    fname = os.path.basename(str(path_str))      # Air_on_the_G_String_k0.mid
    fname = os.path.splitext(fname)[0]           # Air_on_the_G_String_k0
    song = re.sub(r"_k\d+$", "", fname)          # Air_on_the_G_String
    return song

df["song"] = df["song_id"].apply(extract_song_name)

# =========================
# 3) เลือกเฉพาะคอลัมน์ vector
# =========================
vector_cols = [col for col in df.columns if re.fullmatch(r"v\d+", col)]
vectors = df[vector_cols].astype(float).values

# =========================
# 4) คำนวณ cosine similarity matrix
# =========================
cos_sim_matrix = cosine_similarity(vectors)

# =========================
# 5) Intra-song similarity
# เพลงเดียวกัน ต่าง transpose
# =========================
intra_sims = []

for song in df["song"].unique():
    idx = df.index[df["song"] == song].tolist()
    for i, j in combinations(idx, 2):
        intra_sims.append(cos_sim_matrix[i, j])

print("=== Intra-song ===")
print("Count:", len(intra_sims))
print("Mean :", np.mean(intra_sims))
print("Min  :", np.min(intra_sims))
print("Max  :", np.max(intra_sims))

# =========================
# 6) Inter-song similarity
# คนละเพลง
# =========================
inter_sims = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if df.loc[i, "song"] != df.loc[j, "song"]:
            inter_sims.append(cos_sim_matrix[i, j])

print("\n=== Inter-song ===")
print("Count:", len(inter_sims))
print("Mean :", np.mean(inter_sims))
print("Min  :", np.min(inter_sims))
print("Max  :", np.max(inter_sims))

# =========================
# 7) Top-1 Hit Rate
# หาเพื่อนบ้านที่ใกล้สุดของแต่ละเพลง
# แล้วดูว่าเป็นเพลงเดียวกันไหม
# =========================
correct = 0

for i in range(len(df)):
    sims = cos_sim_matrix[i].copy()
    sims[i] = -np.inf   # ตัดตัวเองออก
    nearest = np.argmax(sims)

    if df.loc[i, "song"] == df.loc[nearest, "song"]:
        correct += 1

top1 = correct / len(df)

print("\n=== Top-1 Hit Rate ===")
print(top1)

# =========================
# 8) บันทึกผล similarity รายคู่ (เผื่อเอาไปวิเคราะห์ต่อ)
# =========================
pair_results = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        pair_results.append({
            "song_i": df.loc[i, "song"],
            "song_j": df.loc[j, "song"],
            "file_i": df.loc[i, "song_id"],
            "file_j": df.loc[j, "song_id"],
            "cosine_similarity": cos_sim_matrix[i, j],
            "pair_type": "intra" if df.loc[i, "song"] == df.loc[j, "song"] else "inter"
        })

pair_df = pd.DataFrame(pair_results)
pair_df.to_csv("cosine_pair_results.csv", index=False, encoding="utf-8-sig")

print("\nSaved: cosine_pair_results.csv")