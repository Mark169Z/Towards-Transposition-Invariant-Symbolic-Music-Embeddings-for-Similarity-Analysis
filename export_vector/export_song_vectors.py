import argparse
import os
import numpy as np
import pandas as pd
import csv
from gensim.models import KeyedVectors
import re  
def read_names_robust(csv_path: str) -> pd.DataFrame:
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # id,filename

        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue

            # Expect: [id, filename]
            # If a row has more than 2 fields, we join everything after the first back into filename
            song_id = row[0].strip()
            filename = ",".join(row[1:]).strip()

            # Strip surrounding quotes if present
            if len(filename) >= 2 and filename[0] == '"' and filename[-1] == '"':
                filename = filename[1:-1]

            rows.append({"id": song_id, "filename": filename})

    return pd.DataFrame(rows)

def load_embeddings(emb_path: str) -> KeyedVectors:
    # embedding.bin ของคุณเป็น word2vec TEXT format
    # (บรรทัดแรก: "6032 100")
    return KeyedVectors.load_word2vec_format(emb_path, binary=False)

def to_embedding_song_key(song_id: str) -> str:
    """
    Convert names.csv id like:
      -Air_on_the_G_String/Air_on_the_G_String_k0.mid
    to embedding key like:
      -Air_on_the_G_String-Air_on_the_G_String_k0
    """
    s = str(song_id)
    s = s.replace("\\", "/")
    if s.lower().endswith(".mid"):
        s = s[:-4]
    if s.lower().endswith(".midi"):
        s = s[:-5]
    s = s.replace("/", "-")
    return s

def key_from_filename(midi_path: str) -> str:
    folder = os.path.basename(os.path.dirname(midi_path))
    base = os.path.splitext(os.path.basename(midi_path))[0]

    # หา _kX จากท้ายชื่อไฟล์ เช่น ..._k0, ..._k11
    m = re.search(r"_k(\d+)$", base)
    if not m:
        # ถ้าหาไม่ได้ ก็ fallback เป็นรูปแบบเดิม
        return f"-{folder}-{base}"

    k = m.group(1)
    # รูปแบบที่ embedding ใช้ (จากที่คุณเช็ก key จริง): -folder-folder_kX
    return f"-{folder}-{folder}_k{k}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", required=True, help="Path to names.csv (id,filename).")
    parser.add_argument("--emb", required=True, help="Path to embeddings file (.bin or .vec/.txt).")
    parser.add_argument("--out", default="song_vectors.csv", help="Output CSV path.")
    parser.add_argument("--id_col", default="id", help="Column name for song node id in names.csv.")
    parser.add_argument("--file_col", default="filename", help="Column name for midi filename in names.csv.")
    args = parser.parse_args()

    names = read_names_robust(args.names)
    if args.id_col not in names.columns or args.file_col not in names.columns:
        raise ValueError(
            f"names.csv must contain columns '{args.id_col}' and '{args.file_col}'. "
            f"Found: {list(names.columns)}"
        )

    kv = load_embeddings(args.emb)
    keys = list(kv.key_to_index.keys()) if hasattr(kv, "key_to_index") else list(kv.vocab.keys())
    print("[DEBUG] first 20 keys in embedding:", keys[:20])
    dim = kv.vector_size

    rows = []
    missing = []

    # ids in your names.csv look like: -Minuet_in_G_k0, -Minuet_in_G_k1, ...
    # embeddings keys might be exactly that, or sometimes stringified differently.
    # We'll try a few fallbacks.
    for _, r in names.iterrows():
        song_id = str(r[args.id_col])
        midi_path = str(r[args.file_col])

        key = str(song_id).strip()   # id ใน names_emb.csv = key ใน embedding.bin อยู่แล้ว

        vec = None
        if key in kv:
            vec = kv[key]

        if vec is None:
            missing.append(song_id)
            continue
        # =========================

        row = {
            "song_id": song_id,
            "emb_key": key,
            "midi_file": midi_path,
        }

        for i in range(dim):
            row[f"v{i}"] = float(vec[i])

        rows.append(row)


    out_df = pd.DataFrame(rows)

    out_df = pd.DataFrame(rows)

    # ---------------- save CSV ----------------
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] Wrote {len(out_df)} song vectors to: {args.out}")

    # ---------------- save NPY ----------------
    vec_cols = [c for c in out_df.columns if c.startswith("v")]

    vectors = out_df[vec_cols].values.astype(np.float32)
    meta = out_df[["song_id", "emb_key", "midi_file"]].to_dict("records")

    npy_path = os.path.splitext(args.out)[0] + ".npy"

    np.save(
        npy_path,
        {
            "vectors": vectors,
            "meta": meta
        }
    )

    print(f"[OK] Saved numpy file to: {npy_path}")
    print(f"[INFO] Vector shape: {vectors.shape}")
    print(f"[INFO] Embedding dim: {dim}")

    if missing:
        print(f"[WARN] Missing {len(missing)} song ids in embedding file. First 10:")
        for x in missing[:10]:
            print("  ", x)
        print("[HINT] If all are missing, your embedding keys may not be song ids (e.g., numeric ids).")


if __name__ == "__main__":
    main()
