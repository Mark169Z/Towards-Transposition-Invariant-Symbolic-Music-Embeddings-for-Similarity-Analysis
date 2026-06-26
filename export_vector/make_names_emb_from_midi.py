import os
import re
import csv
from gensim.models import KeyedVectors
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--midi_root", type=str, default="midi")
ap.add_argument("--emb", type=str, default=str(Path("embeddings") / "embedding.bin"))
ap.add_argument("--out", type=str, default=str(Path("edgelist_file") / "names_emb.csv"))
args = ap.parse_args()

MIDI_ROOT = args.midi_root
EMB = args.emb
OUT = args.out


def norm(s: str) -> str:
    # เหลือเฉพาะ a-z 0-9 เพื่อตัดปัญหา + _ - space () , ฯลฯ
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def tail_song_key(k: str) -> str:
    # เอาเฉพาะส่วนท้ายของ key หลัง "...-midi-" ถ้ามี
    # เช่น C:-...-midi-Air_on-Air_on_k0  -> Air_on-Air_on_k0
    s = str(k)
    marker = "-midi-"
    i = s.lower().rfind(marker)
    if i != -1:
        return s[i + len(marker):]
    return s

def get_knum_from_stem(stem: str):
    m = re.search(r"_k(\d+)$", stem.lower())
    return m.group(1) if m else None

# โหลด embedding (ของคุณเป็น word2vec TEXT format)
kv = KeyedVectors.load_word2vec_format(EMB, binary=False)

# ดึงเฉพาะ key ที่เป็น "เพลง" (เริ่มด้วย - และลงท้าย _kตัวเลข)
all_keys = kv.key_to_index.keys() if hasattr(kv, "key_to_index") else kv.vocab.keys()
song_keys = [k for k in all_keys if re.search(r"_k\d+$", k)]
print("[INFO] song_keys in embedding:", len(song_keys))

# ทำ index: (knum, normalized key without _knum) -> full key
index = {}
for k in song_keys:
    kk = tail_song_key(k)  # <-- เพิ่มบรรทัดนี้
    m = re.search(r"_k(\d+)$", kk)
    knum = m.group(1)
    base = kk[: -(len("_k") + len(knum))]
    index[(knum, norm(base))] = k  # <-- เก็บค่าเดิม k ไว้ (full key)

rows = []
missing = []
total_midis = 0

for root, _, files in os.walk(MIDI_ROOT):
    for fn in files:
        if not fn.lower().endswith((".mid", ".midi")):
            continue

        total_midis += 1
        full = os.path.join(root, fn)
        folder = os.path.basename(os.path.dirname(full))
        stem = os.path.splitext(os.path.basename(full))[0]  # no .mid
        knum = get_knum_from_stem(stem)

        # ถ้าไฟล์ไม่มี _kX ก็ข้าม
        if knum is None:
            continue

        # candidates ที่เป็นไปได้ (ก่อน normalize)
        cand1 = f"-{folder}-{folder}"                 # แบบ -folder-folder_kX (ตัด _kX ไว้ทีหลัง)
        cand2 = f"-{folder}-{stem}"                   # แบบ -folder-stem
        # บางกรณี stem อาจยาวมาก ลองตัด _kX ออกก่อน
        stem_no_k = re.sub(r"_k\d+$", "", stem, flags=re.IGNORECASE)
        cand3 = f"-{folder}-{stem_no_k}"

        found_key = None
        for cand in (cand1, cand2, cand3):
            nk = (knum, norm(cand))
            if nk in index:
                found_key = index[nk]
                break

        if found_key is None:
            missing.append((full, folder, stem, knum, cand1, cand2, cand3))
            continue

        rows.append((found_key, full))

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "filename"])
    w.writerows(rows)

print("[OK] wrote:", len(rows), "rows to", OUT)
print("[INFO] total midi files found:", total_midis)
print("[WARN] missing:", len(missing))

# ถ้ายัง missing ให้พิมพ์ตัวอย่าง 10 อันแรกเพื่อดู pattern
if missing:
    print("\n[DEBUG] first 10 missing examples:")
    for (full, folder, stem, knum, c1, c2, c3) in missing[:10]:
        print("file  :", full)
        print("folder:", folder)
        print("stem  :", stem)
        print("knum  :", knum)
        print("cand1 :", c1)
        print("cand2 :", c2)
        print("cand3 :", c3)
        # ลองโชว์ key ใน embedding ที่มี knum เดียวกันและคล้าย folder (แบบหยาบ)
        sample = [k for k in song_keys if k.endswith(f"_k{knum}") and norm(folder) in norm(k)][:5]
        print("embed sample:", sample)
        print("---")
