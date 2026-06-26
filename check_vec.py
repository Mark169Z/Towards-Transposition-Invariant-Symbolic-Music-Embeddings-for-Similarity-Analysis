from gensim.models import KeyedVectors
import pandas as pd

vec_path = r"C:\kmutt\senior\try_embed\midi2vec\midi2vec\embeddings\minuet.bin"
kv = KeyedVectors.load_word2vec_format(vec_path, binary=True)

names = pd.read_csv(r"C:\kmutt\senior\try_embed\midi2vec\midi2vec\edgelist_file\names.csv")
song_id = names.loc[0, "id"]

print("ID เพลง:", song_id)
print("shape เวกเตอร์:", kv.get_vector(song_id).shape)
print("10 มิติแรก:", kv.get_vector(song_id)[:10])
