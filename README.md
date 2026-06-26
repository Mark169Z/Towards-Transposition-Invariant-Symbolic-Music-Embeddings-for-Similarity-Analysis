# midi2vec / GraphSAGE

A toolkit for generating **symbolic MIDI embeddings** from MIDI files using either:

* **Legacy MIDI2vec pipeline** based on node2vec
* **GraphSAGE pipeline** based on graph neural networks

This repository extends the original **MIDI2vec** project by adding a GraphSAGE-based embedding pipeline while preserving compatibility with the original workflow.

---

# Citation

This repository builds upon the original **MIDI2vec** framework by Lisena et al. and extends it with relative-pitch graph representations and a **GraphSAGE-based embedding pipeline**.

If you use the original MIDI2vec methodology, please cite the following paper:

```bibtex
@article{lisena2022midi2vec,
  title={MIDI2vec: Learning MIDI Embeddings for Reliable Prediction of Symbolic Music Metadata},
  author={Lisena, Pasquale and Mero{\~n}o-Pe{\~n}uela, Albert and Troncy, Rapha{\"e}l},
  journal={Semantic Web},
  volume={13},
  number={3},
  pages={357--377},
  year={2022},
  publisher={IOS Press},
  doi={10.3233/SW-210446}
}
```

**Paper**

> Lisena, P., Meroño-Peñuela, A., & Troncy, R. (2022). *MIDI2vec: Learning MIDI Embeddings for Reliable Prediction of Symbolic Music Metadata*. Semantic Web, 13(3), 357–377.

* DOI: https://doi.org/10.3233/SW-210446
* Original repository: https://github.com/pasqLisena/midi2vec
* Original experiments: https://github.com/pasqLisena/midi-embs
* Pre-trained embeddings: https://zenodo.org/record/5082300

---

# Project Structure

```text
.
├── midi/               # Input MIDI files
├── midi2edgelist/      # MIDI → edgelist converter (Node.js)
├── edgelist2vec/       # Original MIDI2vec embedding pipeline
├── graphsage/          # GraphSAGE embedding pipeline
├── embeddings/         # Generated models and song vectors
├── edgelist_file/      # Generated edgelists and mapping files
├── plot/               # Visualization and similarity analysis
└── distance/           # Additional distance metrics
```

---

# Getting Started

The project is designed to run entirely with **Docker**, so no local Python virtual environment is required.

## Build the Docker image

```bash
docker compose build
```

## Start the container

```bash
docker compose run --rm midi2vec bash
```

All commands below assume you are running inside the container.

---

# Prepare MIDI Files

Place your `.mid` files inside the `midi/` directory.

Example:

```text
midi/
└── Minuet_in_G_Major/
    └── Minuet_in_G_Major_k0.mid
```

---

# Convert MIDI to Edgelists

Both workflows share the same preprocessing step.

```bash
cd midi2edgelist
node index.js -i ../midi -o ../edgelist_file
cd ..
```

---

# Workflow 1 — Legacy MIDI2vec

This workflow reproduces the original MIDI2vec pipeline proposed by Lisena et al. (2022), using **node2vec** to learn graph embeddings.

```text
MIDI
   ↓
Edgelists
   ↓
node2vec
   ↓
Song Embeddings
```

## Train embeddings

```bash
python edgelist2vec/embed.py \
    -i edgelist_file \
    -o embeddings/embedding.bin
```

## Export song vectors

```bash
python export_vector/make_names_emb_from_midi.py \
    --midi_root midi \
    --emb embeddings/embedding.bin \
    --out edgelist_file/names_emb.csv

python export_vector/export_song_vectors.py \
    --names edgelist_file/names_emb.csv \
    --emb embeddings/embedding.bin \
    --out embeddings/song_vectors.csv
```

---

# Workflow 2 — GraphSAGE (Recommended)

This repository introduces a GraphSAGE-based alternative to the original MIDI2vec pipeline. Instead of using node2vec, GraphSAGE learns graph representations directly from the generated MIDI graphs.

```text
MIDI
   ↓
Edgelists
   ↓
GraphSAGE
   ↓
Song Embeddings
```

## Train GraphSAGE

```bash
python graphsage/train.py \
    --input edgelist_file \
    --out_dir embeddings
```

To specify the embedding dimension:

```bash
python graphsage/train.py \
    --input edgelist_file \
    --out_dir embeddings \
    --out_dim 100
```

Generated outputs:

```text
embeddings/
├── song_vectors.csv
├── song_vectors.npy
├── graphsage_model.pt
└── debug_tables/
```

---

# Visualization & Analysis

The following scripts work for **both** workflows.

## Verify exported vectors

```bash
python -c "import pandas as pd; df=pd.read_csv('embeddings/song_vectors.csv'); print(df.shape); print(df.columns[:12])"
```

## Plot embeddings

```bash
python plot/plot_embeddings_2d.py \
    --in embeddings/song_vectors.csv \
    --out_dir plot/plots_embeddings
```

Generated figures:

* PCA
* t-SNE
* UMAP

## Compute cosine similarity

```bash
python plot/compute_cosine_similarity.py \
    --in embeddings/song_vectors.csv \
    --out_dir plot/similarity_results
```

## Additional distance metrics

```bash
python distance/distance.py
```

---

# Quick Start (Recommended)

```bash
docker compose build
docker compose run --rm midi2vec bash

cd midi2edgelist
node index.js -i ../midi -o ../edgelist_file
cd ..

python graphsage/train.py \
    --input edgelist_file \
    --out_dir embeddings

python plot/plot_embeddings_2d.py \
    --in embeddings/song_vectors.csv \
    --out_dir plot/plots_embeddings

python plot/compute_cosine_similarity.py \
    --in embeddings/song_vectors.csv \
    --out_dir plot/similarity_results
```

---

# Notes

* The **Legacy MIDI2vec** workflow depends on:

  * `edgelist2vec/embed.py`
  * `make_names_emb_from_midi.py`
  * `export_song_vectors.py`

* The **GraphSAGE** workflow exports song vectors directly and does not require the legacy export scripts.

* All paths are relative to the repository root.

* Docker is the recommended execution environment.

---

# Credits

This project builds upon the original **MIDI2vec** framework developed by **Pasquale Lisena**, **Albert Meroño-Peñuela**, and **Raphaël Troncy**.

The original implementation converts symbolic MIDI files into graph representations and learns embeddings using **node2vec**. This repository extends that framework with a **GraphSAGE-based graph neural network pipeline** while maintaining compatibility with the original preprocessing workflow.
