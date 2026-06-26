import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def extract_base_song(name: str) -> str:
    """
    Example:
    - Air_on_the_G_String_k0.mid -> Air_on_the_G_String
    - some/folder/Air_on_the_G_String_k11.mid -> Air_on_the_G_String
    """
    base = os.path.basename(str(name))
    base = os.path.splitext(base)[0]
    base = re.sub(r"_k\d+$", "", base)
    return base


def get_vector_columns(df: pd.DataFrame):
    vec_cols = [c for c in df.columns if re.fullmatch(r"v\d+", str(c))]
    if not vec_cols:
        raise ValueError("No embedding columns found. Expected columns like v0, v1, v2, ...")
    return vec_cols


def prepare_dataframe(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    label_col = "midi_file" if "midi_file" in df.columns else "song_id"
    if label_col not in df.columns:
        raise ValueError("Expected either 'midi_file' or 'song_id' column in CSV.")

    df["base_song"] = df[label_col].apply(extract_base_song)

    # map base song -> numeric label 1..N
    unique_songs = sorted(df["base_song"].unique())
    song_to_num = {song: i + 1 for i, song in enumerate(unique_songs)}
    df["song_num"] = df["base_song"].map(song_to_num)

    return df


def make_output_dir(output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_coords_csv(df_plot: pd.DataFrame, out_csv: Path):
    cols = ["base_song", "song_num", "x", "y"]
    extra = [c for c in ["song_id", "midi_file", "emb_key"] if c in df_plot.columns]
    df_plot[extra + cols].to_csv(out_csv, index=False, encoding="utf-8")


def plot_grouped_scatter(df_plot: pd.DataFrame, title: str, out_png: Path):
    unique_songs = sorted(df_plot["base_song"].unique())
    cmap = plt.cm.get_cmap("gist_ncar", len(unique_songs))
    color_map = {song: cmap(i) for i, song in enumerate(unique_songs)}

    plt.figure(figsize=(14, 10))

    # plot all points grouped by base song
    for song in unique_songs:
        sub = df_plot[df_plot["base_song"] == song]
        plt.scatter(
            sub["x"],
            sub["y"],
            s=70,
            alpha=0.75,
            color=color_map[song],
            edgecolors="white",
            linewidths=0.8,
        )

    # put song number label at centroid of each group
    centroids = (
        df_plot.groupby(["base_song", "song_num"], as_index=False)[["x", "y"]]
        .mean()
        .sort_values("song_num")
    )

    for _, row in centroids.iterrows():
        plt.text(
            row["x"],
            row["y"],
            str(int(row["song_num"])),
            fontsize=17,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
        )

    plt.title(title, fontsize=22)
    plt.xlabel("dim 1", fontsize=16)
    plt.ylabel("dim 2", fontsize=16)
    plt.tick_params(axis="both", labelsize=12)

    # mimic your current style
    plt.text(
        0.01,
        0.01,
        f"Legend hidden (songs={len(unique_songs)})",
        transform=plt.gca().transAxes,
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def run_pca(X: np.ndarray):
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(X)
    explained = reducer.explained_variance_ratio_.sum()
    title = f"midi2vec PCA 2D (explained var: {explained:.3f})"
    return coords, title


def run_tsne(X: np.ndarray, perplexity: float = 30.0):
    # safe perplexity for smaller datasets
    n = len(X)
    if n <= 3:
        raise ValueError("Need at least 4 samples for t-SNE.")
    perplexity = min(perplexity, max(2.0, (n - 1) / 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    coords = reducer.fit_transform(X)
    title = "midi2vec t-SNE 2D"
    return coords, title


def run_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, len(X) - 1)),
        min_dist=min_dist,
        metric="euclidean",
        random_state=42,
    )
    coords = reducer.fit_transform(X)
    title = "midi2vec UMAP 2D"
    return coords, title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="input_csv",
        default="embeddings/song_vectors.csv",
        help="Input CSV containing embeddings",
    )
    ap.add_argument(
        "--out_dir",
        default="plot/plots_embeddings_midi2vec",
        help="Directory to save PNG and CSV outputs",
    )
    ap.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    ap.add_argument(
        "--umap_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors",
    )
    ap.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist",
    )
    args = ap.parse_args()

    df = prepare_dataframe(args.input_csv)
    vec_cols = get_vector_columns(df)
    X = df[vec_cols].to_numpy(dtype=np.float32)

    out_dir = make_output_dir(args.out_dir)

    # PCA
    pca_coords, pca_title = run_pca(X)
    df_pca = df.copy()
    df_pca["x"] = pca_coords[:, 0]
    df_pca["y"] = pca_coords[:, 1]
    save_coords_csv(df_pca, out_dir / "pca_2d_coords.csv")
    plot_grouped_scatter(df_pca, pca_title, out_dir / "pca_2d.png")
    print(f"Saved: {out_dir / 'pca_2d.png'}")
    print(f"Saved: {out_dir / 'pca_2d_coords.csv'}")

    # t-SNE
    tsne_coords, tsne_title = run_tsne(X, perplexity=args.tsne_perplexity)
    df_tsne = df.copy()
    df_tsne["x"] = tsne_coords[:, 0]
    df_tsne["y"] = tsne_coords[:, 1]
    save_coords_csv(df_tsne, out_dir / "tsne_2d_coords.csv")
    plot_grouped_scatter(df_tsne, tsne_title, out_dir / "tsne_2d.png")
    print(f"Saved: {out_dir / 'tsne_2d.png'}")
    print(f"Saved: {out_dir / 'tsne_2d_coords.csv'}")

    # UMAP
    umap_coords, umap_title = run_umap(
        X,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    df_umap = df.copy()
    df_umap["x"] = umap_coords[:, 0]
    df_umap["y"] = umap_coords[:, 1]
    save_coords_csv(df_umap, out_dir / "umap_2d_coords.csv")
    plot_grouped_scatter(df_umap, umap_title, out_dir / "umap_2d.png")
    print(f"Saved: {out_dir / 'umap_2d.png'}")
    print(f"Saved: {out_dir / 'umap_2d_coords.csv'}")


if __name__ == "__main__":
    main()