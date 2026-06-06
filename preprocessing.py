import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"

MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
TAGS_PATH = os.path.join(DATA_DIR, "tags.csv")

EMBEDDINGS_PATH = os.path.join(ARTIFACT_DIR, "movie_embeddings.npy")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "movie_metadata.pkl")


def load_and_prepare_movies():
    movies = pd.read_csv(MOVIES_PATH)
    tags = pd.read_csv(TAGS_PATH)

    tags["tag"] = tags["tag"].fillna("").astype(str)

    grouped_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(sorted(set(x))))
        .reset_index()
    )

    movies = movies.merge(grouped_tags, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")

    movies["genres_clean"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)

    movies["item_text"] = (
        movies["title"].fillna("") + " " +
        movies["genres_clean"] + " " +
        movies["tag"]
    )

    return movies


def generate_embeddings():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    movies = load_and_prepare_movies()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generating movie embeddings...")
    embeddings = model.encode(
        movies["item_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    movies.to_pickle(METADATA_PATH)

    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    generate_embeddings()