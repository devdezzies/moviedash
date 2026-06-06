import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"

RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
EMBEDDINGS_PATH = os.path.join(ARTIFACT_DIR, "movie_embeddings.npy")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "movie_metadata.pkl")


class MovieRecommender:
    def __init__(self):
        self.ratings = pd.read_csv(RATINGS_PATH)
        self.movies = pd.read_pickle(METADATA_PATH)
        self.embeddings = np.load(EMBEDDINGS_PATH)

        self.movie_id_to_index = {
            movie_id: idx
            for idx, movie_id in enumerate(self.movies["movieId"].tolist())
        }

    def is_valid_user(self, user_id):
        return user_id in set(self.ratings["userId"].unique())

    def is_active_user(self, user_id, min_ratings=20):
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        return len(user_ratings) >= min_ratings

    def get_user_summary(self, user_id):
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= 4.0]

        liked_movies = liked.merge(self.movies, on="movieId", how="left")

        all_genres = []
        for genres in liked_movies["genres"].dropna():
            all_genres.extend(genres.split("|"))

        top_genres = (
            pd.Series(all_genres)
            .value_counts()
            .head(5)
            .index
            .tolist()
        )

        return {
            "total_ratings": len(user_ratings),
            "liked_movies": len(liked),
            "top_genres": top_genres,
            "liked_movie_table": liked_movies[["title", "genres", "rating"]]
                .sort_values("rating", ascending=False)
                .head(10)
        }

    def build_user_embedding(self, user_id, ratings_df=None):
        if ratings_df is None:
            ratings_df = self.ratings

        user_ratings = ratings_df[ratings_df["userId"] == user_id]

        liked = user_ratings[user_ratings["rating"] >= 4.0]
        disliked = user_ratings[user_ratings["rating"] <= 2.5]

        if liked.empty:
            return None

        liked_vectors = []
        liked_weights = []

        for _, row in liked.iterrows():
            movie_id = row["movieId"]

            if movie_id not in self.movie_id_to_index:
                continue

            idx = self.movie_id_to_index[movie_id]
            liked_vectors.append(self.embeddings[idx])
            liked_weights.append(row["rating"] - 3.0)

        if not liked_vectors:
            return None

        liked_vectors = np.array(liked_vectors)
        liked_weights = np.array(liked_weights)

        positive_profile = np.average(
            liked_vectors,
            axis=0,
            weights=liked_weights
        )

        if not disliked.empty:
            disliked_vectors = []

            for _, row in disliked.iterrows():
                movie_id = row["movieId"]

                if movie_id not in self.movie_id_to_index:
                    continue

                idx = self.movie_id_to_index[movie_id]
                disliked_vectors.append(self.embeddings[idx])

            if disliked_vectors:
                negative_profile = np.mean(np.array(disliked_vectors), axis=0)
                user_embedding = positive_profile - 0.5 * negative_profile
            else:
                user_embedding = positive_profile
        else:
            user_embedding = positive_profile

        norm = np.linalg.norm(user_embedding)

        if norm == 0:
            return None

        user_embedding = user_embedding / norm

        return user_embedding

    def recommend(self, user_id, top_n=10, ratings_df=None):
        if ratings_df is None:
            ratings_df = self.ratings

        empty_result = pd.DataFrame(
            columns=["movieId", "title", "genres", "similarity"]
        )

        user_embedding = self.build_user_embedding(user_id, ratings_df)

        if user_embedding is None:
            return empty_result

        similarities = cosine_similarity(
            user_embedding.reshape(1, -1),
            self.embeddings
        )[0]

        result = self.movies.copy()
        result["similarity"] = similarities

        watched_movie_ids = set(
            ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
        )

        result = result[~result["movieId"].isin(watched_movie_ids)]

        if result.empty:
            return empty_result

        result = result.sort_values("similarity", ascending=False)

        return result[["movieId", "title", "genres", "similarity"]].head(top_n)