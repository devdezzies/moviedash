import numpy as np
import pandas as pd
from recommender import MovieRecommender


def precision_at_k(recommended_ids, relevant_ids, k):
    recommended_at_k = recommended_ids[:k]
    relevant_recommended = set(recommended_at_k) & set(relevant_ids)

    return len(relevant_recommended) / k


def recall_at_k(recommended_ids, relevant_ids, k):
    if len(relevant_ids) == 0:
        return None

    recommended_at_k = recommended_ids[:k]
    relevant_recommended = set(recommended_at_k) & set(relevant_ids)

    return len(relevant_recommended) / len(relevant_ids)


def ndcg_at_k(recommended_ids, relevant_ids, k):
    recommended_at_k = recommended_ids[:k]
    relevant_set = set(relevant_ids)

    dcg = 0.0

    for i, movie_id in enumerate(recommended_at_k):
        if movie_id in relevant_set:
            dcg += 1 / np.log2(i + 2)

    ideal_relevant_count = min(len(relevant_ids), k)

    if ideal_relevant_count == 0:
        return None

    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_relevant_count))

    return dcg / idcg


def temporal_train_test_split(user_ratings, train_ratio=0.8):
    user_ratings = user_ratings.sort_values("timestamp")

    split_idx = int(len(user_ratings) * train_ratio)

    train = user_ratings.iloc[:split_idx]
    test = user_ratings.iloc[split_idx:]

    return train, test


def evaluate(k=10, min_ratings=20):
    recommender = MovieRecommender()
    ratings = recommender.ratings

    active_users = (
        ratings.groupby("userId")
        .size()
        .reset_index(name="count")
    )

    active_users = active_users[active_users["count"] >= min_ratings]["userId"].tolist()

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for user_id in active_users:
        user_ratings = ratings[ratings["userId"] == user_id]

        train_user, test_user = temporal_train_test_split(user_ratings)

        liked_train_items = train_user[train_user["rating"] >= 4.0]["movieId"].tolist()

        if len(liked_train_items) == 0:
            continue

        relevant_test_items = test_user[test_user["rating"] >= 4.0]["movieId"].tolist()

        if len(relevant_test_items) == 0:
            continue

        train_ratings = ratings[ratings["userId"] != user_id]
        train_ratings = pd.concat([train_ratings, train_user], ignore_index=True)

        recommendations = recommender.recommend(
            user_id=user_id,
            top_n=k,
            ratings_df=train_ratings
        )

        if recommendations.empty or "movieId" not in recommendations.columns:
            continue

        recommended_ids = recommendations["movieId"].tolist()

        if len(recommended_ids) == 0:
            continue

        precision = precision_at_k(recommended_ids, relevant_test_items, k)
        recall = recall_at_k(recommended_ids, relevant_test_items, k)
        ndcg = ndcg_at_k(recommended_ids, relevant_test_items, k)

        precision_scores.append(precision)

        if recall is not None:
            recall_scores.append(recall)

        if ndcg is not None:
            ndcg_scores.append(ndcg)

    results = {
        "Precision@10": np.mean(precision_scores),
        "Recall@10": np.mean(recall_scores),
        "NDCG@10": np.mean(ndcg_scores),
        "Evaluated Users": len(precision_scores)
    }

    return results


if __name__ == "__main__":
    results = evaluate(k=10, min_ratings=20)

    print("Evaluation Results")
    print("==================")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")