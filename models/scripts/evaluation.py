import numpy as np
from typing import List, Dict


def precision_at_k(y_true: List[int], y_pred: List[int], k: int = 5) -> float:
    """
    Compute Precision@K.

    Precision@K measures the proportion of recommended items that are relevant.

    Args:
        y_true (List[int]): List of actual purchased product IDs.
        y_pred (List[int]): List of predicted recommended product IDs.
        k (int): The number of top-K recommendations to consider.

    Returns:
        float: Precision score at K.
    """
    actual_set = set(y_true)
    predicted_set = set(y_pred[:k])
    return len(actual_set & predicted_set) / k if k > 0 else 0.0


def recall_at_k(y_true: List[int], y_pred: List[int], k: int = 5) -> float:
    """
    Compute Recall@K.

    Recall@K measures how many of the actual purchased items were successfully recommended.

    Args:
        y_true (List[int]): List of actual purchased product IDs.
        y_pred (List[int]): List of predicted recommended product IDs.
        k (int): The number of top-K recommendations to consider.

    Returns:
        float: Recall score at K.
    """
    actual_set = set(y_true)
    predicted_set = set(y_pred[:k])
    return len(actual_set & predicted_set) / len(actual_set) if actual_set else 0.0


def dcg_at_k(y_true: List[int], y_pred: List[int], k: int = 5) -> float:
    """
    Compute Discounted Cumulative Gain (DCG) at K.

    Args:
        y_true (List[int]): List of actual purchased product IDs.
        y_pred (List[int]): List of predicted recommended product IDs.
        k (int): The number of top-K recommendations to consider.

    Returns:
        float: DCG score at K.
    """
    actual_set = set(y_true)
    dcg = 0.0
    for i, item in enumerate(y_pred[:k]):
        if item in actual_set:
            dcg += 1 / np.log2(i + 2)  # relevance / log2(rank+1)
    return dcg


def ndcg_at_k(y_true: List[int], y_pred: List[int], k: int = 5) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.

    Args:
        y_true (List[int]): List of actual purchased product IDs.
        y_pred (List[int]): List of predicted recommended product IDs.
        k (int): The number of top-K recommendations to consider.

    Returns:
        float: NDCG score at K.
    """
    dcg = dcg_at_k(y_true, y_pred, k)
    idcg = dcg_at_k(y_true, y_true, k)  # Ideal DCG (IDCG) - best possible ranking
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(y_true: Dict[int, List[int]], y_pred: Dict[int, List[int]], k: int = 5) -> Dict[str, float]:
    """
    Evaluate recommendation model performance using Precision@K, Recall@K, and NDCG@K.

    Args:
        y_true (Dict[int, List[int]]): A dictionary where keys are client IDs and values are lists of purchased product IDs.
        y_pred (Dict[int, List[int]]): A dictionary where keys are client IDs and values are lists of recommended product IDs.
        k (int): The number of top-K recommendations to consider.

    Returns:
        Dict[str, float]: A dictionary containing the average Precision, Recall, and NDCG scores.
    """
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for client in y_true.keys():
        actual_products = y_true.get(client, [])
        recommended_products = y_pred.get(client, [])

        precision_scores.append(precision_at_k(actual_products, recommended_products, k))
        recall_scores.append(recall_at_k(actual_products, recommended_products, k))
        ndcg_scores.append(ndcg_at_k(actual_products, recommended_products, k))

    return {
        "Precision": np.mean(precision_scores),
        "Recall": np.mean(recall_scores),
        "NDCG": np.mean(ndcg_scores)
    }


# Example Usage
if __name__ == "__main__":
    # Sample data for testing
    y_true_sample = {
        1: [101, 102, 103],
        2: [104, 105],
        3: [106, 107, 108, 109]
    }

    y_pred_sample = {
        1: [101, 104, 105, 107, 109],
        2: [103, 104, 106, 108, 110],
        3: [106, 107, 110, 111, 112]
    }

    results = evaluate_model(y_true_sample, y_pred_sample, k=5)
    print("Evaluation Results:", results)