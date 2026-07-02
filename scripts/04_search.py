import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

QUERY = "computer vision object recognition image classification neural networks"
RL_QUERY = "reinforcement learning Markov decision process policy optimization"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet")
embeddings = np.load("embeddings/embeddings.npy")


def encode_query(query: str) -> np.ndarray:
    text = f"{query} [SEP] {query}"

    return model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

def print_pinecone_results(title: str, results):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, match in enumerate(results["matches"], start=1):
        metadata = match["metadata"]

        print(f"\n{i}. Score: {match['score']:.4f}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Year: {metadata.get('year')}")
        print(f"Abstract: {metadata.get('abstract', '')[:300]}...")


def search_pinecone(query: str, filter_dict=None):
    query_vector = encode_query(query).tolist()

    return index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_dict,
    )


def print_local_results(title: str, scores, reverse=True):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if reverse:
        top_indices = np.argsort(scores)[::-1][:TOP_K]
    else:
        top_indices = np.argsort(scores)[:TOP_K]

    for rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[idx]

        print(f"\n{rank}. Score: {scores[idx]:.4f}")
        print(f"Title: {row['title']}")
        print(f"Category: {row['category']}")
        print(f"Year: {row['year']}")
        print(f"Abstract: {str(row['abstract'])[:300]}...")


def compare_local_metrics(query: str):
    query_embedding = encode_query(query)

    # Оскільки embeddings і query_embedding нормалізовані,
    # cosine similarity == dot product.
    dot_scores = embeddings @ query_embedding

    cosine_scores = dot_scores / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    l2_distances = np.linalg.norm(embeddings - query_embedding, axis=1)

    print_local_results("LOCAL SEARCH — Cosine similarity", cosine_scores, reverse=True)
    print_local_results("LOCAL SEARCH — Dot product", dot_scores, reverse=True)
    print_local_results("LOCAL SEARCH — L2 distance", l2_distances, reverse=False)


def main():
    print("\nQUERY:", QUERY)

    pure_results = search_pinecone(QUERY)
    print_pinecone_results("PINECONE — Pure semantic search", pure_results)

    current_year = pd.Timestamp.now().year
    last_5_years_filter = {
        "category": {"$eq": "cs.LG"},
        "year": {"$gte": current_year - 5},
    }

    rl_recent_results = search_pinecone(RL_QUERY, last_5_years_filter)
    print_pinecone_results(
        "PINECONE — Reinforcement learning, last 5 years, category cs.LG",
        rl_recent_results,
    )

    old_papers_filter = {
        "year": {"$lt": 2015}
    }

    old_results = search_pinecone(QUERY, old_papers_filter)
    print_pinecone_results(
        "PINECONE — Older papers before 2015, any category",
        old_results,
    )

    print("\n" + "=" * 80)
    print("FILTER COMPARISON")
    print("=" * 80)
    print(
        "Фільтр cs.LG + останні 5 років звужує пошук до новіших робіт з machine learning. "
        "Фільтр до 2015 року повертає старіші статті з будь-яких категорій, тому результати "
        "можуть бути менш сучасними і тематично ширшими."
    )

    compare_local_metrics(QUERY)


if __name__ == "__main__":
    main()