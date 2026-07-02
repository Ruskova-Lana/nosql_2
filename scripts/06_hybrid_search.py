import os
import re

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 20
FINAL_TOP_K = 5
RRF_K = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def build_bm25_index():
    corpus = []

    for _, row in df.iterrows():
        text = f"{row['title']} {row['abstract']} {row['authors']}"
        corpus.append(tokenize(text))

    return BM25Okapi(corpus)


bm25 = build_bm25_index()


def encode_query(query: str):
    text = f"{query} [SEP] {query}"

    return model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()


def bm25_search(query: str, top_k: int = TOP_K):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[idx]

        results.append(
            {
                "rank": rank,
                "id": f"paper_{idx}",
                "score": float(scores[idx]),
                "title": row["title"],
                "category": row["category"],
                "year": int(row["year"]),
                "abstract": str(row["abstract"])[:300],
            }
        )

    return results


def vector_search(query: str, top_k: int = TOP_K):
    results = index.query(
        vector=encode_query(query),
        top_k=top_k,
        include_metadata=True,
    )

    output = []

    for rank, match in enumerate(results["matches"], start=1):
        metadata = match["metadata"]

        output.append(
            {
                "rank": rank,
                "id": match["id"],
                "score": float(match["score"]),
                "title": metadata.get("title"),
                "category": metadata.get("category"),
                "year": int(metadata.get("year")),
                "abstract": metadata.get("abstract", "")[:300],
            }
        )

    return output


def reciprocal_rank_fusion(bm25_results, vector_results, rrf_k: int = RRF_K):
    fused = {}

    for result_list, source in [
        (bm25_results, "bm25"),
        (vector_results, "vector"),
    ]:
        for item in result_list:
            doc_id = item["id"]

            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "title": item["title"],
                    "category": item["category"],
                    "year": item["year"],
                    "abstract": item["abstract"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "vector_rank": None,
                }

            fused[doc_id]["rrf_score"] += 1 / (rrf_k + item["rank"])
            fused[doc_id][f"{source}_rank"] = item["rank"]

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )[:FINAL_TOP_K]


def hybrid_search(query: str):
    bm25_results = bm25_search(query, TOP_K)
    vector_results = vector_search(query, TOP_K)

    return reciprocal_rank_fusion(bm25_results, vector_results)


def print_results(title: str, results, score_name: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, item in enumerate(results[:FINAL_TOP_K], start=1):
        print(f"\n{i}. {score_name}: {item.get(score_name):.4f}")
        print(f"ID: {item['id']}")
        print(f"Title: {item['title']}")
        print(f"Category: {item['category']}")
        print(f"Year: {item['year']}")
        print(f"Abstract: {item['abstract']}...")


def print_hybrid_results(title: str, results):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, item in enumerate(results, start=1):
        print(f"\n{i}. RRF score: {item['rrf_score']:.6f}")
        print(f"BM25 rank: {item['bm25_rank']}")
        print(f"Vector rank: {item['vector_rank']}")
        print(f"ID: {item['id']}")
        print(f"Title: {item['title']}")
        print(f"Category: {item['category']}")
        print(f"Year: {item['year']}")
        print(f"Abstract: {item['abstract']}...")


def main():
    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text",
    ]

    for query in queries:
        print("\n\n" + "#" * 100)
        print(f"QUERY: {query}")
        print("#" * 100)

        bm25_results = bm25_search(query)
        vector_results = vector_search(query)
        hybrid_results = reciprocal_rank_fusion(bm25_results, vector_results)

        print_results("BM25 top-5", bm25_results, "score")
        print_results("Vector search top-5", vector_results, "score")
        print_hybrid_results("Hybrid search top-5 with RRF", hybrid_results)


if __name__ == "__main__":
    main()