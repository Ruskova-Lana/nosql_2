import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"

FIXED_CHUNK_SIZE = 120
FIXED_OVERLAP = 30
SEMANTIC_MAX_WORDS = 120
BATCH_SIZE = 100
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


def fixed_size_chunk(text: str, chunk_size: int = 120, overlap: int = 30):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def split_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str, max_words: int = 120):
    sentences = split_sentences(text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count <= max_words:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = [sentence]
            current_word_count = sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_index_if_needed(index_name: str):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while not pc.describe_index(index_name).status["ready"]:
            print(f"Waiting for index {index_name} to be ready...")
            time.sleep(5)
    else:
        print(f"Index already exists: {index_name}")


def prepare_chunk_records(papers_df: pd.DataFrame, strategy: str):
    records = []

    for paper_number, row in papers_df.iterrows():
        text = str(row["abstract"])

        if strategy == "fixed":
            chunks = fixed_size_chunk(text, FIXED_CHUNK_SIZE, FIXED_OVERLAP)
        elif strategy == "semantic":
            chunks = semantic_chunk(text, SEMANTIC_MAX_WORDS)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for chunk_number, chunk_text in enumerate(chunks):
            records.append(
                {
                    "id": f"{strategy}_{paper_number}_{chunk_number}",
                    "text_for_embedding": f"{row['title']} [SEP] {chunk_text}",
                    "metadata": {
                        "arxiv_id": str(row["id"]),
                        "title": str(row["title"]),
                        "chunk_text": chunk_text[:1000],
                        "chunk_number": int(chunk_number),
                        "year": int(row["year"]),
                        "category": str(row["category"]),
                    },
                }
            )

    return records


def upload_chunks(index_name: str, records: list):
    index = pc.Index(index_name)

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Uploading {index_name}"):
        end = min(start + BATCH_SIZE, len(records))
        batch = records[start:end]

        texts = [item["text_for_embedding"] for item in batch]

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vectors = []

        for item, embedding in zip(batch, embeddings):
            vectors.append(
                {
                    "id": item["id"],
                    "values": embedding.tolist(),
                    "metadata": item["metadata"],
                }
            )

        index.upsert(vectors=vectors)

    time.sleep(5)

    stats = index.describe_index_stats()
    print(f"Total vectors in {index_name}: {stats['total_vector_count']}")


def encode_query(query: str):
    text = f"{query} [SEP] {query}"

    return model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()


def search_chunks(index_name: str, query: str):
    index = pc.Index(index_name)

    results = index.query(
        vector=encode_query(query),
        top_k=TOP_K,
        include_metadata=True,
    )

    print("\n" + "=" * 80)
    print(f"Search in {index_name}")
    print(f"Query: {query}")
    print("=" * 80)

    for i, match in enumerate(results["matches"], start=1):
        metadata = match["metadata"]

        print(f"\n{i}. Score: {match['score']:.4f}")
        print(f"Title: {metadata.get('title')}")
        print(f"Year: {metadata.get('year')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Chunk #{metadata.get('chunk_number')}")
        print(f"Chunk text: {metadata.get('chunk_text', '')[:400]}...")


def main():
    longest_papers = (
        df.assign(abstract_length=df["abstract"].astype(str).str.split().str.len())
        .sort_values("abstract_length", ascending=False)
        .head(30)
    )

    print(f"Selected longest papers: {len(longest_papers)}")

    create_index_if_needed(FIXED_INDEX)
    create_index_if_needed(SEMANTIC_INDEX)

    fixed_records = prepare_chunk_records(longest_papers, strategy="fixed")
    semantic_records = prepare_chunk_records(longest_papers, strategy="semantic")

    print(f"Fixed-size chunks: {len(fixed_records)}")
    print(f"Semantic chunks: {len(semantic_records)}")

    upload_chunks(FIXED_INDEX, fixed_records)
    upload_chunks(SEMANTIC_INDEX, semantic_records)

    test_queries = [
        "machine learning for scientific data analysis",
        "quantum chromodynamics and particle collisions",
        "mathematical models in physics",
    ]

    for query in test_queries:
        search_chunks(FIXED_INDEX, query)
        search_chunks(SEMANTIC_INDEX, query)


if __name__ == "__main__":
    main()