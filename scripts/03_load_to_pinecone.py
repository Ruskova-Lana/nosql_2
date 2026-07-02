# scripts/03_load_to_pinecone.py
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс, або перевіряємо, чи існує індекс (щоб не створювати дублікат)
existing_indexes = [index["name"] for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(5)
else:
    print(f"Index already exists: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

print("Loading data...")

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

if len(df) != len(embeddings):
    raise ValueError(
        f"Data and embeddings size mismatch: {len(df)} rows vs {len(embeddings)} embeddings"
    )

print(f"Loaded {len(df)} papers")
print(f"Loaded embeddings shape: {embeddings.shape}")

print("Uploading vectors to Pinecone...")

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading batches"):
    end = min(start + BATCH_SIZE, len(df))

    vectors = []

    for i in range(start, end):
        row = df.iloc[i]

        vector = {
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"]),
            },
        }

        vectors.append(vector)

    index.upsert(vectors=vectors)

print("Waiting for Pinecone to update stats...")
time.sleep(10)

stats = index.describe_index_stats()
print(f"Total vectors in index: {stats['total_vector_count']}")
