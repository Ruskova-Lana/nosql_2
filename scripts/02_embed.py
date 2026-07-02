import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64


def main():

    print("Loading dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Loaded {len(df)} papers")

    texts = [
        f"{title} [SEP] {abstract}"
        for title, abstract in zip(df["title"], df["abstract"])
    ]

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print()
    print(f"Processed texts: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Norm of first embedding: {np.linalg.norm(embeddings[0]):.6f}")

    os.makedirs("embeddings", exist_ok=True)

    np.save(OUTPUT_FILE, embeddings)

    print(f"Embeddings saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()