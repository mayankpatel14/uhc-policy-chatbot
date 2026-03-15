"""
Upsert precomputed embeddings and chunk metadata into Qdrant.

Usage:
    python store_qdrant.py
    python store_qdrant.py --recreate
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
)

from config import (
    RAG_CHUNKS_PATH,
    EMBEDDINGS_FILE,
    EMBEDDING_DIM,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_URL,
    QDRANT_API_KEY,
    PROVIDER_NAME,
    PROVIDER_SLUG,
)

UPSERT_BATCH_SIZE = 100


def get_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)


def ensure_collection(client: QdrantClient, recreate: bool = False):
    exists = client.collection_exists(QDRANT_COLLECTION)

    if exists and recreate:
        print(f"Dropping existing collection '{QDRANT_COLLECTION}'...")
        client.delete_collection(QDRANT_COLLECTION)
        exists = False

    if not exists:
        print(f"Creating collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIM})...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
                on_disk=False,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )
        print("Collection created.")

    print("Ensuring payload indexes for filtered search...")
    for field in ("section", "policy_name", "plan_type", "doc_type", "provider"):
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print("  Indexes created: section, policy_name, plan_type, doc_type, provider")


def load_data():
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    ids = data["ids"]
    embeddings = data["embeddings"]
    print(f"  Loaded {len(ids)} embeddings of dim {embeddings.shape[1]}")

    print("Loading chunk metadata...")
    with open(RAG_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_map = {c["id"]: c for c in chunks}
    print(f"  Loaded {len(chunks)} chunks")

    return ids, embeddings, chunk_map


def build_payload(chunk: dict) -> dict:
    return {
        "policy_name": chunk.get("policy_name", ""),
        "policy_number": chunk.get("policy_number", ""),
        "effective_date": chunk.get("effective_date", ""),
        "plan_type": chunk.get("plan_type", ""),
        "doc_type": chunk.get("doc_type", ""),
        "section": chunk.get("section", ""),
        "page_start": chunk.get("page_start", 0),
        "page_end": chunk.get("page_end", 0),
        "chunk_index": chunk.get("chunk_index", 0),
        "total_chunks_in_section": chunk.get("total_chunks_in_section", 0),
        "text": chunk.get("text", ""),
        "provider": PROVIDER_SLUG,
    }


def upsert_points(client, ids, embeddings, chunk_map):
    points = []
    skipped = 0

    for i, (chunk_id, vector) in enumerate(zip(ids, embeddings)):
        chunk_id_str = str(chunk_id)
        if chunk_id_str not in chunk_map:
            skipped += 1
            continue

        payload = build_payload(chunk_map[chunk_id_str])

        points.append(
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=payload,
            )
        )

    if skipped:
        print(f"  Skipped {skipped} embeddings (no matching chunk metadata)")

    print(f"  Upserting {len(points)} points in batches of {UPSERT_BATCH_SIZE}...")

    for batch_start in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="Upserting"):
        batch = points[batch_start : batch_start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch, wait=True)

    return len(points)


def main():
    parser = argparse.ArgumentParser(description="Store embeddings in Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()

    client = get_client()
    ensure_collection(client, recreate=args.recreate)

    ids, embeddings, chunk_map = load_data()
    total = upsert_points(client, ids, embeddings, chunk_map)

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"\nDone. Collection '{QDRANT_COLLECTION}' now has {info.points_count} points.")
    print(f"  Vectors dim: {EMBEDDING_DIM}")
    print(f"  Distance:    COSINE")
    print(f"  Provider:    {PROVIDER_NAME}")


if __name__ == "__main__":
    main()
