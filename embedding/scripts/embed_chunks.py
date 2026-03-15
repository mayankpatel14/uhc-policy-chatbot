"""
Embed RAG chunks using MedEmbed-large-v0.1 and save to .npz.

Usage:
    python embed_chunks.py
    python embed_chunks.py --limit 100
    python embed_chunks.py --batch-size 64      # override batch size
"""

import argparse
import json
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    RAG_CHUNKS_PATH,
    EMBEDDINGS_DIR,
    EMBEDDINGS_FILE,
    EMBEDDING_MODEL_NAME,
    BATCH_SIZE,
    MAX_SEQ_LENGTH,
)


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chunks(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if limit:
        chunks = chunks[:limit]
    return chunks


def build_embedding_text(chunk: dict) -> str:
    """
    Construct the text that gets embedded. Prepend key metadata so the
    embedding captures policy context, not just the raw paragraph.
    """
    parts = []

    policy = chunk.get("policy_name", "").replace("-", " ").title()
    if policy:
        parts.append(f"Policy: {policy}")

    section = chunk.get("section", "")
    if section:
        parts.append(f"Section: {section}")

    parts.append(chunk["text"])

    return " | ".join(parts)


def embed_in_batches(model, texts, batch_size, device):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        )
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Embed RAG chunks with MedEmbed")
    parser.add_argument("--limit", type=int, default=None, help="Limit chunks to embed (for testing)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for encoding")
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}")
    print(f"Model:  {EMBEDDING_MODEL_NAME}")
    print(f"Batch:  {args.batch_size}")

    print("\nLoading chunks...")
    chunks = load_chunks(RAG_CHUNKS_PATH, limit=args.limit)
    print(f"Loaded {len(chunks)} chunks")

    chunk_ids = [c["id"] for c in chunks]
    texts = [build_embedding_text(c) for c in chunks]

    print(f"\nLoading model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"Model loaded — embedding dim: {model.get_sentence_embedding_dimension()}")

    print("\nGenerating embeddings...")
    start = time.time()
    embeddings = embed_in_batches(model, texts, args.batch_size, device)
    elapsed = time.time() - start

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Time: {elapsed:.1f}s ({len(texts) / elapsed:.1f} chunks/sec)")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_FILE,
        ids=np.array(chunk_ids, dtype=object),
        embeddings=embeddings,
    )
    size_mb = EMBEDDINGS_FILE.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {EMBEDDINGS_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
