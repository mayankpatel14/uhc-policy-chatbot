"""
Search the Qdrant vector store with a natural language query.

Usage:
    python search.py "Is bariatric surgery covered for BMI over 40?"
    python search.py "criteria for sleep apnea treatment" --top-k 5
    python search.py "coverage for gene therapy hemophilia" --section "Coverage Rationale"
"""

import argparse
import sys
import time

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import (
    EMBEDDING_MODEL_NAME,
    MAX_SEQ_LENGTH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_URL,
    QDRANT_API_KEY,
    TOP_K,
)

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def get_client() -> QdrantClient:
    if QDRANT_URL:
        if not QDRANT_API_KEY or QDRANT_API_KEY == "YOUR_API_KEY_HERE":
            print(
                "WARNING: QDRANT_API_KEY is not set or still a placeholder.\n"
                "  Filtered queries WILL fail. Set a real key in .env",
                file=sys.stderr,
            )
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
            prefer_grpc=False,
        )
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)


def load_model():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=False)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model, device


def search(client, model, device, query, top_k=TOP_K, section_filter=None, policy_filter=None):
    query_vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    ).tolist()

    conditions = []
    if section_filter:
        conditions.append(FieldCondition(key="section", match=MatchValue(value=section_filter)))
    if policy_filter:
        conditions.append(FieldCondition(key="policy_name", match=MatchValue(value=policy_filter)))

    search_filter = Filter(must=conditions) if conditions else None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            results = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
            )
            return results.points
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt
                print(f"  Connection error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed after {MAX_RETRIES} attempts. Last error: {e}\n"
                    "Check that QDRANT_URL and QDRANT_API_KEY in .env are correct."
                ) from e


def format_result(hit, rank):
    p = hit.payload
    lines = [
        f"\n{'='*80}",
        f"  Rank #{rank}  |  Score: {hit.score:.4f}",
        f"  Policy:    {p.get('policy_name', 'N/A')}",
        f"  Section:   {p.get('section', 'N/A')}",
        f"  Effective: {p.get('effective_date', 'N/A')}",
        f"  Plan:      {p.get('plan_type', 'N/A')}",
        f"  Pages:     {p.get('page_start', '?')}-{p.get('page_end', '?')}",
        f"{'─'*80}",
    ]
    text = p.get("text", "")
    preview = text[:500] + ("..." if len(text) > 500 else "")
    lines.append(f"  {preview}")
    lines.append(f"{'='*80}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Search UHC policy chunks")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of results")
    parser.add_argument("--section", type=str, default=None, help="Filter by section name")
    parser.add_argument("--policy", type=str, default=None, help="Filter by policy slug")
    args = parser.parse_args()

    print(f"Query: \"{args.query}\"")
    if args.section:
        print(f"Section filter: {args.section}")
    if args.policy:
        print(f"Policy filter: {args.policy}")

    print("\nLoading model...")
    model, device = load_model()

    print("Connecting to Qdrant...")
    client = get_client()

    print(f"Searching (top-{args.top_k})...\n")
    try:
        results = search(client, model, device, args.query, args.top_k, args.section, args.policy)
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No results found.")
        return

    for i, hit in enumerate(results, 1):
        print(format_result(hit, i))


if __name__ == "__main__":
    main()
