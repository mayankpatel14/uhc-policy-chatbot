"""Policy retrieval via MedEmbed embeddings and Qdrant vector search."""

import sys
import time
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchExcept,
)

from chatbot.config import (
    EMBEDDING_SCRIPTS_DIR,
    RETRIEVAL_TOP_K,
    EXCLUDED_SECTIONS,
)

sys.path.insert(0, str(EMBEDDING_SCRIPTS_DIR))
from config import (
    EMBEDDING_MODEL_NAME,
    MAX_SEQ_LENGTH,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
)

MAX_RETRIES = 3
RETRY_BACKOFF = 2


@dataclass
class ChunkResult:
    text: str
    policy_name: str
    section: str
    plan_type: str
    score: float
    page_start: int = 0
    page_end: int = 0


class PolicyRetriever:
    """Loads models once, reuses across queries."""

    def __init__(self):
        self._model = None
        self._device = None
        self._client = None

    def _ensure_model(self):
        if self._model is not None:
            return
        self._device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=False)
        self._model.max_seq_length = MAX_SEQ_LENGTH

    def _ensure_client(self):
        if self._client is not None:
            return
        if QDRANT_URL:
            self._client = QdrantClient(
                url=QDRANT_URL, api_key=QDRANT_API_KEY,
                timeout=30, prefer_grpc=False,
            )
        else:
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    def init(self, status_callback=None):
        """Eagerly load model + client. Optional callback for progress messages."""
        cb = status_callback or (lambda msg: None)

        cb("Loading MedEmbed model...")
        self._ensure_model()
        cb(f"  Model loaded on {self._device}")

        cb("Connecting to Qdrant...")
        self._ensure_client()
        cb("  Connected")
        return self

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        section_filter: str | None = None,
        policy_filter: str | None = None,
        exclude_sections: bool = True,
    ) -> list[ChunkResult]:
        self._ensure_model()
        self._ensure_client()

        vec = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self._device,
        ).tolist()

        conditions = []
        if exclude_sections and EXCLUDED_SECTIONS:
            conditions.append(
                FieldCondition(
                    key="section",
                    match=MatchExcept(**{"except": list(EXCLUDED_SECTIONS)}),
                )
            )
        if section_filter:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section_filter)))
        if policy_filter:
            conditions.append(FieldCondition(key="policy_name", match=MatchValue(value=policy_filter)))

        qf = Filter(must=conditions) if conditions else None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                hits = self._client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vec,
                    query_filter=qf,
                    limit=top_k,
                    with_payload=True,
                ).points
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF ** attempt)
                else:
                    raise RuntimeError(
                        f"Qdrant query failed after {MAX_RETRIES} retries: {e}"
                    ) from e

        results = []
        for hit in hits:
            p = hit.payload
            results.append(ChunkResult(
                text=p.get("text", ""),
                policy_name=p.get("policy_name", ""),
                section=p.get("section", ""),
                plan_type=p.get("plan_type", ""),
                score=hit.score,
                page_start=p.get("page_start", 0),
                page_end=p.get("page_end", 0),
            ))

        SECTION_BOOST = {
            "Coverage Rationale": 0.04,
            "Coverage Summary": 0.03,
            "Benefit Considerations": 0.01,
            "Documentation Requirements": 0.01,
        }
        for r in results:
            r.score += SECTION_BOOST.get(r.section, 0.0)

        results.sort(key=lambda r: r.score, reverse=True)
        return results
