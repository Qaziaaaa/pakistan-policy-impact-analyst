import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

load_dotenv("backend/.env")

COLLECTION_NAME = "pakistan_economic_policy"
EMBEDDING_MODEL = "gemini-embedding-001"

def tokenize(text: str) -> list[str]:
    """Tokenize and normalize text for BM25 keyword matching."""
    return re.findall(r'\b\w+\b', text.lower())

class PolicyRetriever:
    """
    Production-grade Hybrid Retriever:
    Combines Qdrant Cloud Dense Vector Search (semantic meaning) with
    BM25 Okapi Sparse Search (exact figures, percentages, table codes)
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self):
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Load chunks in-memory for fast BM25 keyword scoring
        chunks_file = Path("backend/data/processed/chunks.json")
        self.chunks = []
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

        if self.chunks:
            self.corpus_tokens = [tokenize(c["content"]) for c in self.chunks]
            self.bm25 = BM25Okapi(self.corpus_tokens)
            print(f"⚡ [Hybrid Retriever]: BM25 index loaded with {len(self.chunks)} chunks.")
        else:
            self.bm25 = None

    def search_dense(self, query: str, top_k: int = 5, fiscal_year: str = None) -> list[dict]:
        """Dense Vector Search via Qdrant Cloud."""
        res = self.ai.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        query_vector = res.embeddings[0].values

        must_conditions = []
        if fiscal_year:
            must_conditions.append(
                FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year))
            )
        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        return [
            {
                "chunk_id": hit.payload["chunk_id"],
                "content": hit.payload["content"],
                "source": hit.payload["source"],
                "fiscal_year": hit.payload["fiscal_year"],
                "file_name": hit.payload["file_name"],
                "page_number": hit.payload["page_number"],
                "dense_score": hit.score
            }
            for hit in response.points
        ]

    def search_sparse_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """Sparse BM25 Keyword Search (exact matches for numbers, WALR, percentages)."""
        if not self.bm25:
            return []

        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                c = self.chunks[idx]
                results.append({
                    "chunk_id": c["id"],
                    "content": c["content"],
                    "source": c["metadata"]["source"],
                    "fiscal_year": c["metadata"]["fiscal_year"],
                    "file_name": c["metadata"]["file_name"],
                    "page_number": c["metadata"]["page_number"],
                    "bm25_score": float(scores[idx])
                })
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        fiscal_year: str = None,
        source: str = None,
        rrf_k: int = 60
    ) -> list[dict]:
        """
        Executes Hybrid Search combining Dense (Vectors) + Sparse (BM25)
        via Reciprocal Rank Fusion (RRF):
          RRF_Score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        # Fetch Top-K * 2 from each engine to build candidate pool
        dense_hits = self.search_dense(query, top_k=top_k * 2, fiscal_year=fiscal_year)
        sparse_hits = self.search_sparse_bm25(query, top_k=top_k * 2)

        # Map by chunk_id and aggregate RRF scores
        candidates = {}
        rrf_scores = {}

        # 1. Add Dense ranks
        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["chunk_id"]
            candidates[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Add Sparse BM25 ranks
        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit["chunk_id"]
            if cid not in candidates:
                candidates[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # 3. Sort by highest combined RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_hits = []
        for cid in sorted_cids[:top_k]:
            hit = candidates[cid]
            hit["score"] = round(rrf_scores[cid], 5)
            final_hits.append(hit)

        return final_hits

if __name__ == "__main__":
    retriever = PolicyRetriever()
    query = "What was Pakistan's real GDP growth rate in FY 2024?"
    print(f"\n🔍 Testing Hybrid Search (BM25 + Dense RRF) for: '{query}'")
    hits = retriever.search(query, top_k=3)

    for i, h in enumerate(hits, 1):
        print(f"\n--- Rank #{i} (RRF Score: {h['score']}) ---")
        print(f"📍 Citation: {h['source']} ({h['fiscal_year']}), Page {h['page_number']}")
        print(f"📝 Snippet:\n{h['content'][:220]}...")
