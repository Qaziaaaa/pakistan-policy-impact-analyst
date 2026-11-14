import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv("backend/.env")

COLLECTION_NAME = "pakistan_economic_policy"
EMBEDDING_MODEL = "gemini-embedding-001"

class PolicyRetriever:
    def __init__(self):
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def search(
        self,
        query: str,
        top_k: int = 3,
        fiscal_year: str = None,
        source: str = None
    ) -> list[dict]:
        """
        Embeds the search query and finds the Top-K nearest chunks in Qdrant Cloud,
        with optional metadata filtering.
        """
        # 1. Embed user query
        res = self.ai.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        query_vector = res.embeddings[0].values

        # 2. Build metadata filter conditions
        must_conditions = []
        if fiscal_year:
            must_conditions.append(
                FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year))
            )
        if source:
            must_conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # 3. Top-K similarity search (Qdrant v1.20+ query_points API)
        response = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        return [
            {
                "score": hit.score,
                "content": hit.payload["content"],
                "source": hit.payload["source"],
                "fiscal_year": hit.payload["fiscal_year"],
                "file_name": hit.payload["file_name"],
                "page_number": hit.payload["page_number"]
            }
            for hit in response.points
        ]

if __name__ == "__main__":
    retriever = PolicyRetriever()
    test_query = "What was the headline CPI inflation rate in May 2024?"
    print(f"\n🔍 Testing Query: '{test_query}'")
    hits = retriever.search(test_query, top_k=2)

    for i, h in enumerate(hits, 1):
        print(f"\n--- Result #{i} (Cosine Score: {h['score']:.4f}) ---")
        print(f"📍 Citation: {h['source']} ({h['fiscal_year']}), Page {h['page_number']}")
        print(f"📝 Text Snippet:\n{h['content'][:250]}...")
