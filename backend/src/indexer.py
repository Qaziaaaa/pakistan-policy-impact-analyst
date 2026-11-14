import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv("backend/.env")

COLLECTION_NAME = "pakistan_economic_policy"
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_DIM = 3072

def index_chunks():
    print("🚀 Connecting to Qdrant Cloud and Gemini...")
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 1. Setup collection if not exists
    if not qdrant.collection_exists(COLLECTION_NAME):
        print(f"📦 Creating collection: {COLLECTION_NAME}")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )

    # 2. Load extracted chunks
    chunks_path = Path("backend/data/processed/chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"📄 Loaded {len(chunks)} chunks. Indexing with rate-limit pacing...")

    # Index in smaller batches (10 chunks) with pause to stay well below 30,000 tokens/min free tier
    batch_size = 10
    total_indexed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]

        success = False
        while not success:
            try:
                res = ai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                )
                success = True
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print("  ⏳ Free tier rate-limit hit, waiting 35s...")
                    time.sleep(35)
                else:
                    raise e

        points = []
        for j, chunk in enumerate(batch):
            vector = res.embeddings[j].values
            point_id = i + j + 1
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["id"],
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "fiscal_year": chunk["metadata"]["fiscal_year"],
                        "file_name": chunk["metadata"]["file_name"],
                        "page_number": chunk["metadata"]["page_number"]
                    }
                )
            )

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        total_indexed += len(points)
        print(f"  Indexed {total_indexed}/{len(chunks)} chunks...")
        # Brief breather between batches
        time.sleep(3)

    print(f"\n🎉 Successfully indexed all {total_indexed} chunks into Qdrant Cloud!")

if __name__ == "__main__":
    index_chunks()
