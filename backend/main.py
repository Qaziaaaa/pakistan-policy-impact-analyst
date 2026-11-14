import os
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.src.agents import build_research_graph
from backend.src.memory import LongTermMemory

load_dotenv("backend/.env")

app = FastAPI(title="Pakistan Policy Impact Analyst API")

# Enable CORS for modern frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_research_graph()
memory = LongTermMemory()

class ResearchRequest(BaseModel):
    user_id: str = "analyst_qazi_01"
    query: str

@app.get("/api/memory/{user_id}")
async def get_user_memory(user_id: str):
    user = memory.get_or_create_user(user_id)
    summary = memory.get_user_memory_summary(user_id)
    return {
        "user_id": user.user_id,
        "role": user.role,
        "preferred_topics": user.preferred_topics,
        "summary": summary
    }

@app.post("/api/research/stream")
async def stream_research(request: ResearchRequest):
    async def event_generator() -> AsyncGenerator[str, None]:
        test_input = {
            "user_id": request.user_id,
            "query": request.query,
            "retry_count": 0
        }
        config = {"configurable": {"thread_id": f"session_{request.user_id}"}}

        # Stream node updates in real-time
        yield json.dumps({
            "type": "status",
            "node": "planner",
            "message": "🧠 Planning research strategy and sub-queries..."
        })
        await asyncio.sleep(0.5)

        # Run the LangGraph execution in background thread
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(None, lambda: graph.invoke(test_input, config=config))

        # Send retrieved evidence citations
        yield json.dumps({
            "type": "evidence",
            "chunks": [
                {
                    "source": c["source"],
                    "fiscal_year": c["fiscal_year"],
                    "page_number": c["page_number"],
                    "snippet": c["content"][:200] + "..."
                }
                for c in final_state.get("retrieved_chunks", [])
            ]
        })

        # Send final report
        yield json.dumps({
            "type": "final",
            "report": final_state.get("final_response", ""),
            "critique": final_state.get("critique", "")
        })

    return EventSourceResponse(event_generator())

# Embedded ultra-sleek UI served directly by FastAPI
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
