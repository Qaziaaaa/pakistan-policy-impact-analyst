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
from backend.src.evaluation import LLMJudge

load_dotenv("backend/.env")

app = FastAPI(title="Pakistan Policy Impact Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_research_graph()
memory = LongTermMemory()
judge = LLMJudge()

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

        # 1. Status: Planning
        yield json.dumps({
            "type": "status",
            "node": "planner",
            "message": "🧠 Planning research strategy and sub-queries..."
        })
        await asyncio.sleep(0.3)

        # 2. Run multi-agent LangGraph workflow
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(None, lambda: graph.invoke(test_input, config=config))

        retrieved_chunks = final_state.get("retrieved_chunks", [])
        final_report = final_state.get("final_response", "")

        # 3. Stream retrieved evidence chunks
        yield json.dumps({
            "type": "evidence",
            "chunks": [
                {
                    "source": c["source"],
                    "fiscal_year": c["fiscal_year"],
                    "page_number": c["page_number"],
                    "snippet": c["content"][:200] + "..."
                }
                for c in retrieved_chunks
            ]
        })

        # 4. Stream Final Report
        yield json.dumps({
            "type": "final",
            "report": final_report,
            "critique": final_state.get("critique", "")
        })

        # 5. Run Live LLM-as-a-Judge Evaluation & Stream Metrics
        context_str = "\n".join([c["content"] for c in retrieved_chunks])
        eval_scores = await loop.run_in_executor(
            None,
            lambda: judge.evaluate(request.query, context_str, final_report)
        )

        yield json.dumps({
            "type": "evaluation",
            "scores": eval_scores
        })

    return EventSourceResponse(event_generator())

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=5005, reload=False)
