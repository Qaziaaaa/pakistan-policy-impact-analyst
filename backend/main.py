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
from backend.src.observability import ObservabilityTracker

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
tracker = ObservabilityTracker()

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
        tracker.reset()
        test_input = {
            "user_id": request.user_id,
            "query": request.query,
            "retry_count": 0
        }
        config = {"configurable": {"thread_id": f"session_{request.user_id}"}}

        yield json.dumps({
            "type": "status",
            "node": "supervisor",
            "message": "🧭 Routing query & analyzing intent..."
        })
        await asyncio.sleep(0.3)

        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(None, lambda: graph.invoke(test_input, config=config))

        is_non_research = final_state.get("intent") in ("greeting", "off_topic")
        retrieved_chunks = final_state.get("retrieved_chunks", [])
        final_report = final_state.get("final_response", "")

        # Compute telemetry metrics
        tracker.end_step("supervisor", input_text=request.query, output_text=final_report)
        metrics = tracker.get_metrics()

        # If it was a greeting or off-topic query, skip RAG evaluation and evidence rendering
        if is_non_research:
            yield json.dumps({
                "type": "final",
                "report": final_report,
                "is_greeting": True,
                "telemetry": metrics
            })
            return

        # Evidence chunks
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

        # Final Report with Telemetry
        yield json.dumps({
            "type": "final",
            "report": final_report,
            "is_greeting": False,
            "critique": final_state.get("critique", ""),
            "telemetry": metrics
        })

        # Real-time Evaluation
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
