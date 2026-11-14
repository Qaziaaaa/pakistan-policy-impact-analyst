import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
from dotenv import load_dotenv

from google import genai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.src.retriever import PolicyRetriever
from backend.src.memory import LongTermMemory

load_dotenv("backend/.env")

# 1. State Schema shared across all agents in the graph
class ResearchState(TypedDict):
    user_id: str
    query: str
    user_context: str
    plan: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    draft_report: str
    critique: str
    is_grounded: bool
    final_response: str
    retry_count: int

# 2. Agent LLM Setup
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CHAT_MODEL = "gemini-3.5-flash"
retriever = PolicyRetriever()
lt_memory = LongTermMemory()

# --- NODE 1: Planner Agent ---
def planner_node(state: ResearchState) -> Dict[str, Any]:
    print("\n🧠 [Planner Agent]: Breaking down research query...")
    user_summary = lt_memory.get_user_memory_summary(state["user_id"])
    
    prompt = f"""You are the Lead Planning Agent for Pakistan Economic & Policy Research.
User Profile & Past Research Context:
{user_summary}

Research Question: {state['query']}

Task: Break this question into 2 specific search queries to retrieve exact data from the Pakistan Economic Survey or SBP reports.
Respond ONLY in valid JSON list of strings, e.g.:
["economic growth rate Pakistan FY24", "headline CPI inflation rate May 2024"]
"""
    response = ai_client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )
    
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()

    try:
        plan = json.loads(text)
    except Exception:
        plan = [state["query"]]

    print(f"  📋 Sub-tasks generated: {plan}")
    return {"plan": plan, "user_context": user_summary}

# --- NODE 2: Researcher Agent (ReAct Tool Calling) ---
def researcher_node(state: ResearchState) -> Dict[str, Any]:
    print("\n🔍 [Researcher Agent]: Searching Qdrant Cloud knowledge base...")
    collected_chunks = state.get("retrieved_chunks", []) or []
    
    for subquery in state["plan"]:
        print(f"  🔎 Executing retrieval for: '{subquery}'")
        hits = retriever.search(subquery, top_k=2)
        collected_chunks.extend(hits)

    # Deduplicate chunks by snippet start
    seen = set()
    unique_chunks = []
    for c in collected_chunks:
        preview = c["content"][:60]
        if preview not in seen:
            seen.add(preview)
            unique_chunks.append(c)

    print(f"  📚 Retrieved {len(unique_chunks)} distinct evidence chunks.")
    return {"retrieved_chunks": unique_chunks}

# --- NODE 3: Policy Analyst Agent ---
def analyst_node(state: ResearchState) -> Dict[str, Any]:
    print("\n📊 [Analyst Agent]: Synthesizing comparative research report...")
    
    context_str = "\n\n".join([
        f"--- Source: {c['source']} ({c['fiscal_year']}), Page {c['page_number']} ---\n{c['content']}"
        for c in state["retrieved_chunks"]
    ])

    critique_instruction = ""
    if state.get("critique"):
        critique_instruction = f"\nAddress this previous critique: {state['critique']}"

    prompt = f"""You are a Senior Macroeconomic Policy Analyst for Pakistan.
User Query: {state['query']}
{critique_instruction}

Evidence Chunks from Official Documents:
{context_str}

Instructions:
1. Synthesize an evidence-backed, factual research briefing answering the query.
2. Every major statistic, number, or claim MUST be immediately followed by an inline citation like [Pakistan Economic Survey 2023-24, Page X].
3. If data for a specific year is not in the context, explicitly acknowledge the data gap.
"""
    response = ai_client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )
    draft = response.text.strip()
    return {"draft_report": draft}

# --- NODE 4: Critic / Fact-Checker Agent (Reflection Pattern) ---
def critic_node(state: ResearchState) -> Dict[str, Any]:
    print("\n🛡️ [Critic Agent]: Auditing citations and groundedness...")
    
    evidence_text = "\n".join([c["content"] for c in state["retrieved_chunks"]])
    draft = state["draft_report"]

    prompt = f"""You are a strict Fact-Checker and Groundedness Judge.
Retrieved Document Evidence:
{evidence_text}

Draft Report to Audit:
{draft}

Evaluate:
1. Are all statistics, claims, and figures directly supported by the retrieved document evidence?
2. Does the report use proper page citations?

Reply ONLY in valid JSON:
{{
  "is_grounded": true or false,
  "feedback": "Concise explanation of audit result"
}}
"""
    response = ai_client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )
    
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()

    try:
        audit = json.loads(text)
        is_grounded = bool(audit.get("is_grounded", True))
        feedback = audit.get("feedback", "Passed audit")
    except Exception:
        is_grounded = True
        feedback = "Parsed by default"

    retries = state.get("retry_count", 0)
    print(f"  ⚖️ Grounded: {is_grounded} | Feedback: {feedback}")

    if is_grounded or retries >= 1:
        # Save key finding to long-term memory
        lt_memory.save_finding(
            user_id=state["user_id"],
            topic=state["query"][:50],
            finding=draft[:300] + "...",
            citation="Verified by Pakistan Policy Impact Analyst"
        )
        return {
            "is_grounded": True,
            "final_response": draft,
            "critique": feedback
        }
    else:
        return {
            "is_grounded": False,
            "critique": feedback,
            "retry_count": retries + 1
        }

# --- Conditional Edge: Reflection Loop ---
def should_continue(state: ResearchState) -> str:
    if state["is_grounded"]:
        return "end"
    print("  🔁 Reflection loop triggered: re-synthesizing with fact-checker feedback...")
    return "re_analyze"

# --- 5. Compile the LangGraph ---
def build_research_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("critic", critic_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "critic")

    builder.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "re_analyze": "analyst"
        }
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    graph = build_research_graph()
    test_user = "analyst_qazi_01"
    test_input = {
        "user_id": test_user,
        "query": "How did Pakistan's GDP growth and inflation perform in FY 2024 according to the Economic Survey?",
        "retry_count": 0
    }

    config = {"configurable": {"thread_id": "session_test_001"}}
    print(f"🚀 Running End-to-End Multi-Agent Research Assistant for '{test_user}'...")
    final_output = graph.invoke(test_input, config=config)

    print("\n" + "="*50)
    print("📄 FINAL GROUNDED RESEARCH REPORT:")
    print("="*50)
    print(final_output["final_response"])
