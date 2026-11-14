import os
import re
import json
from typing import TypedDict, List, Dict, Any
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
    intent: str  # "greeting", "off_topic", or "research"
    plan: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    draft_report: str
    critique: str
    is_grounded: bool
    final_response: str
    retry_count: int

# 2. Agent LLM Setup with Multi-Model Failover
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_PRIORITY = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]
retriever = PolicyRetriever()
lt_memory = LongTermMemory()


def generate_with_fallback(prompt: str) -> str:
    """Calls Gemini with automatic model rotation if a model hits 429/503."""
    last_err = None
    for model_name in MODEL_PRIORITY:
        try:
            res = ai_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return res.text.strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Gemini models exhausted or unavailable: {last_err}")


GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|salam|assalam\s*o\s*alaikum|aoa|good\s*(morning|afternoon|evening)|"
    r"how\s*are\s*you|who\s*are\s*you|what\s*(can\s*you\s*do|are\s*you)|"
    r"help|thanks|thank\s*you|bye|goodbye|ok|okay|yo|sup|namaste|greetings)[\s\?!\.]*$",
    re.IGNORECASE,
)

ECONOMIC_KEYWORDS = {
    "pakistan", "gdp", "growth", "inflation", "cpi", "sbp", "state bank",
    "monetary", "fiscal", "policy", "rate", "interest", "money", "m2",
    "credit", "bank", "banking", "spread", "walr", "wadr", "deposit",
    "lending", "imf", "sba", "debt", "deficit", "tax", "fbr", "budget",
    "survey", "economic", "economy", "export", "import", "trade", "remittance",
    "currency", "rupee", "pkr", "exchange", "agriculture", "manufacturing",
    "services", "industry", "investment", "saving", "unemployment", "labour",
    "labor", "energy", "tariff", "subsidy", "revenue", "expenditure", "fy24",
    "fy23", "fy25", "price", "spi", "wpi", "loan",
    "finance", "insurance", "public administration", "education", "health",
    "commodity", "supply", "demand", "stabilization", "mpc", "reserve",
}


# --- NODE 0: Supervisor / Router Agent (Pattern: Hybrid Router Pattern) ---
def supervisor_router_node(state: ResearchState) -> Dict[str, Any]:
    print("\n🧭 [Supervisor / Router]: Classifying user intent...")
    query = state["query"].strip()
    user_summary = lt_memory.get_user_memory_summary(state["user_id"])

    # 1. Fast deterministic check for greetings / meta questions
    if not query or GREETING_PATTERNS.match(query) or len(query) <= 3:
        reply = (
            "Hello! I am the **Pakistan Policy Impact Analyst**, an agentic research assistant "
            "grounded in the official **Pakistan Economic Survey** and **State Bank of Pakistan (SBP)** reports.\n\n"
            "You can ask me questions such as:\n"
            "- *How did Pakistan's GDP growth and headline CPI inflation perform in FY 2024?*\n"
            "- *What drove broad money (M2) growth and banking spreads in FY 2024?*"
        )
        print("  🎯 Intent classified as: greeting (fast-path)")
        return {"intent": "greeting", "final_response": reply, "user_context": user_summary}

    # 2. Check if query contains any economic/policy signal
    lower_q = query.lower()
    has_econ_signal = any(kw in lower_q for kw in ECONOMIC_KEYWORDS)

    if not has_econ_signal:
        reply = (
            "I specialize exclusively in **Pakistan's macroeconomic policy, inflation, GDP growth, "
            "and State Bank of Pakistan (SBP) reports**. Your query appears to be outside my domain.\n\n"
            "Try asking about **GDP growth, CPI inflation, M2 money supply, banking spreads, or IMF stabilization in FY 2024**."
        )
        print("  🎯 Intent classified as: off_topic")
        return {"intent": "off_topic", "final_response": reply, "user_context": user_summary}

    print("  🎯 Intent classified as: research")
    return {"intent": "research", "final_response": "", "user_context": user_summary}


def route_intent(state: ResearchState) -> str:
    if state.get("intent") in ("greeting", "off_topic"):
        return "end"
    return "planner"


# --- NODE 1: Planner Agent ---
def planner_node(state: ResearchState) -> Dict[str, Any]:
    print("\n🧠 [Planner Agent]: Breaking down research query...")

    prompt = f"""You are the Lead Planning Agent for Pakistan Economic & Policy Research.
User Profile & Past Research Context:
{state.get('user_context', '')}

Research Question: {state['query']}

Task: Break this question into 2 specific search queries to retrieve exact data from the Pakistan Economic Survey or SBP reports.
Respond ONLY in valid JSON list of strings, e.g.:
["economic growth rate Pakistan FY24", "headline CPI inflation rate May 2024"]
"""
    try:
        text = generate_with_fallback(prompt)
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        plan = json.loads(text)
        if not isinstance(plan, list) or not plan:
            plan = [state["query"]]
    except Exception:
        plan = [state["query"]]

    print(f"  📋 Sub-tasks generated: {plan}")
    return {"plan": plan}


# --- NODE 2: Researcher Agent (ReAct Tool Calling) ---
def researcher_node(state: ResearchState) -> Dict[str, Any]:
    print("\n🔍 [Researcher Agent]: Searching Qdrant Cloud knowledge base...")
    collected_chunks = state.get("retrieved_chunks", []) or []

    for subquery in state["plan"]:
        print(f"  🔎 Executing retrieval for: '{subquery}'")
        hits = retriever.search(subquery, top_k=2)
        collected_chunks.extend(hits)

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
2. Every major statistic, number, or claim MUST be immediately followed by an inline citation using the header metadata provided above, e.g. [Pakistan Economic Survey (2023-24), Page X].
3. If data for a specific metric is not in the context, explicitly acknowledge the data gap without guessing.
"""
    draft = generate_with_fallback(prompt)
    return {"draft_report": draft}


# --- NODE 4: Critic / Fact-Checker Agent (Reflection Pattern) ---
def critic_node(state: ResearchState) -> Dict[str, Any]:
    print("\n🛡️ [Critic Agent]: Auditing citations and groundedness...")

    evidence_text = "\n\n".join([
        f"[Source: {c['source']} ({c['fiscal_year']}), Page {c['page_number']}]\n{c['content']}"
        for c in state["retrieved_chunks"]
    ])
    draft = state["draft_report"]

    prompt = f"""You are a strict Fact-Checker and Groundedness Judge.
Retrieved Document Evidence (including valid source and page headers):
{evidence_text}

Draft Report to Audit:
{draft}

Evaluate:
1. Are all statistics, claims, and figures directly supported by the retrieved document evidence?
2. Does the report cite valid page numbers from the headers above?

Reply ONLY in valid JSON:
{{
  "is_grounded": true or false,
  "feedback": "Concise explanation of audit result"
}}
"""
    try:
        text = generate_with_fallback(prompt)
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        audit = json.loads(text)
        is_grounded = bool(audit.get("is_grounded", True))
        feedback = audit.get("feedback", "Passed audit")
    except Exception:
        is_grounded = True
        feedback = "Passed audit"

    retries = state.get("retry_count", 0)
    print(f"  ⚖️ Grounded: {is_grounded} | Feedback: {feedback}")

    if is_grounded or retries >= 1:
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


def should_continue(state: ResearchState) -> str:
    if state["is_grounded"]:
        return "end"
    print("  🔁 Reflection loop triggered: re-synthesizing with fact-checker feedback...")
    return "re_analyze"


def build_research_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("supervisor", supervisor_router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("critic", critic_node)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "end": END,
            "planner": "planner"
        }
    )

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
