# Pakistan Policy Impact Analyst

> An end-to-end, knowledge-grounded multi-agent research assistant that analyzes Pakistani economic policies, inflation, fiscal developments, and monetary trends using official government publications.

---

## 🏛️ Project Overview

- **Theme:** Knowledge-Grounded AI Research Assistant
- **Domain:** Economy, Finance & Public Policy (Pakistan)
- **Primary Corpora:**
  - Official **Pakistan Economic Survey 2023–24** (Finance Division, Government of Pakistan)
  - **State Bank of Pakistan (SBP)** Monetary Policy Statements & Sectoral Reports
- **Core Capabilities:** Real-time multi-agent decomposition, vector retrieval over dense PDF tables with exact page citations, reflection self-critique, cross-session long-term memory, and live LLM-as-a-Judge evaluation.

---

## 🏗️ System Architecture

```text
[ User Prompt via Web UI / API ]
                 │
                 ▼
    [ Supervisor / Router Node ]
        │                  │
   (Greeting / Off-Topic)  │ (Economic Research Question)
        ▼                  ▼
  [ Instant Reply ]   [ Long-Term Memory (Neon PostgreSQL) ]
                           │ (User persona & past research findings)
                           ▼
                 [ Planner Agent (Gemini) ]
                           │ (Decomposes into targeted sub-queries)
                           ▼
                [ Researcher Agent (ReAct) ]
                           │ ◄───► [ Qdrant Cloud Vector DB ]
                           │       (Top-K 3072-dim Cosine Similarity)
                           ▼
                 [ Policy Analyst Agent ]
                           │ (Synthesizes comparative report with [Page X] citations)
                           ▼
              [ Critic / Fact-Checker Node ]
              (Reflection / Self-Critique Pattern)
                 │                   │
         (Unverified Claim)          │ (100% Grounded)
                 ▼                   ├─► [ Persist Finding to Neon Postgres ]
         [ Re-synthesize ]           ├─► [ LLM-as-a-Judge Evaluation ]
                                     ▼
                      [ Stream Report to Lovable UI ]
```

---

## 🛠️ Tech Stack & Requirements Mapping

| Project Requirement | Implementation | Technology |
| :--- | :--- | :--- |
| **Real-World Corpus** | Pakistan Economic Survey 2023–24 (`growth`, `money_and_credit`, `highlights`) | Government of Pakistan Finance Division |
| **RAG Pipeline** | Cleaned sliding-window chunking + 3072-dim embeddings + Top-K similarity | Gemini Embeddings (`gemini-embedding-001`) + **Qdrant Cloud** |
| **Short-Term Memory** | Per-session thread state | LangGraph `MemorySaver` Checkpointer |
| **Long-Term Memory** | User profiles, research preferences, and verified facts saved across sessions | **Neon Serverless PostgreSQL** (`user_profiles`, `research_findings`) |
| **Multi-Agent Orchestration** | 5 specialized agents wired in a cyclic state graph | **LangGraph** (`Supervisor`, `Planner`, `Researcher`, `Analyst`, `Critic`) |
| **Agentic Design Patterns** | Router, Planning, ReAct tool calling, Reflection / Self-Critique loops | LangGraph state transitions & conditional edges |
| **Evaluation** | Faithfulness, Answer Relevance, and Citation Quality metrics | **LLM-as-a-Judge** + live UI scorecard |
| **Frontend UI/UX** | Lovable-style dark mode dashboard with real-time SSE streaming | Tailwind CSS, Marked.js, FontAwesome |
| **Backend API** | High-performance asynchronous REST & SSE streaming server | **FastAPI** + **Uvicorn** |

---

## 📂 Repository Structure

```text
├── backend/
│   ├── data/
│   │   ├── raw/                 # Official Economic Survey & SBP PDFs
│   │   └── processed/           # Extracted chunks & 100-prompt benchmark results
│   ├── src/
│   │   ├── ingestion.py         # PDF text & table parser with rich metadata extraction
│   │   ├── indexer.py           # Embeds chunks via Gemini and indexes to Qdrant Cloud
│   │   ├── retriever.py         # Top-K vector search with metadata filters
│   │   ├── memory.py            # SQLAlchemy models & Neon PostgreSQL store
│   │   ├── agents.py            # LangGraph multi-agent state graph & router
│   │   ├── evaluation.py        # LLM-as-a-Judge evaluation engine
│   │   └── benchmark_100.py     # 100-prompt end-to-end benchmark test suite
│   ├── main.py                  # FastAPI application with SSE streaming endpoints
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html               # Lovable-style interactive research dashboard
├── AGENT_RULES.md               # Engineering guidelines & working protocols
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Free Google Gemini API Key
- Free Qdrant Cloud Cluster
- Free Neon PostgreSQL Database

### 2. Setup Environment
```bash
git clone <your-repo-url>
cd pakistan-policy-impact-analyst

python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `backend/.env` and insert your keys:
```env
GEMINI_API_KEY="your-gemini-api-key"
QDRANT_API_KEY="your-qdrant-api-key"
QDRANT_URL="https://your-cluster.cloud.qdrant.io:6333"
DATABASE_URL="postgresql://user:password@your-neon-host/neondb?sslmode=require"
```

### 4. Ingest and Index Documents
```bash
# Extract clean chunks from PDFs
python backend/src/ingestion.py

# Embed and upload chunks to Qdrant Cloud
python backend/src/indexer.py
```

### 5. Launch the Web Application
```bash
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 5005
```
Open **`http://localhost:5005`** in your browser.

---

## 🧪 Benchmark Results

Tested against a comprehensive **100-Prompt Benchmark Suite** ([`backend/data/processed/benchmark_100_results.json`](backend/data/processed/benchmark_100_results.json)):
- **25 / 25 Greetings:** Filtered at `<1ms` without unnecessary LLM/vector calls.
- **25 / 25 Off-Topic Guardrails:** Non-economic queries gracefully redirected to domain topics.
- **5 / 5 Multi-Agent End-to-End:** 100% verified grounded with average LLM judge scores of `0.98 – 1.00`.
- **45 / 45 Economic Vector Retrieval:** Successfully retrieved grounded chunks with page citations.
- **Overall Pass Rate:** **100%**
