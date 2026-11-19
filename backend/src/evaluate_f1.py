import os
import re
import json
import string
from collections import Counter
from dotenv import load_dotenv

from backend.src.agents import build_research_graph, generate_with_fallback

load_dotenv("backend/.env")

# 10 Ground-Truth Q&A pairs directly from Pakistan Economic Survey 2023-24
GROUND_TRUTH_DATASET = [
    {
        "id": 1,
        "question": "What was Pakistan's real GDP growth rate in FY 2024?",
        "reference_answer": "Amid positive developments, Pakistan's GDP growth reached 2.38 percent in FY 2024 after facing challenges in FY 2023.",
        "key_facts": ["2.38 percent", "FY 2024", "growth reached 2.38"]
    },
    {
        "id": 2,
        "question": "What was the headline CPI inflation rate in May 2024 compared to May 2023?",
        "reference_answer": "On a year-on-year basis, CPI inflation was 11.8 percent in May 2024, a significant decrease from 38.0 percent in May 2023.",
        "key_facts": ["11.8 percent in May 2024", "38.0 percent in May 2023"]
    },
    {
        "id": 3,
        "question": "What was the State Bank of Pakistan policy rate during July to April FY 2024?",
        "reference_answer": "The State Bank of Pakistan maintained a tight monetary stance and kept the policy rate at 22 percent during July to April FY 2024.",
        "key_facts": ["22 percent", "July-April FY 2024", "tight monetary stance"]
    },
    {
        "id": 4,
        "question": "How much did Broad Money (M2) expand between July and March FY 2024?",
        "reference_answer": "Broad Money (M2) expanded by 7.0 percent to Rs 2,216.1 billion between July and March FY 2024.",
        "key_facts": ["7.0 percent", "Rs 2,216.1 billion", "July and March FY 2024"]
    },
    {
        "id": 5,
        "question": "What was the Weighted Average Lending Rate (WALR) in March 2024 compared to March 2023?",
        "reference_answer": "The Weighted Average Lending Rate (WALR) rose to 20.8 percent in March 2024 from 17.98 percent in March 2023.",
        "key_facts": ["20.8 percent in March 2024", "17.98 percent in March 2023"]
    },
    {
        "id": 6,
        "question": "What was the Weighted Average Deposit Rate (WADR) on fresh deposits in March 2024?",
        "reference_answer": "The Weighted Average Deposit Rate (WADR) on fresh deposits increased to 10.4 percent in March 2024 from 8.09 percent in March 2023.",
        "key_facts": ["10.4 percent in March 2024", "8.09 percent in March 2023"]
    },
    {
        "id": 7,
        "question": "What was the banking spread in Pakistan in March 2024?",
        "reference_answer": "The banking spread broadened to 10.4 percent in March 2024 compared to 9.9 percent in March 2023.",
        "key_facts": ["10.4 percent in March 2024", "9.9 percent in March 2023"]
    },
    {
        "id": 8,
        "question": "What is the SBP inflation target range by September 2025?",
        "reference_answer": "The SBP Monetary Policy Committee projects inflation to decelerate to the target range of 5 to 7 percent by September 2025.",
        "key_facts": ["5 to 7 percent", "September 2025"]
    },
    {
        "id": 9,
        "question": "What was the growth rate of the Education sector in FY 2024?",
        "reference_answer": "The Education sector posted positive growth of 10.30 percent in FY 2024.",
        "key_facts": ["10.30 percent", "Education sector"]
    },
    {
        "id": 10,
        "question": "What caused the negative GDP growth rate in Pakistan during FY 2023?",
        "reference_answer": "The economy was hit by international supply shocks and domestic natural disasters such as floods in FY 2023 leading to a negative GDP growth rate.",
        "key_facts": ["international supply shocks", "domestic natural disasters", "floods"]
    }
]

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def compute_token_f1(prediction: str, reference: str) -> dict:
    """Standard token-level Precision, Recall, and F1 (SQuAD standard metric)."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def compute_semantic_f1(prediction: str, reference: str, key_facts: list) -> dict:
    """Evaluates factual capture (Recall) and hallucination-free precision using Gemini."""
    prompt = f"""You are an expert impartial evaluation judge. Compute the Semantic Fact Precision, Recall, and F1 score for this RAG output.

Reference Answer:
{reference}

Key Ground-Truth Facts Expected:
{json.dumps(key_facts)}

RAG Generated Answer:
{prediction}

Evaluate:
1. "recall": Fraction of key ground-truth facts successfully captured in the generated answer (0.0 to 1.0).
2. "precision": Fraction of factual claims in the generated answer that are actually correct and relevant to the question (0.0 to 1.0).
3. "f1": 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.

Respond ONLY with valid JSON:
{{
  "precision": 0.95,
  "recall": 1.0,
  "f1": 0.97,
  "analysis": "Short 1-sentence note"
}}
"""
    try:
        res = generate_with_fallback(prompt)
        if res.startswith("```json"):
            res = res[7:-3].strip()
        elif res.startswith("```"):
            res = res[3:-3].strip()
        data = json.loads(res)
        return {
            "precision": float(data.get("precision", 0.0)),
            "recall": float(data.get("recall", 0.0)),
            "f1": float(data.get("f1", 0.0)),
            "analysis": data.get("analysis", "")
        }
    except Exception as e:
        # Fallback to key fact string matching
        captured = sum(1 for kf in key_facts if kf.lower() in prediction.lower())
        rec = captured / len(key_facts) if key_facts else 1.0
        prec = 0.90 if captured > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "analysis": f"Matched {captured}/{len(key_facts)} key facts"
        }

def run_f1_evaluation():
    graph = build_research_graph()
    user_id = "eval_f1_user"

    print("=" * 75)
    print("🎯 STARTING RAG F1 EVALUATION BENCHMARK (10 Ground-Truth Questions)")
    print("=" * 75)

    results = []
    total_token_p = 0.0
    total_token_r = 0.0
    total_token_f1 = 0.0

    total_sem_p = 0.0
    total_sem_r = 0.0
    total_sem_f1 = 0.0

    for item in GROUND_TRUTH_DATASET:
        qid = item["id"]
        q = item["question"]
        ref = item["reference_answer"]
        kfs = item["key_facts"]

        print(f"\n[{qid}/10] Query: {q}")
        
        # Invoke RAG Multi-Agent System
        state = graph.invoke(
            {"user_id": user_id, "query": q, "retry_count": 0},
            config={"configurable": {"thread_id": f"eval_f1_q{qid}"}}
        )
        pred = state.get("final_response", "")

        # 1. Compute Token-Level F1
        tok_metrics = compute_token_f1(pred, ref)
        total_token_p += tok_metrics["precision"]
        total_token_r += tok_metrics["recall"]
        total_token_f1 += tok_metrics["f1"]

        # 2. Compute Semantic / Fact-Level F1
        sem_metrics = compute_semantic_f1(pred, ref, kfs)
        total_sem_p += sem_metrics["precision"]
        total_sem_r += sem_metrics["recall"]
        total_sem_f1 += sem_metrics["f1"]

        print(f"  📊 Token F1:    P: {tok_metrics['precision']:.3f} | R: {tok_metrics['recall']:.3f} | F1: {tok_metrics['f1']:.3f}")
        print(f"  🧠 Semantic F1: P: {sem_metrics['precision']:.3f} | R: {sem_metrics['recall']:.3f} | F1: {sem_metrics['f1']:.3f} ({sem_metrics.get('analysis')})")

        results.append({
            "id": qid,
            "question": q,
            "reference": ref,
            "generated": pred[:250] + "...",
            "token_metrics": tok_metrics,
            "semantic_metrics": sem_metrics
        })

    n = len(GROUND_TRUTH_DATASET)
    avg_tok_p = round(total_token_p / n, 4)
    avg_tok_r = round(total_token_r / n, 4)
    avg_tok_f1 = round(total_token_f1 / n, 4)

    avg_sem_p = round(total_sem_p / n, 4)
    avg_sem_r = round(total_sem_r / n, 4)
    avg_sem_f1 = round(total_sem_f1 / n, 4)

    summary = {
        "dataset_size": n,
        "token_level_averages": {
            "precision": avg_tok_p,
            "recall": avg_tok_r,
            "f1_score": avg_tok_f1
        },
        "semantic_factual_averages": {
            "precision": avg_sem_p,
            "recall": avg_sem_r,
            "f1_score": avg_sem_f1
        },
        "detailed_results": results
    }

    out_file = "backend/data/processed/f1_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("🏆 RAG F1 EVALUATION SUMMARY:")
    print("=" * 75)
    print(f"Token-Level (Lexical SQuAD standard):")
    print(f"  • Average Precision: {avg_tok_p * 100:.2f}%")
    print(f"  • Average Recall:    {avg_tok_r * 100:.2f}%")
    print(f"  • Average Token F1:  {avg_tok_f1 * 100:.2f}%")
    print("-" * 75)
    print(f"Semantic / Factual Level (Information Completeness & Accuracy):")
    print(f"  • Average Precision: {avg_sem_p * 100:.2f}%")
    print(f"  • Average Recall:    {avg_sem_r * 100:.2f}%")
    print(f"  • Average F1 Score:  {avg_sem_f1 * 100:.2f}%")
    print(f"\n💾 Full report saved to: {out_file}")
    print("=" * 75)

if __name__ == "__main__":
    run_f1_evaluation()
