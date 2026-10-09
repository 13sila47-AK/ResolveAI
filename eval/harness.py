"""
harness.py — runs the full agent over the golden set and computes:
  1. Automated intent + escalation metrics, using the EXACT SAME protocol
     (accuracy, F1, precision, recall) as baselines/tfidf_baseline.py, so
     results/llm_agent_results.json is directly comparable to
     results/tfidf_baseline_results.json and results/trivial_baseline_results.json.
  2. LLM-as-judge reply-quality scores (eval/llm_judge.py) for every reply.

Usage:
  Real run (needs ANTHROPIC_API_KEY, ~200 golden rows x 4 LLM calls each =
  ~800 API calls -- budget a few minutes and check your rate limits):
      export ANTHROPIC_API_KEY=sk-ant-...
      python -m eval.harness

  Smoke test (no API key, no network, validates the pipeline runs and the
  result files are well-formed -- NOT a quality measurement, see
  agent/llm_client.py):
      python -m eval.harness --mock

  Quick subsample of the golden set (either mode):
      python -m eval.harness --mock --n 20
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from agent.config import GOLDEN_PATH, RESULTS_DIR
from agent.llm_client import get_client
from agent.pipeline import SpotifySupportAgent
from eval.llm_judge import judge_reply


def run_harness(mock: bool = False, n: int | None = None, sleep_s: float = 0.0, verbose: bool = True):
    golden = pd.read_csv(GOLDEN_PATH)
    if n:
        golden = golden.head(n)
    golden_ids = set(pd.read_csv(GOLDEN_PATH)["customer_tweet_id"])

    agent = SpotifySupportAgent(mock=mock, exclude_ids=golden_ids)
    judge_client = agent.client  # same model as the agent by default -- see eval/llm_judge.py docstring

    rows = []
    for i, (_, gold_row) in enumerate(golden.iterrows()):
        result = agent.run(gold_row["customer_text"])

        # Re-fetch the same grounding examples used by the drafter, purely
        # so the judge sees exactly what the drafter saw (auditable, not
        # re-retrieved independently which could drift).
        grounding = agent.retriever.retrieve(
            gold_row["customer_text"], k=3, intent_filter=result.intent
        )
        judgement = judge_reply(
            judge_client,
            gold_row["customer_text"],
            result.intent,
            result.escalate,
            grounding,
            result.reply,
        )

        row = {
            "customer_tweet_id": gold_row["customer_tweet_id"],
            "customer_text": gold_row["customer_text"],
            "true_intent": gold_row["true_intent"],
            "true_escalate": gold_row["escalate"],
            "pred_intent": result.intent,
            "intent_confidence": result.intent_confidence,
            "intent_parse_ok": result.intent_parse_ok,
            "pred_escalate": result.escalate,
            "escalate_reason": result.escalate_reason,
            "escalate_parse_ok": result.escalate_parse_ok,
            "reply": result.reply,
            "reply_parse_ok": result.reply_parse_ok,
            "n_grounding_examples": result.n_grounding_examples,
            "grounding_ids": result.grounding_ids,
            **{f"judge_{k}": v for k, v in asdict(judgement).items() if k not in ("raw_response",)},
        }
        rows.append(row)

        if verbose and (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(golden)}")
        if sleep_s:
            time.sleep(sleep_s)

    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame, mock: bool) -> dict:
    intent_acc = accuracy_score(df["true_intent"], df["pred_intent"])
    intent_f1_macro = f1_score(df["true_intent"], df["pred_intent"], average="macro", zero_division=0)
    intent_f1_weighted = f1_score(df["true_intent"], df["pred_intent"], average="weighted", zero_division=0)
    intent_report = classification_report(df["true_intent"], df["pred_intent"], zero_division=0)

    esc_acc = accuracy_score(df["true_escalate"], df["pred_escalate"])
    esc_precision = precision_score(df["true_escalate"], df["pred_escalate"], zero_division=0)
    esc_recall = recall_score(df["true_escalate"], df["pred_escalate"], zero_division=0)
    esc_f1 = f1_score(df["true_escalate"], df["pred_escalate"], zero_division=0)

    judge_ok = df[df["judge_parse_ok"]]
    n_judge_excluded = len(df) - len(judge_ok)
    judge_means = (
        judge_ok[["judge_groundedness", "judge_correctness", "judge_tone", "judge_actionability", "judge_safety", "judge_overall"]]
        .mean()
        .round(3)
        .to_dict()
        if len(judge_ok)
        else {}
    )

    results = {
        "baseline": "llm_agent" + ("_MOCK" if mock else ""),
        "mock_mode": mock,
        "n_golden": len(df),
        "intent_accuracy": round(intent_acc, 4),
        "intent_f1_macro": round(intent_f1_macro, 4),
        "intent_f1_weighted": round(intent_f1_weighted, 4),
        "escalate_accuracy": round(esc_acc, 4),
        "escalate_precision": round(esc_precision, 4),
        "escalate_recall": round(esc_recall, 4),
        "escalate_f1": round(esc_f1, 4),
        "intent_parse_failure_rate": round(1 - df["intent_parse_ok"].mean(), 4),
        "escalate_parse_failure_rate": round(1 - df["escalate_parse_ok"].mean(), 4),
        "reply_parse_failure_rate": round(1 - df["reply_parse_ok"].mean(), 4),
        "judge_parse_failure_rate": round(n_judge_excluded / len(df), 4) if len(df) else None,
        "judge_mean_scores_excl_parse_failures": judge_means,
    }
    if mock:
        results["WARNING"] = (
            "mock_mode=True: intent/escalate metrics reflect a rule-based "
            "stand-in (MockLLMClient), NOT a real LLM. Judge scores are "
            "fixed dummy values. This file validates that the pipeline and "
            "harness run correctly end-to-end -- it is not a quality "
            "result. Run without --mock (and with ANTHROPIC_API_KEY set) "
            "for real numbers."
        )
    return results, intent_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--n", type=int, default=None, help="Subsample size (default: full 200-row golden set)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between golden rows (rate limiting)")
    args = parser.parse_args()

    print(f"Running harness (mock={args.mock}, n={args.n or 'all 200'}) ...")
    df = run_harness(mock=args.mock, n=args.n, sleep_s=args.sleep)
    results, intent_report = compute_metrics(df, mock=args.mock)

    print(json.dumps(results, indent=2))
    print("\n--- Per-intent classification report ---")
    print(intent_report)

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_mock" if args.mock else ""
    results_path = RESULTS_DIR / f"llm_agent_results{suffix}.json"
    preds_path = RESULTS_DIR / f"llm_agent_predictions{suffix}.csv"
    report_path = RESULTS_DIR / f"llm_agent_classification_report{suffix}.txt"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(preds_path, index=False)
    with open(report_path, "w") as f:
        f.write(intent_report)

    print(f"\nWrote results to {RESULTS_DIR} (suffix='{suffix}')")


if __name__ == "__main__":
    main()
