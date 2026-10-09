# Spotify AI Support Agent — PART 1 of 2

> **This is Part 1.** It covers data pipeline, intent taxonomy, golden
> evaluation set, and two baselines — all runnable today with no API key.
> **Part 2** (LLM classifier + grounded reply drafter + escalation decision
> module + evaluation harness with LLM-as-judge + full report) is being
> built next; see `docs/decision_log.md` and the bottom of this README for
> exactly what's left.

Brand chosen: **Spotify** (`SpotifyCares` on Twitter). Why: see decision
log #1.

## What's in this zip

```
spotify-support-agent/
├── data/
│   ├── raw/                     # put twcs.csv here ONLY if you want to
│   │                             #   re-derive everything from scratch
│   ├── processed/
│   │   ├── spotify_pairs.csv          # 43,031 real customer<->agent pairs (already extracted)
│   │   └── spotify_weak_labeled.csv   # + weak intent/escalate labels (already labeled)
│   └── golden/
│       ├── golden_sample_raw.csv      # the 200-row stratified sample, unlabeled
│       └── golden_set.csv             # + hand labels (the actual golden eval set)
├── src/
│   ├── extract_brand_data.py    # raw twcs.csv -> spotify_pairs.csv
│   ├── weak_label.py            # spotify_pairs.csv -> spotify_weak_labeled.csv
│   ├── golden_labels.py         # the 200 hand-assigned (intent, escalate) labels
│   └── build_golden_set.py      # merges golden_labels.py onto golden_sample_raw.csv
├── baselines/
│   ├── trivial_baseline.py      # majority-class / never-escalate baseline
│   └── tfidf_baseline.py        # TF-IDF + Logistic Regression baseline
├── results/                     # JSON + CSV outputs from both baselines (already run)
├── docs/
│   ├── intent_taxonomy.md       # the 9 intents, derived from the data
│   ├── golden_set_methodology.md
│   └── decision_log.md          # 10 decisions so far, Part 2 adds more
└── requirements.txt
```

## Reproduce the headline results (under 2 minutes, no download needed)

Everything needed to reproduce the baseline numbers below is **already
included** in `data/processed/` and `data/golden/` — you do not need to
download the 516MB raw Twitter dataset to reproduce these results.

```bash
pip install -r requirements.txt
python baselines/trivial_baseline.py
python baselines/tfidf_baseline.py
```

### Results (Part 1 baselines only — the real LLM-based agent comes in Part 2)

| Metric | Trivial baseline | TF-IDF + LogReg baseline |
|---|---|---|
| Intent accuracy (9-way, n=200) | 9.5% | **65.0%** |
| Intent F1 (macro) | — | 66.1% |
| Escalate accuracy | 63.5% | **83.5%** |
| Escalate F1 | 0% (never predicts True) | 77.6% |

Full per-intent precision/recall is in
`results/tfidf_baseline_classification_report.txt`. Row-level predictions
(for failure analysis) are in `results/tfidf_baseline_predictions.csv`.

**Read this before quoting the 65%/83.5% numbers anywhere** — the TF-IDF
baseline is trained on weak (regex-labeled) data, so it inherits that
noise, and the golden set's class balance is stratified, not
traffic-representative (see `docs/golden_set_methodology.md`). A fuller
"what's misleading about this number" writeup will be in the Part 2 report,
since the LLM-based agent's headline number needs the same scrutiny and
it's more useful to do that analysis once, side by side.

## (Optional) Re-derive everything from the raw dataset

Only needed if you want to verify the extraction/weak-labeling yourself, or
re-run on a refreshed copy of the dataset.

1. Download `twcs.csv` from Kaggle: **thoughtvector/customer-support-on-twitter**
2. Place it at `data/raw/twcs.csv` (~516MB, ~3M rows)
3. Run:
   ```bash
   python src/extract_brand_data.py   # ~30-60s, streams the file twice
   python src/weak_label.py           # ~5s
   python src/build_golden_set.py     # <1s (re-applies the same hand labels)
   ```
4. Re-run the two baseline scripts above.

Note: `build_golden_set.py` re-merges the *existing* hand labels in
`src/golden_labels.py` onto `golden_sample_raw.csv`. It does not re-sample
or re-label — the golden set is fixed, checked-in data, not something
regenerated each run (see decision log #6 on why the sample is stratified
with a fixed seed and checked in rather than resampled).

## What's done (Part 1)
- [x] Brand selected and justified from real data
- [x] Full data pipeline: raw 3M-row Twitter dump → 43k clean Spotify
      customer↔agent pairs → weak intent/escalate labels
- [x] 9-class intent taxonomy, derived by reading the data (not guessed)
- [x] 200-example golden evaluation set, hand-labeled with a documented,
      consistent escalation policy — plus a measured 63% weak-label
      agreement rate (a real, reportable number)
- [x] Two working baselines with real, reproducible metrics on the golden set
- [x] Decision log (10 entries; Part 2 will add ~5 more)

## What's NOT done yet (coming in Part 2)
- [ ] LLM-based intent classifier (few-shot or fine-tuned prompt), evaluated
      the same way as the baselines above, on the same golden set
- [ ] Grounded reply drafter — retrieves similar historical resolved
      threads from `spotify_pairs.csv` and drafts a reply conditioned on them
- [ ] Escalation decision module with a stated reason string per decision
      (not just a boolean)
- [ ] Evaluation harness: automated metrics + LLM-as-judge rubric for reply
      quality, with a human-agreement check on a subset of judge scores
- [ ] Full report (≤6 pages): problem framing, results vs both baselines,
      top-5 failure modes with real examples, "what's misleading about my
      headline number," what's next with one more week
- [ ] Decision log entries #10+ covering the agent/judge design choices
- [ ] Note: the LLM parts require an Anthropic (or other) API key to run —
      the pipeline code will be written so you can run it with your own key;
      I can't execute live API calls from my current build environment, so
      Part 2 will ship with the code plus worked example outputs I generate
      manually, clearly labeled as such, alongside the real reproducible
      pipeline.

## Requirements
- Python 3.10+
- `pip install -r requirements.txt` (pandas, numpy, scikit-learn — all
  Part 1 work is 100% local, no API key needed)
