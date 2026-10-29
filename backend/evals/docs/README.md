# DocMind evaluation

This folder is the record of how DocMind is measured and how each change moved the numbers. The rule is simple: **no change to chunking, retrieval or prompting is called an improvement unless it is measured here against a fixed baseline.**

```
docs/evaluation/
  README.md                       this file: method, rules, results history
  reports/_template.md            copy this for every experiment
  reports/<date>-<name>.md        one report per experiment
  runs/<date>-<name>.txt          raw console output of the run (committed as evidence)
```

## 1. Results history

Every row is measured on the same dataset version and split as its baseline. Counts are out of the dev split's answerable questions (n=44 for dataset v1). MRR is computed over the top 20, so it does not depend on `top_k`.

| ID | Date | Change | Level | Dataset | App `top_k` | Hit@`top_k` | Hit@20 | MRR | Fixed / broken vs. R0 | Status | Report |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| R0 | 2026-10-28 | Baseline: chunks 1000/200, top 5 | retrieval | v1 | 5 | 21 (48%) | 36 (82%) | 0.344 | - | Baseline | [report](reports/2026-10-28-baseline-chunking-grid.md) |
| R1 | 2026-10-28 | `top_k` 5 to 10 | retrieval | v1 | 10 | 31 (70%) | 36 (82%) | 0.344 | 10 / 0 (no question can get worse: top 10 contains top 5) | Proposed: about 2x retrieved context (up to ~10,000 chars); generation check pending | [report](reports/2026-10-28-baseline-chunking-grid.md) |
| R2 | | Hybrid search (BM25 + dense) | retrieval | v1 | | | | | | Planned | |
| R3 | | Cross-encoder reranker over the top 20 | retrieval | v1 | | | | | | Planned | |
| R4 | | Embedding model comparison | retrieval | v1 | | | | | | Planned | |
| R5 | | Prompt variants (strict vs. partial-answer rule) | generation | v1 | | | | | | Planned | |
| Final | | Single run on the test split | both | v1 | | | | | | Planned | |

Generation-level rows will add: false-refusal rate (answerable but refused), hallucination rate (unanswerable but answered), citation validity and faithfulness. Retrieval and generation rows are kept separate because a retrieval gain is not an application gain until generation confirms it.

## 2. Dataset

| Version | Date | Contents | Known issues |
|---|---|---|---|
| v1 | 2026-10-28 | 100 questions, 25 per document: `alice`, `nist_ai_rmf`, `nist_playbook`, `peers_paper`. Dev split: 44 answerable questions (alice 12, nist_ai_rmf 9, nist_playbook 11, peers_paper 12). | `nist-ai-rmf-019`: evidence quote not found in the extracted text, so the evidence ceiling is 43/44 |
| v2 (planned) | | Fix `nist-ai-rmf-019`; add entity, numeric and table questions (currently only 2 each in dev) | |

**Any edit to the question files creates a new dataset version, and the baseline must be re-run on it before comparing anything.** Record the version with `git log -1 --format=%h -- backend/evals/questions`.

**Documents** (PDFs are not committed; place them in `backend/evals/docs/` under the names shown):

| `doc` | Source | Licence |
|---|---|---|
| `alice` | *Alice's Adventures in Wonderland*, Project Gutenberg | Public domain |
| `nist_ai_rmf` | NIST AI 100-1, Artificial Intelligence Risk Management Framework 1.0 (2023) | US government work: confirm and record |
| `nist_playbook` | NIST AI RMF Playbook | US government work: confirm and record |
| `peers_paper` | Divoli, Nakov, Hearst (2012), *Do peers see more in a paper than its authors?* | Confirm the licence on the publisher's page and record it here |

**Labels.** Each answerable question lists one or more verbatim evidence quotes. A retrieved chunk is relevant if it fully contains any of them. Matching ignores case, whitespace, punctuation, hyphenation and ligatures. Unanswerable questions have no evidence and are typed `hard_negative`, `off_topic` or `false_premise`.

**Splits.** A stable hash of the question ID assigns about 40% of questions to `test`; the rest are `dev`. Adding questions never moves existing ones.

## 3. Metrics

| Metric | Meaning |
|---|---|
| **Hit@k** | A relevant chunk is in the top k. Counted per question. |
| **MRR** | Mean of 1 / rank of the first relevant chunk (0 if none in the top 20). |
| **Intact** | Questions whose evidence quote survives chunking in at least one chunk. The ceiling for Hit@k. |
| **Misses at k** | Answerable questions with no relevant chunk in the top k, with their evidence status (`intact`, `split`, `missing`) and rank. The starting point for error analysis. |

**Why not precision and recall?** Evidence labels give no stable count of "all relevant chunks": the same quote sits in one chunk without overlap and in two with overlap, so a chunk-level recall denominator changes with the very settings being compared. With one or two relevant chunks per question, precision@k is capped near 1/k, overlap inflates it, and string labels miss chunks that answer in different words. Hit rate answers the question the generator cares about: did it receive a chunk containing what it needs? Recall over evidence quotes would only differ from hit rate for multi-chunk questions, and LLM-judged context precision is the better tool for measuring noise in the context.

## 4. Rules for claiming an improvement

1. **Change one variable at a time.**
2. **Same dataset version and split as the baseline.** If the dataset changed, re-run the baseline first.
3. **Report counts (k/44), not only percentages.**
4. **Compare question by question.** List the questions fixed (missed before, hit now) and broken (hit before, missed now). Rule of thumb for 44 questions: a net gain under about 4 is noise; 8 fixed vs. 1 broken is convincing; 5 fixed vs. 3 broken is not.
5. **Tune on dev only.** The test split is run once, at the end, for the headline numbers.
6. **Record everything needed to reproduce:** git commit, dataset version, split, model, chunk settings, `top_k`, exact command, raw output.
7. **Note the cost** of a change (context size, latency, extra compute) next to its gain.
8. **A retrieval gain is not an application gain** until the generation metrics confirm it.

## 5. How to run and record an experiment

From `backend/`:

```powershell
# commit your code change first, so the hash identifies what was measured
git rev-parse --short HEAD
git log -1 --format=%h -- evals/questions        # dataset version

# run and keep the raw output (Out-File hides progress; view with Get-Content)
$run = "..\docs\evaluation\runs\2026-10-28-baseline-chunking-grid.txt"
python -m app.evaluation.run_retrieval --docs evals\docs --questions evals\questions --detail | Out-File -Encoding utf8 $run
Get-Content $run
```

Then:

1. Copy `reports/_template.md` to `reports/<date>-<name>.md` and fill it in.
2. Compute **fixed / broken**: take the "misses at k" list of the baseline run and of the new run (at the `top_k` the app uses) and compare the question IDs. Fixed = in the baseline list only. Broken = in the new list only. Use `--miss-k 10` to list misses at k=10.
3. Add a row to the results history above (and to the improvement log in the root README if it is a headline change).
4. Commit the code change, the report and the raw output together.

## 6. Planned experiments

1. **Hybrid search** (BM25 + dense, merged with reciprocal rank fusion). Hypothesis, untested: exact-term and numeric questions fail because dense embeddings blur exact terms.
2. **Reranker** over the top 20. Its ceiling is the Hit@20 of the first stage (82% in R0), and it can only reorder what was retrieved.
3. **Embedding model comparison**, watching each model's maximum input length against the chunk size.
4. **Generation evaluation:** refusal confusion matrix, citation validity, key-fact match, and LLM-judged faithfulness validated against about 30 hand-graded answers.
5. **Dataset v2** with more entity, numeric, table and multi-chunk questions, then a final run on the test split.
