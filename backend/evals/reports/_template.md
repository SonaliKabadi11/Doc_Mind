# R<n>: <short name of the change>

| | |
|---|---|
| **Date** | YYYY-MM-DD |
| **Level** | retrieval / generation |
| **Compared against** | R<m> (the baseline this change is measured against) |
| **Dataset** | v<x> (hash: `git log -1 --format=%h -- backend/evals/questions`) |
| **Split** | dev (n = <answerable questions>) |
| **Code version** | commit `<git rev-parse --short HEAD>` |
| **Embedding model** | |
| **Chunk size / overlap** | |
| **App `top_k`** | |
| **Command** | `<exact command>` |
| **Raw output** | [`../runs/<date>-<name>.txt`](../runs/<date>-<name>.txt) |

## 1. Hypothesis and change

What was expected to improve, and why. **Exactly one variable changed:** <which one>. Everything else is identical to the baseline.

## 2. Results

Counts out of n. Put the baseline and the new run side by side.

| | Hit@`top_k` | Hit@20 | MRR | (generation runs: false-refusal rate, hallucination rate, citation validity, faithfulness) |
|---|---:|---:|---:|---|
| Baseline (R<m>) | | | | |
| This change (R<n>) | | | | |
| Difference | | | | |

Cost of the change (context size, latency, extra compute, extra dependencies):

## 3. Question by question

Take the "misses at k" list of both runs (at the `top_k` the app uses).

- **Fixed** (missed before, hit now): <ids>
- **Broken** (hit before, missed now): <ids>
- Net: <fixed> fixed, <broken> broken

Reading it (44 questions): a net gain under about 4 is noise; 8 fixed vs. 1 broken is convincing; 5 fixed vs. 3 broken is not.

## 4. By document and question type

Where did the change help or hurt? Remember that groups of 2 questions support no conclusion.

## 5. Why it worked or did not

Look at the questions that moved. State what you checked (retrieved chunks, scores, ranks) and what you could not check. Mark anything untested as a hypothesis.

## 6. Decision

Adopted / rejected / needs more data, and the reason. If adopted, name the code and config change that makes it the new default.

## 7. Limitations

Sample size, dev split only, label quality, anything specific to this experiment.

## 8. Next step
