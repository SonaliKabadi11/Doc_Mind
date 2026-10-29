# R0 / R1: Baseline retrieval evaluation (chunking grid)

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Level** | Retrieval only (no LLM involved) |
| **Dataset** | v1: 100 questions across 4 documents (commit/hash: `<fill in>`) |
| **Split** | dev, 44 answerable questions (alice 12, nist_ai_rmf 9, nist_playbook 11, peers_paper 12) |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Code version** | commit `<fill in>` |
| **Grid** | chunk size {500, 1000, 1500} x overlap {0, 100, 200} |
| **Command** | `python -m app.evaluation.run_retrieval --docs evals\docs --questions evals\questions --detail` |
| **Raw output** | [`../runs/2026-10-28-baseline-chunking-grid.txt`](../runs/2026-10-28-baseline-chunking-grid.txt) |

## 1. Summary

- Retrieval is the weak point. At the current defaults (1000/200, top 5) a relevant chunk is retrieved for **21 of 44** answerable questions (48%). Retrieving the top 10 raises this to **31 of 44** (70%); the top 20 reaches **36 of 44** (82%).
- Chunk size and overlap matter less than the raw table suggests. At matched context size the nine configurations land within a few questions of each other, inside the noise for 44 questions.
- Chunking is not hiding the evidence: `intact` is 41 to 43 of 44 everywhere.
- The remaining headroom is in ranking (10 questions sit at ranks 6-10) and in candidate generation (8 questions never reach the top 20). That points to reranking and hybrid search as the next experiments.

## 2. How far to trust these numbers

44 questions: one question is 2.3 points. For a hit rate near 50% the sampling error is about 7 points (one standard error), and about 0.05 for MRR. **Differences of 1 to 4 questions between rows are noise** unless almost all changed questions move in the same direction. These are dev-split numbers used for decisions; they are not the final results.

## 3. Results: full grid

Counts out of 44. `Intact` = questions whose evidence is fully inside at least one chunk (ceiling for the hit columns).

| Size / overlap | Chunks | Intact | Hit@1 | Hit@3 | Hit@5 | Hit@8 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 / 0 | 1617 | 42 (95%) | 11 (25%) | 15 (34%) | 17 (39%) | 22 (50%) | 24 (55%) | 27 (61%) | 0.317 |
| 500 / 100 | 1809 | 41 (93%) | 11 (25%) | 13 (30%) | 16 (36%) | 21 (48%) | 24 (55%) | 30 (68%) | 0.319 |
| 500 / 200 | 2305 | 43 (98%) | 9 (20%) | 13 (30%) | 15 (34%) | 19 (43%) | 22 (50%) | 28 (64%) | 0.287 |
| 1000 / 0 | 852 | 43 (98%) | 7 (16%) | 10 (23%) | 15 (34%) | 22 (50%) | 27 (61%) | 33 (75%) | 0.261 |
| 1000 / 100 | 867 | 43 (98%) | 9 (20%) | 12 (27%) | 17 (39%) | 25 (57%) | 28 (64%) | 32 (73%) | 0.302 |
| **1000 / 200** | 933 | 43 (98%) | 11 (25%) | 14 (32%) | 21 (48%) | 26 (59%) | 31 (70%) | 36 (82%) | 0.344 |
| 1500 / 0 | 592 | 41 (93%) | 11 (25%) | 18 (41%) | 24 (55%) | 26 (59%) | 27 (61%) | 33 (75%) | 0.364 |
| 1500 / 100 | 593 | 43 (98%) | 12 (27%) | 17 (39%) | 27 (61%) | 29 (66%) | 29 (66%) | 34 (77%) | 0.390 |
| 1500 / 200 | 611 | 43 (98%) | 11 (25%) | 21 (48%) | 25 (57%) | 28 (64%) | 29 (66%) | 36 (82%) | 0.389 |

Ranges across the nine configurations: Hit@5 34-61%, Hit@10 50-70%, Hit@20 61-82%, MRR 0.26-0.39. Hit@1 is only 7-12 of 44 (16-27%) everywhere.

## 4. Chunk size: compare at equal context, not equal k

At equal k, larger chunks retrieve more text: k=5 is about 2,500 characters at size 500, 5,000 at size 1000 and 7,500 at size 1500. Matching the amount of text (characters estimated as k x chunk size, assuming near-full chunks):

| Approx. context | 500 family | 1000 family | 1500 family |
|---|---|---|---|
| ~5,000 chars | 22-24 (50-55%) at k=10 | 15-21 (34-48%) at k=5 | 17-21 (39-48%) at k=3 (4,500) |
| ~8,000 chars | not measured | 22-26 (50-59%) at k=8 | 24-27 (55-61%) at k=5 (7,500) |
| ~10-12,000 chars | 27-30 (61-68%) at k=20 | 27-31 (61-70%) at k=10 | 26-29 (59-66%) at k=8 (12,000) |

Within each row the spread is 3 to 6 questions with no consistent winner by chunk size. **How much text is retrieved drives hit rate more than how the text is cut.** All-MiniLM-L6-v2 reads only about 256 tokens (roughly 1,000 characters), so 1500-character chunks may have an under-represented tail. The 1500 rows show no sign of harm, but this sample cannot rule it out.

## 5. Overlap has no consistent effect

| Size | Hit@10 (overlap 0 / 100 / 200) | MRR (overlap 0 / 100 / 200) |
|---|---|---|
| 500 | 24 / 24 / 22 | 0.317 / 0.319 / 0.287 |
| 1000 | 27 / 28 / 31 | 0.261 / 0.302 / 0.344 |
| 1500 | 27 / 29 / 29 | 0.364 / 0.390 / 0.389 |

Overlap does not help at size 500, helps steadily at 1000 and slightly at 1500. The only consistent pattern is that 1000/0 is the weakest row for MRR. The supportable statement is "overlap of 100-200 is harmless and sometimes helps", not "overlap helps".

## 6. Where the failures sit (1000/200)

| Rank of first relevant chunk | Questions | What would address it |
|---|---:|---|
| 1-5 | 21 (48%) | Already works |
| 6-10 | 10 (23%) | Reranking or a larger `top_k` |
| 11-20 | 5 (11%) | Reranking, with diminishing returns |
| Beyond 20 | 8 (18%) | Better candidate generation; includes `nist-ai-rmf-019` (bad quote) |

Fifteen questions (34%) are retrieved within the top 20 but not the top 5. An ideal reranker could in principle lift Hit@5 toward 82%; a real one recovers part of that. The 8 questions beyond rank 20 cannot be rescued by reranking.

Misses at k=5 for 1000/200 (23 questions): alice-001, alice-007, alice-017, nist-ai-rmf-004, -007, -017, -018, -019, -020, nist-playbook-005, -006, -007, -008, -009, -019, -020, peers-paper-003, -004, -005, -006, -007, -018, -020.

## 7. By document and question type (1000/200)

| Document | n | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| alice | 12 | 9 (75%) | 12 (100%) | 0.650 |
| nist_ai_rmf | 9 | 3 (33%) | 5 (56%) | 0.220 |
| nist_playbook | 11 | 4 (36%) | 8 (73%) | 0.279 |
| peers_paper | 12 | 5 (42%) | 6 (50%) | 0.191 |

| Question type | n | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| fact | 29 | 15 (52%) | 20 (69%) | 0.429 |
| multi_chunk | 9 | 5 (56%) | 6 (67%) | 0.226 |
| entity | 2 | 0 | 2 (100%) | 0.118 |
| numeric | 2 | 0 | 1 (50%) | 0.050 |
| table | 2 | 1 (50%) | 2 (100%) | 0.171 |

Alice, a short novel in plain prose, is easy. The two technical documents and the research paper are much harder. At 500-character chunks the research paper was far worse (Hit@10 of 3/12 and 2/12). Per-document groups hold 9-12 questions, so their differences are only suggestive, and the entity, numeric and table groups (n=2 each) support no conclusion. Hypothesis, untested: in narrow-topic documents every chunk looks semantically similar, and exact terms (numbers, names) are what distinguishes the right chunk, which dense embeddings blur.

## 8. Dataset issue found

`nist-ai-rmf-019` has `evidence=missing` in every configuration: its quote is not in the text extracted from the PDF. It caps `intact` at 43/44 (98%). Fix or replace the quote in dataset v2, and re-run the baseline on it before comparing.

## 9. R1: raising `top_k` from 5 to 10 (same run, no re-run needed)

Hit@k at a fixed chunking is derived from one ranking of the top 20, so the effect of `top_k` is read directly from the table (1000/200):

| | `top_k` | Hit@`top_k` | MRR |
|---|---:|---:|---:|
| R0 baseline | 5 | 21/44 (48%) | 0.344 |
| R1 | 10 | 31/44 (70%) | 0.344 |

**10 questions fixed, 0 broken** (by construction, the top 10 contains the top 5). Cost: roughly twice the retrieved context (up to about 10,000 characters, inside the 12,000-character prompt budget), so higher token cost and latency. This is a retrieval-level result. Whether it also reduces false refusals without raising unsupported claims is measured in the generation phase, so R1 stays "proposed" until then.

## 10. Decision

- **Provisional defaults:** chunk size 1000, overlap 200, `top_k` 10. It has the best Hit@10 and Hit@20 and an MRR near the best, and it fits the prompt's context budget.
- **Alternative:** 1500/100 with `top_k` 8 (best MRR, about 35% fewer vectors). Retrieval metrics cannot separate the two; decide using generation metrics.
- Do not tune chunk parameters further on this dataset: the remaining differences are within noise.

## 11. Limitations

- Dev split only; the test split has not been used.
- 44 answerable questions; sampling error of several points.
- Labels are lower bounds: a chunk that answers in different words than the quote counts as a miss.
- Context estimates in section 4 assume near-full chunks.
- Retrieval only; no generation metrics yet.

## 12. Next steps

1. Fix `nist-ai-rmf-019` and create dataset v2; re-run this baseline as R0b.
2. Read the 8 questions beyond rank 20 by hand (top 5 retrieved chunks next to the evidence chunk) to classify the causes before building anything.
3. R2: hybrid search. R3: reranker over the top 20. Target: Hit@20 above 82% and Hit@5 above 60%.
4. Grow the dataset to 100+ answerable questions with more entity, numeric and table questions.
5. Build the generation evaluation and settle `top_k` with it.
