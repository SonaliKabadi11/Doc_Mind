# DocMind Evaluation Dataset — Sources and Licenses

This directory documents the sources, licensing notes, and intended evaluation use of the documents in the DocMind RAG evaluation corpus.

## 1. Corpus inventory

| ID | Document | Source | License / reuse notes |
|---|---|---|---|
| `alice` | *Alice’s Adventures in Wonderland* — Lewis Carroll | https://www.gutenberg.org/ebooks/11 | Public domain in the United States. If redistributing the Project Gutenberg ebook with its branding or license material, comply with the Project Gutenberg License. Check local copyright rules if distributing outside the United States. |
| `nist_ai_rmf` | *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* — NIST, NIST AI 100-1 (2023) | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf | NIST government-authored works are generally not protected by US copyright, subject to exceptions. Check the publication for third-party content and applicable notices. Acknowledge NIST as the source. |
| `nist_playbook` | *NIST AI RMF Playbook* — National Institute of Standards and Technology | https://airc.nist.gov/airmf-resources/playbook/ | NIST public information is generally available for copying and distribution unless marked as copyrighted. Check individual content and notices before redistribution. Acknowledge NIST as the source. |
| `peers_paper` | *Do Peers See More in a Paper Than Its Authors?* — Anna Divoli, Preslav Nakov, and Marti A. Hearst (2012) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3514807/ | Open access under the Creative Commons Attribution License. Reuse and redistribution are permitted when the original work is properly cited, subject to the license terms. Preserve attribution and any applicable notices. |

## 2. Intended use

These documents are selected to evaluate different retrieval and document-processing challenges:

- **Alice:** Narrative text, entities, facts, and multi-passage questions.
- **NIST AI RMF:** Technical definitions, structured sections, and procedural questions.
- **NIST Playbook:** Structured guidance, categories, and table-related retrieval.
- **Research paper:** Academic terminology, research findings, and evidence-based questions.

## 3. Dataset and evaluation policy

1. Store the original source URL and license notes for every document.
2. Keep evaluation questions, reference answers, and verbatim evidence quotes in the dataset.
3. Verify that each evidence quote occurs in the extracted document text.
4. Do not assume that free access means unrestricted redistribution.
5. Preserve required attribution and licensing notices.
6. Before publishing a release, recheck the license and any third-party content in each downloaded file.

## 4. Attribution and limitations

This corpus is intended for evaluating retrieval and answer generation in DocMind. Document availability and licensing were checked against the linked source pages; this inventory is not a substitute for legal advice or a guarantee that every embedded image, figure, or third-party item is reusable.

For NIST materials, consult:
https://www.nist.gov/copyrights-disclaimers

For Project Gutenberg terms, consult:
https://www.gutenberg.org/policy/license.html

**Last reviewed:** 2026-10-15
