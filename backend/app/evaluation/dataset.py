"""Evaluation dataset: loading, validation and a stable dev/test split"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from app.models.models import Question

POSITIVE_TYPES = {"fact", "entity", "numeric", "table", "multi_chunk"}
NEGATIVE_TYPES = {"hard_negative", "off_topic", "false_premise"}
TEST_FRACTION = 0.4
MIN_EVIDENCE_CHARS = 15  # after squashing; shorter quotes risk accidental matches


def hash_split(question_id: str, test_fraction:float =TEST_FRACTION ) -> str:
    """Stable assignment: depends only on the id, so adding questions never moves old ones."""
    bucket = int(hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    return "test" if bucket < test_fraction else "dev"

def _text(raw:dict, key:str, where:str)-> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' must be a non-empty string")
    return value

def _parse(raw: object, where:str) -> Question:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: each line must be a JSON object")
    qid, doc, text = _text(raw, "id", where), _text(raw, "doc", where), _text(raw, "question", where)
    answerable = raw.get("answerable")
    if not isinstance(answerable, bool):
        raise ValueError(f"{where}: 'answerable' must be true or false")

    qtype = _text(raw, "type", where)
    allowed = POSITIVE_TYPES if answerable else NEGATIVE_TYPES
    if qtype in allowed:
        kind = "answerable" if answerable else "unanswerable"
    else:
        raise ValueError(f"{where}: type {qtype!r} is not valid for {kind} questions {sorted(allowed)}")
    evidence = raw.get("evidence") or []
    if not isinstance(evidence, list) or not all(isinstance(e, str) and e.strip() for e in evidence):
        raise ValueError(f"{where}: 'evidence' must be a list of non-empty strings")
    if answerable and not evidence:
        raise ValueError(f"{where}: answerable questions need at least one evidence quote")
    if not answerable and evidence:
        raise ValueError(f"{where}: unanswerable questions must not have evidence")

    split = raw.get("Split") or hash_split(qid)
    if split not in {"dev", "test"}:
        raise ValueError(f"{where}: 'split' must be 'dev' or 'test'")
    reference = raw.get("reference_answer")
    if reference is not None and not isinstance(reference, str):
        raise ValueError(f"{where}: 'reference_answer' must be a string")
    return Question(qid, doc, text, answerable, qtype, split, tuple(evidence), reference)



def load_questions(path: Path) -> list[Question]:
    """Load one .jsonl file, or every .jsonl file in a directory."""
    if not path.exists():
        raise FileNotFoundError(f"questions path not found: {path.resolve()} (current directory: {Path.cwd()})")
    
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No .jsonl files found in {path}")
    questions: list[Question] = []
    seen: set[str] = set()
    for file in files:
        for n, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(),start=1):
            if not line.strip():
                continue
            where = f"{file.name}:{n}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON ({e.msg})") from e
            question = _parse(raw, where)
            if question.id in seen:
                raise ValueError(f"{where}: duplicate id {question.id!r}")
            seen.add(question.id)
            questions.append(question)
        return questions

def select_split(questions: list[Question], split:str)-> list[Question]:
    if split == "all":
        return list(questions)
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev', 'test' or 'all'")
    return [q for q in questions if q.split == split]

"""
Imagine you run an evaluation script on your dataset today and get an accuracy score of 85%. Two weeks later, you run the exact same code, but the accuracy drops to 82%.
• Without this fingerprint: You might waste days debugging your model code, unaware that a colleague secretly added or edited a few lines of questions in the dataset files.
• With this fingerprint: You can look at your run files. If the fingerprint for the first run was 8f2b31ca9e04 and the second run is c71a33b92f11, you instantly know: "The code didn't break; my source data changed!"
"""

def dataset_fingerprint(path:Path) -> str:
    """Short hash of the question files, stored in run files so results are traceable."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:12]

def describe(questions:list[Question]) -> str:
    lines = [f"{len(questions)} questions"]
    docs = sorted({q.doc for q in questions})
    for doc in docs:
        qs = [q for q in questions if q.doc == doc]
        n_ans = sum(q.answerable for q in qs)
        n_test = sum(q.split == "test" for q in qs)
        lines.append(f"  {doc:<16} {len(qs):>3} total | {n_ans} answerable, {len(qs) - n_ans} unanswerable"
                     f" | dev {len(qs) - n_test}, test {n_test}")
    types = Counter(q.type for q in questions)
    lines.append("  types: " + ", ".join(f"{t}={c}" for t, c in sorted(types.items())))
    return "\n".join(lines)