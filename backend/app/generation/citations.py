"""
You told the model "only cite numbers that appear in the context", but a prompt is a request, not a guarantee. Models sometimes cite [7] when they were given 5 sources, or cite loosely. This function enforces in code what the prompt only asks for.
"""
"""Parse and validate [n] citation markers in an LLM answer (pure functions)."""


import re
from collections.abc import Sequence

from app.models.models import ParsedAnswer, Citation, SearchResult

# Optional leading blanks + "[1]" or "[1, 2]". Max 2 digits so "[2024]" is left alone.
_CITE_RE = re.compile(r"([ \t]*)\[(\d{1,2}(?:[ \t]*,[ \t]*\d{1,2})*)\]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?])")


def parse_citations(answer: str, sources: Sequence[SearchResult]) -> ParsedAnswer:
    """Validate [n] markers against ``sources`` (``sources[n - 1]`` is cited as [n])."""
    seen: list[int] = []
    invalid: list[int] = []

    def rewrite(match:re.Match[str]) -> str:
        refs = [int(part) for part in match.group(2).split(",")]
        valid = [r for r in refs if 1<= r <= len(sources)]

        for r in refs:
            if r not in valid and r not in invalid:
                invalid.append(r)

        for r in valid:
            if r not in seen:
                seen.append(r)

        if len(valid) == len(refs):
            return match.group(0)
        if not valid:
            return ""
        return f"{match.group(1)}[{','.join(map(str, valid))}]"

    text = _CITE_RE.sub(rewrite, answer)
    if not invalid:
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text).strip()

    citations = [_to_citation(ref, sources[ref-1]) for ref in seen]
    return ParsedAnswer(text=text, citations=citations, invalid_refs = invalid)

def _to_citation(ref: int, result: SearchResult) -> Citation:
    chunk = result.chunk
    return Citation(
        ref=ref,
        chunk_id=chunk.chunk_id,
        page=chunk.page,
        text=chunk.text,
        score=result.score,
    )
