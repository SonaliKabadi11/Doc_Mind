"""Text matching helpers for evidence-based relevance labels."""

import re
import unicodedata
from collections.abc import Sequence

_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)

# Higher = better. Used to combine the status of several evidence strings.
STATUS_ORDER = {"missing": 0, "split": 1, "intact": 2}


def squash(text: str) -> str:
    """Lowercase letters and digits only, NFKC-normalised.

    Whitespace, line breaks, hyphenation, curly vs straight quotes, dashes, markup
    underscores and ligatures all disappear, so a quote copied from a reader still matches
    the text a PDF parser produced. Evidence strings are long enough that accidental
    matches are not a practical concern.
    """
    return _NON_ALNUM.sub("", unicodedata.normalize("NFKC", text).casefold())


def locate(evidence: str, squashed_chunks: Sequence[str], squashed_doc: str) -> str:
    """Where does this evidence live, given a particular chunking?

    intact:  fully inside at least one chunk (retrieval can find it)
    split:   in the document, but cut by a chunk or page boundary (retrieval cannot)
    missing: not in the extracted text at all (bad quote, or extraction problem)
    """
    needle = squash(evidence)
    if not needle:
        return "missing"
    if any(needle in chunk for chunk in squashed_chunks):
        return "intact"
    if needle in squashed_doc:
        return "split"
    return "missing"


def best_status(statuses: Sequence[str]) -> str:
    return max(statuses, key=STATUS_ORDER.__getitem__) if statuses else "missing"