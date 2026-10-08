import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.CRITICAL)  # quiet, we just care about exception types
# Ensure the `backend` directory is on sys.path so the `app` package imports
# resolve when running tests from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.base import DocumentNotFoundError
from app.retrieval.in_memory_store import InMemoryStore

logging.basicConfig(level=logging.CRITICAL)  # quiet, we just care about exception types

store = InMemoryStore()

print("--- 1) add_chunks with mismatched lengths ---")
try:
    store.add_chunks("docX", ["only one"], np.array([[1.0, 0.0], [0.0, 1.0]]))
except ValueError as e:
    print("Correctly got ValueError:", e)

print("\n--- 2) search on missing document, check exact exception type ---")
try:
    store.search("nope", np.array([1.0, 0.0]), 2)
except DocumentNotFoundError as e:
    print("Correctly got DocumentNotFoundError, str(e) =", repr(str(e)))

print("\n--- 3) delete_document on missing document, check exact exception type ---")
try:
    store.delete_document("nope")
except DocumentNotFoundError as e:
    print("Correctly got DocumentNotFoundError:", e)
