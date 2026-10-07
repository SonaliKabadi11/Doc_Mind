import logging

import numpy as np

from app.exception import DocumentNotFoundError
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
