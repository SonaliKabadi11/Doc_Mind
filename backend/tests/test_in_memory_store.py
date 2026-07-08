import unittest
import numpy as np
import sys
import os

# Ensure the `backend` directory is on sys.path so the `app` package imports
# resolve when running tests from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retreival.in_memory_store import InMemoryStore
from app.retreival.base import DocumentNotFoundError
from app.exception import CustomException

class TestInMemoryStore(unittest.TestCase):

    def setUp(self):
        """Runs before every single test to provide a fresh, clean database instance."""
        self.store = InMemoryStore()
        self.embedding_dim = 4  # Keep dimensions small for easy debugging
        
        # Standard valid mock data
        self.doc_id = "doc_123"
        self.chunks = ["Hello world", "Machine learning is fun", "Python programming"]
        self.embeddings = np.array([
            [0.1, 0.2, 0.7, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.2, 0.2, 0.2, 0.4]
        ], dtype=np.float32)

    # ==========================================
    # 1. ADD_CHUNKS EDGE CASES
    # ==========================================

    def test_add_chunks_success(self):
        """Happy Path: Ensure standard correct inputs successfully store data."""
        count = self.store.add_chunks(self.doc_id, self.chunks, self.embeddings)
        self.assertEqual(count, 3)
        self.assertIn(self.doc_id, self.store.chunk_db)

    def test_add_chunks_shape_mismatch(self):
        """Edge Case: Number of text chunks does NOT match vector rows."""
        mismatched_embeddings = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32) # Only 1 row
        
        # Should raise your CustomException due to length/shape mismatch
        with self.assertRaises(CustomException):
            self.store.add_chunks(self.doc_id, self.chunks, mismatched_embeddings)

    def test_add_chunks_empty_inputs(self):
        """Edge Case: Passing totally empty lists and an empty 2D array."""
        empty_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
        count = self.store.add_chunks("empty_doc", [], empty_embeddings)
        
        self.assertEqual(count, 0)
        self.assertEqual(len(self.store.chunk_db["empty_doc"]["chunks"]), 0)

    # ==========================================
    # 2. SEARCH EDGE CASES
    # ==========================================

    def test_search_document_not_found(self):
        """Edge Case: Querying a document ID that does not exist in the database."""
        query_vector = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        
        # Should raise DocumentNotFoundError explicitly
        with self.assertRaises(DocumentNotFoundError):
            self.store.search("non_existent_doc", query_vector, top_k=2)

    def test_search_top_k_larger_than_total_chunks(self):
        """Edge Case: Requesting a top_k of 10 when only 3 chunks exist."""
        self.store.add_chunks(self.doc_id, self.chunks, self.embeddings)
        query_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Should gracefully return all 3 available chunks, sorted, without crashing
        results = self.store.search(self.doc_id, query_vector, top_k=10)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].chunk_text, "Machine learning is fun") # Closest to [1,0,0,0]

    def test_search_query_vector_2d_handling(self):
        """Edge Case: Query vector arrives as a 2D shape (1, D) instead of 1D (D,)."""
        self.store.add_chunks(self.doc_id, self.chunks, self.embeddings)
        
        # Shape is (1, 4) instead of (4,)
        query_vector_2d = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        
        # Your code uses .flatten() on the dot product output.
        # This test verifies that dot-multiplying (3,4) by (1,4) doesn't break the math.
        try:
            results = self.store.search(self.doc_id, query_vector_2d, top_k=1)
            self.assertEqual(len(results), 1)
        except Exception as e:
            self.fail(f"Search crashed with 2D query vector: {e}")

    def test_search_score_sorting_order(self):
        """Edge Case: Verifying the results are strictly sorted from highest score to lowest."""
        self.store.add_chunks(self.doc_id, self.chunks, self.embeddings)
        query_vector = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32) # Targets index 0 [0.1, 0.2, 0.7, 0.0]
        
        results = self.store.search(self.doc_id, query_vector, top_k=3)
        
        # Verify scores decrease descendingly
        self.assertGreater(results[0].score, results[1].score)
        self.assertGreater(results[1].score, results[2].score)
        self.assertEqual(results[0].index, 0)

    # ==========================================
    # 3. DELETE_DOCUMENT EDGE CASES
    # ==========================================

    def test_delete_document_success(self):
        """Happy Path: Successfully drop a document and clean memory."""
        self.store.add_chunks(self.doc_id, self.chunks, self.embeddings)
        
        deleted_count = self.store.delete_document(self.doc_id)
        self.assertEqual(deleted_count, 3)
        self.assertNotIn(self.doc_id, self.store.chunk_db)

    def test_delete_document_not_found(self):
        """Edge Case: Deleting a document that was never added or already deleted."""
        with self.assertRaises(DocumentNotFoundError):
            self.store.delete_document("ghost_identity")


if __name__ == "__main__":
    unittest.main()
