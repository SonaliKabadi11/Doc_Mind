import re
import zlib

import numpy as np

from app.models import Chunk


class FakeIngestion:
    DIM = 64

    def extract_pages(self, source):
        return [
            (1, "Cats purr and sleep all day. Dogs bark loudly at strangers."),
            (2, "Python is a popular programming language for machine learning."),
        ]

    def chunk_pages(self, document_id, pages):
        return [
            Chunk(f"{document_id}:{i}", document_id, text, page=page, index=i)
            for i, (page, text) in enumerate(pages)
        ]

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(word.encode()) % self.DIM] += 1
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_texts(self, texts):
        return np.stack([self._embed(t) for t in texts])

    def embed_chunks(self, chunks):
        return self.embed_texts(chunks)

    def embed_query(self, query):
        return self._embed(query)


class EmptyIngestion(FakeIngestion):
    def extract_pages(self, source):
        return []