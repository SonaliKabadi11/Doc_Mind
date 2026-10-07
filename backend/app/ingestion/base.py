
import logging
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.models import Chunk

logger = logging.getLogger(__name__)



class Ingestion:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " "],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=True
            )


    def extract_pages(self, source: str | Path | BinaryIO) -> list[tuple[int, str]]:
        """
        Opens a PDF and concatenates text page by page.

        Parameters:
        file_path: str = path of the PDF

        Returns:
        [(page_number, text), ...]: list[tuple[int, str]] =  for pages that contain text (1-based).
        """
        
        reader = PdfReader(source)
        pages: list[tuple[int, str]] = []
        for page_no, page in enumerate(reader.pages, start =1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((page_no, text))
          
        logger.info("Extracted text from %d/%d pages", len(pages), len(reader.pages))
        return pages      

    def chunk_pages(self, document_id: str, pages: list[tuple[int, str]]) -> list[Chunk]:
        """
        Chunk each page separately so every chunk maps to exactly one page.

        Parameters:
        document_id: str = id of the document 
        pages: list[tuple[int, str]] = list of page_number and respective extracted content

        Returns:
        list[Chunk] = list of chunks 
        """
        chunks : list[Chunk] = []
        for page_no, text in pages:
            for piece in self.splitter.split_text(text):
                if not piece.strip():
                    continue
                idx = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id = f"{document_id}:{idx}",
                        document_id = document_id,
                        text = piece,
                        page = page_no,  # 1-based
                        index= idx  # position within the document
                    )
                )

        logger.info("Created %d chunks", len(chunks))
        return chunks

    def embed_chunks(self, chunks: list[str]) -> np.ndarray:
        """
        Embeds the chunks.

        Parameters:
        chunks: list[str] = chunks of text

        Returns:
        ndarray = vector array of floats
        """
        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )
        logger.info("Embeddings are created with the shappe %s", embeddings.shape)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds the user query

        Parameters:
        query: str = user query 

        Returns:
        ndarray = vector array of floats
        """
        query_embedding = self.model.encode(query, normalize_embeddings=True )  # shape (dim,)
        logger.info("Query Embedding is created with the shappe %s", query_embedding.shape)
        return query_embedding
    
# if __name__ == "__main__":
#     object = Ingestion()
#     res = object.extract_pdf("Doc_Mind/backend/app/data/HP1.pdf")
#     # print(res)
#     chunks = object.chunk_text(res)
#     embeddings = object.embed_chunks(chunks)

 