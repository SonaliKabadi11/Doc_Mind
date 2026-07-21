
import logging
import sys

import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.exception import CustomException

logger = logging.getLogger(__name__)


class Ingestion:
    def __init__(self):
      
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
           

    def extract_pdf(self, file_path: str) -> str:
        """
        Opens a PDF and concatenates text page by page.

        Parameters:
        file_path: str = path of the PDF

        Returns:
        text: str = content of the PDF
        """
        try:
            reader = PdfReader(file_path)
            number_of_pages = len(reader.pages)
            text = ""
            for page_number in range(number_of_pages):
                page = reader.pages[page_number]
                text += page.extract_text() or ""
           
            logger.info(
                "Extracted all the content from the PDF with %s pages and text length %s",
                number_of_pages,
                len(text),
            )
            print(f"Extracted all the content from the PDF with {number_of_pages} pages and text length {  len(text)}")
            return text
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

        

    def chunk_text(self, text: str) -> list[str]:
        """
        Creates the chunk of the PDF text.

        Parameters:
        text: str = content of the PDF

        Returns:
        list[str] = list of sentences/chunks
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " "],
                chunk_size=1000,
                chunk_overlap=200,
                keep_separator=True
            )
            chunks = splitter.split_text(text)
            
            logger.info("Converted text into chunks with length %s", len(chunks))
            print("Converted text into chunks with length ", len(chunks))
            # print("First chunk: ", chunks[0])
            return chunks
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def embed_chunks(self, chunks: list[str]) -> np.ndarray:
        """
        Embeds the chunks.

        Parameters:
        chunks: list[str] = chunks of text

        Returns:
        ndarray = vector array of floats
        """
        try:
            #  That normalize_embeddings=True flag matters directly: it's what makes your InMemoryStore's dot-product-based search mathematically equal to cosine similarity, matching what you already documented in base.py's docstrings.
            embeddings = self.model.encode(chunks, normalize_embeddings=True)
            print("Shape of embeddings:", embeddings.shape)
            logger.info("Embeddings are created with the shappe %s", embeddings.shape)

            return embeddings
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     object = Ingestion()
#     res = object.extract_pdf(str(pdf_path))
#     # print(res)
#     chunks = object.chunk_text(res)
#     embeddings = object.embed_chunks(chunks)
 