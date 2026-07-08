from abc import ABC, abstractmethod

class Retrieval(ABC):
    @abstractmethod
    def add_chunks(document_id, chunks, embeddings):
        print(f"Document_id: {document_id}, chunk_length: {len(chunks)} Embedding Shape: {embeddings.shape}")
    
    @abstractmethod
    def search(document_id, query_embedding, top_k):
        print(f"Document_id: {document_id}, top_k: {top_k}")

    @abstractmethod
    def delete_document(document_id):
        print(f"Deleting the document with id: {document_id}")