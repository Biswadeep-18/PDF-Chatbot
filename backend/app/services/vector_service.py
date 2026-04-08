from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Tuple
from ..core.config import settings

class VectorService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def create_vector_store(self, texts_by_source: Dict[str, str]) -> FAISS:
        """Create a FAISS vector store from texts"""
        documents = []
        
        for source, text in texts_by_source.items():
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": source}
                ))
        
        return FAISS.from_documents(documents, self.embeddings)

    def get_context(self, vectorstore: FAISS, query: str, k: int = 10) -> str:
        """Retrieve context from vector store"""
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([
            f"### From {d.metadata.get('source', 'Unknown')}:\n{d.page_content}" 
            for d in docs
        ])
        return context
