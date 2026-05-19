"""
RAG retrieval component.
Supports: mock (default for CI) and ChromaDB backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from pipeline.config import PipelineConfig


@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    score: float
    source: str


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        ...


# ---------------------------------------------------------------------------
# Sample knowledge base for the mock retriever
# ---------------------------------------------------------------------------
_MOCK_KB = [
    RetrievedDoc("doc_001", "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.", 0.95, "kb/geography.txt"),
    RetrievedDoc("doc_002", "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.", 0.93, "kb/programming.txt"),
    RetrievedDoc("doc_003", "Machine learning is a subset of artificial intelligence that enables systems to learn from data.", 0.91, "kb/ai.txt"),
    RetrievedDoc("doc_004", "The Great Wall of China stretches over 13,000 miles and was built over many centuries.", 0.89, "kb/history.txt"),
    RetrievedDoc("doc_005", "Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce oxygen and energy.", 0.88, "kb/biology.txt"),
    RetrievedDoc("doc_006", "The speed of light in a vacuum is approximately 299,792,458 meters per second.", 0.87, "kb/physics.txt"),
    RetrievedDoc("doc_007", "DNA, or deoxyribonucleic acid, carries genetic instructions for the development of all living organisms.", 0.86, "kb/biology.txt"),
    RetrievedDoc("doc_008", "The French Revolution began in 1789 and led to the rise of Napoleon Bonaparte.", 0.85, "kb/history.txt"),
    RetrievedDoc("doc_009", "GPT (Generative Pre-trained Transformer) models are large language models trained on vast text corpora.", 0.92, "kb/ai.txt"),
    RetrievedDoc("doc_010", "Transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al., 2017.", 0.90, "kb/ai.txt"),
]


class MockRetriever(BaseRetriever):
    """Returns semantically plausible documents from the in-memory KB."""

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        # Simple keyword overlap scoring for mock
        q_words = set(query.lower().split())
        scored = []
        for doc in _MOCK_KB:
            d_words = set(doc.content.lower().split())
            overlap = len(q_words & d_words) / max(len(q_words), 1)
            scored.append((overlap + doc.score * 0.1, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class ChromaRetriever(BaseRetriever):
    """ChromaDB-backed retriever — requires chromadb package."""

    def __init__(self, config: PipelineConfig):
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self._client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self._collection = self._client.get_or_create_collection("rag_kb", embedding_function=ef)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        results = self._collection.query(query_texts=[query], n_results=top_k)
        docs = []
        for i, (doc_id, content, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
        )):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            docs.append(RetrievedDoc(
                doc_id=doc_id,
                content=content,
                score=1.0 - distance,
                source=meta.get("source", "unknown"),
            ))
        return docs


def get_retriever(config: PipelineConfig | None = None) -> BaseRetriever:
    cfg = config or PipelineConfig()
    if cfg.rag_backend.lower() == "chromadb":
        return ChromaRetriever(cfg)
    return MockRetriever()
