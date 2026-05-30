"""Dataset-grounded RAG engine and Kimi chat orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.career_recommender.paths import RAW_DATA_DIR
from backend.core.api_manager import APIManager
from backend.core.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DatasetChunk:
    """Searchable dataset text chunk with source metadata."""

    source: str
    row_id: int
    text: str
    metadata: Mapping[str, Any]


class DatasetRAGEngine:
    """Build and query a lightweight vector index over local career datasets."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, max_rows_per_file: int = 2500) -> None:
        """Initialize the RAG engine with a raw dataset directory."""
        self.raw_dir = raw_dir
        self.max_rows_per_file = max_rows_per_file
        self.chunks: list[DatasetChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None

    def discover_dataset_files(self, limit: int | None = None) -> list[Path]:
        """Return CSV datasets from the configured raw directory."""
        if not self.raw_dir.exists():
            return []
        files = sorted(path for path in self.raw_dir.glob("*.csv") if path.is_file())
        return files if limit is None else files[:limit]

    def build_index(self) -> int:
        """Load datasets and build a local TF-IDF vector index."""
        self.chunks = self._load_chunks()
        if not self.chunks:
            self._vectorizer = None
            self._matrix = None
            return 0
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        LOGGER.info("Built RAG index with %s chunks from %s.", len(self.chunks), self.raw_dir)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[DatasetChunk]:
        """Retrieve the most relevant dataset chunks for a user query."""
        if not query.strip() or top_k <= 0:
            return []
        if self._vectorizer is None or self._matrix is None:
            self.build_index()
        if self._vectorizer is None or self._matrix is None or not self.chunks:
            return []
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][:top_k]
        return [self.chunks[int(index)] for index in ranked_indexes if float(scores[int(index)]) > 0]

    def context_block(self, query: str, top_k: int = 5) -> str:
        """Return retrieved context formatted for a chat system prompt."""
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return "No relevant local dataset context was retrieved."
        lines: list[str] = []
        for chunk in chunks:
            lines.append(f"Source: {chunk.source} row {chunk.row_id}\n{chunk.text}")
        return "\n\n".join(lines)

    def _load_chunks(self) -> list[DatasetChunk]:
        """Read CSV files into compact row-level text chunks."""
        chunks: list[DatasetChunk] = []
        for path in self.discover_dataset_files():
            try:
                dataframe = pd.read_csv(path, nrows=self.max_rows_per_file)
            except Exception as exc:
                LOGGER.warning("Skipping dataset %s because it could not be read: %s", path.name, exc)
                continue
            for row_id, row in dataframe.fillna("").iterrows():
                text = self._row_to_text(row)
                if text:
                    chunks.append(
                        DatasetChunk(
                            source=path.name,
                            row_id=int(row_id),
                            text=text,
                            metadata={"columns": list(dataframe.columns)},
                        )
                    )
        return chunks

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        """Convert a dataframe row into a compact natural-language chunk."""
        parts: list[str] = []
        for column, value in row.items():
            cleaned = str(value).strip()
            if cleaned:
                parts.append(f"{column}: {cleaned}")
        return " | ".join(parts)


class KimiRAGChatbot:
    """Kimi-powered chatbot with local dataset context and session history support."""

    def __init__(
        self,
        api_manager: APIManager | None = None,
        rag_engine: DatasetRAGEngine | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize chatbot dependencies."""
        self.api_manager = api_manager or APIManager()
        self.rag_engine = rag_engine or DatasetRAGEngine()
        self.model = model

    def answer(
        self,
        user_query: str,
        history: Sequence[Mapping[str, str]] | None = None,
        uploaded_resume_context: str = "",
        internet_context: str = "",
    ) -> str:
        """Answer a user query using local dataset context, optional resume text, and Kimi."""
        dataset_context = self.rag_engine.context_block(user_query, top_k=5)
        system_prompt = (
            "You are an AI career advisor. Ground answers in the supplied local dataset context when it is relevant. "
            "If the dataset is insufficient, state the limitation and give practical next steps. "
            "Do not claim live internet access unless internet context is explicitly supplied.\n\n"
            f"Local dataset context:\n{dataset_context}\n\n"
            f"Uploaded resume context:\n{uploaded_resume_context[:5000] or 'None'}\n\n"
            f"Internet context supplied by app:\n{internet_context[:3000] or 'None'}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in list(history or [])[-10:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_query})
        provider = self.api_manager.preferred_chat_provider()
        model = self.model or self.api_manager.provider_model(provider)
        return self.api_manager.chat_completion(
            messages=messages,
            provider=provider,
            model=model,
            max_tokens=1200,
            temperature=0.25,
            extra_body={"thinking": {"type": "disabled"}},
        )


def build_mock_interview_prompt(target_role: str, latest_answer: str, history: Sequence[Mapping[str, str]]) -> str:
    """Build a grounded multi-turn mock interview prompt for the chatbot."""
    recent_history = "\n".join(f"{item.get('role', '')}: {item.get('content', '')}" for item in list(history)[-6:])
    return (
        "Continue a realistic mock interview. Ask one concise follow-up question after briefly evaluating the latest answer. "
        "Use the target role and conversation history.\n\n"
        f"Target role: {target_role}\nConversation:\n{recent_history}\nLatest answer:\n{latest_answer}"
    )
