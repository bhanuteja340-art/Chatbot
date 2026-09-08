import hashlib
from pathlib import Path
from typing import Any

import chromadb


VECTOR_STORE_PATH = (
    Path(__file__).resolve().parent.parent / "vector_store"
)

COLLECTION_NAME = "interview_knowledge_base"


class VectorStore:
    def __init__(self):
        VECTOR_STORE_PATH.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def clear(self) -> None:
        ids = self.collection.get()["ids"]

        if ids:
            self.collection.delete(ids=ids)

    def document_id(
        self,
        chunk: dict[str, Any],
    ) -> str:
        return self._build_id(chunk)

    def existing_ids(
        self,
        ids: list[str],
    ) -> set[str]:
        if not ids:
            return set()

        result = self.collection.get(ids=ids)

        return set(result.get("ids", []))

    @staticmethod
    def _build_id(
        chunk: dict[str, Any],
    ) -> str:
        metadata = chunk["metadata"]
        chunk_id = metadata["chunk_id"]

        source = (
            metadata.get("source")
            or metadata.get("document_id")
            or metadata.get("file_name")
            or metadata.get("pdf_hash")
        )

        if source is not None:
            return f"{source}_{chunk_id}"

        text_hash = hashlib.sha256(
            chunk["text"].encode("utf-8")
        ).hexdigest()[:12]

        return f"{text_hash}_{chunk_id}"

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be the same."
            )

        if not chunks:
            raise ValueError("No chunks provided.")

        ids = [
            self._build_id(chunk)
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            chunk["metadata"]
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict[str, Any]:
        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        count = self.collection.count()

        if count == 0:
            raise RuntimeError(
                "Vector store is empty."
            )

        top_k = min(top_k, count)

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        return self.collection.count()