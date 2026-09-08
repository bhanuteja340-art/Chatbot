import time
from typing import Any

from src.generator import Generator
from src.retriever import Retriever


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def answer(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        start_time = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(question)
        retrieval_latency = time.perf_counter() - retrieval_start

        generation_start = time.perf_counter()
        answer = self.generator.generate(question, retrieved_chunks)
        generation_latency = time.perf_counter() - generation_start

        total_latency = time.perf_counter() - start_time

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "total_latency": total_latency,
            "status": "success",
            "error": None,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    question = input("Enter your question: ").strip()
    result = pipeline.answer(question)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("PERFORMANCE")
    print("=" * 80)
    print(f"Retrieval latency: {result['retrieval_latency']:.3f} seconds")
    print(f"Generation latency: {result['generation_latency']:.3f} seconds")
    print(f"Total latency: {result['total_latency']:.3f} seconds")
    print(f"Status: {result['status']}")

    print("\n" + "=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for index, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(f"\nRank: {index}")
        print(f"Page: {chunk['metadata'].get('page')}")
        print(f"Distance: {chunk['distance']}")
        print(chunk["text"])