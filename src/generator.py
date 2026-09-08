import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors,types


load_dotenv()


GENERATION_MODEL = "gemini-3.1-flash-lite"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504}


class Generator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not retrieved_chunks:
            raise ValueError(
                "Retrieved chunks cannot be empty."
            )

        context_parts = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            context_parts.append(
                f"Context {index}:\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        return f"""You are a question-answering assistant.

Answer the user's question using only the provided knowledge-base context.

If the context does not contain enough information to answer the question, say:
"I don't have enough information in the knowledge base to answer that."

Do not invent facts or use information that is not supported by the provided context.

Knowledge-base context:

{context}

User question:
{question}

Answer:"""

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        prompt = self.build_prompt(
            question,
            retrieved_chunks,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                    }
                )

                answer = response.text

                if not answer or not answer.strip():
                    raise RuntimeError(
                        "The generation model returned an empty answer."
                    )

                return answer.strip()

            except errors.APIError as exc:
                if exc.code not in RETRYABLE_STATUS_CODES:
                    raise

                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Generation failed after {MAX_RETRIES} attempts "
                        f"with status {exc.code}."
                    ) from exc

                delay = RETRY_DELAY_SECONDS * (2 ** attempt)

                print(
                    f"Generation request failed with status {exc.code}. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

        raise RuntimeError("Generation failed.")