import streamlit as st

from src.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="AI",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }

    .brand {
        font-size: 18px;
        font-weight: 700;
        color: #24527a;
        margin-bottom: 20px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #17324d;
        margin-bottom: 25px;
    }

    .section-title {
        color: #17324d;
        font-size: 24px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .answer-box {
        background-color: #ffffff;
        border: 1px solid #dce5ed;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
    }

    .footer {
        text-align: center;
        color: #7a8b99;
        font-size: 14px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #dce5ed;
    }

    div.stButton > button {
        background-color: #24527a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 24px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1c4263;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="brand">Heuristiclabs.ai</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">AI Interview Assistant</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except Exception:
    st.error(
        "Could not initialize the RAG pipeline. "
        "Check your API key, knowledge base, and vector store."
    )
    st.stop()


question = st.text_input(
    "Enter your question",
    placeholder="Example: What are your strengths?",
)


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        result = pipeline.answer(question)

        st.markdown(
            '<div class="section-title">Answer</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="answer-box">
                {result["answer"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Performance</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Retrieval",
            f"{result['retrieval_latency']:.3f}s",
        )

        col2.metric(
            "Generation",
            f"{result['generation_latency']:.3f}s",
        )

        col3.metric(
            "Total",
            f"{result['total_latency']:.3f}s",
        )

        col4.metric(
            "Status",
            result["status"],
        )

        st.markdown(
            '<div class="section-title">Retrieved Sources</div>',
            unsafe_allow_html=True,
        )

        for index, chunk in enumerate(
            result["retrieved_chunks"],
            start=1,
        ):
            metadata = chunk.get("metadata", {})
            page = metadata.get("page")
            distance = chunk.get("distance", 0)

            with st.expander(
                f"Rank {index} | Page {page} | Distance {distance:.4f}"
            ):
                st.write(chunk.get("text", ""))

    except RuntimeError as exc:
        st.error(f"Couldn't answer the question: {exc}")

    except Exception:
        st.error(
            "Something went wrong while processing the question."
        )


st.markdown(
    '<div class="footer">Powered by heuristiclabs.ai</div>',
    unsafe_allow_html=True,
)