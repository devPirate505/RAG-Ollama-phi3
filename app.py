import streamlit as st
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

OLLAMA_MODEL = "phi3:latest"

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k":3})

# Streamlit UI

st.title("RAG Demo: With vs Without RAG")

st.write(
    """
    This demo lets you ask questions and see responses **with or without Retrieval-Augmented Generation (RAG)**.
    """
)

query = st.text_input("Enter your query:")

mode = st.radio("Select mode:", ["Without RAG", "With RAG"])


if query:
    if mode == "Without RAG":
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=query
        )
        st.subheader("Response without RAG")
        st.write(response.response)  # Ollama response text

    elif mode == "With RAG":
        # Retrieve documents
        docs = retriever._get_relevant_documents(query, run_manager=None)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        # Combine query + context for RAG
        rag_prompt = f"Use the following context to answer the question:\n\n{context_text}\n\nQuestion: {query}"

        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=rag_prompt
        )
        st.subheader("Response with RAG")
        st.write(response.response)