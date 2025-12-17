import streamlit as st
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "vectorstore"
OLLAMA_MODEL = "phi3:latest"
SIMILARITY_THRESHOLD = 1.0

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
@st.cache_resource
def get_retriever():
    db = load_vectorstore()
    return db.as_retriever(search_kwargs={"k":3})

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
            prompt=query,
            options={"num_predict": 256}
        )
        st.subheader("Response without RAG")
        st.write(response.response)  # Ollama response text

    elif mode == "With RAG":
        # Retrieve documents
        db = load_vectorstore()
        results = db.similarity_search_with_score(query, k=3)

        # Check best match score
        best_score = results[0][1]

        if best_score > SIMILARITY_THRESHOLD:    
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=query,
                options={"num_predict": 128}
            )
            st.subheader("Response (RAG skipped – low relevance)")
            st.write(response.response) 
        else:
            docs = [doc for doc, score in results]
            context_text = "\n\n".join([doc.page_content for doc in docs])

            # Combine query + context for RAG
            rag_prompt = f"Use the following context to concisely answer the question:\n\n{context_text}\n\nQuestion: {query}"

            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=rag_prompt,
                options={"num_predict": 512}
            )
            st.subheader("Response with RAG")
            st.write(response.response)