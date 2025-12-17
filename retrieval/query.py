from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

DB_PATH = "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k":3})

query = "Hi How are you doing?"

docs = retriever._get_relevant_documents(query, run_manager=None)

# combine chunks into a single context string
context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"

response_rag = ollama.generate(
    model = "phi3:latest",
    prompt = prompt
)

print("\nWith RAG:\n", response_rag)

