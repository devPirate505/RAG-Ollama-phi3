import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import  HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/raw"
DB_path = "vectorstore"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
            documents.extend(loader.load())
    
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )

    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
         model_name = "sentence-transformers/all-MiniLM-L6-v2" 
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_path)

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)
    print("Success! Document ingestion complete. Vector store created.")
