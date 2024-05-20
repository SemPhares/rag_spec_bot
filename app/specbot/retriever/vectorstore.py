from transformer import sentence_transfomer, llm_transformer, others_transformer
from langchain_community.vectorstores.faiss import FAISS

config = {
    "store_path" : "app/specbot/retriever/store",
}

# db = FAISS.from_documents(docs, embeddings)