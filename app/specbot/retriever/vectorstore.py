from .doc_transformer import sentence_transfomer, llm_transformer, others_transformer
from langchain_community.vectorstores.faiss import FAISS

config = {
    "store_path" : "app/specbot/retriever/store",
}


def retrieve_docs(query:str, 
                  vector_store):
    """
    
    """
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
    # Retrieve documents
    retrieved_docs = retriever.invoke(query)
    
    return retrieved_docs
