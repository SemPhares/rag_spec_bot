from .doc_transformer import sentence_transfomer, llm_transformer, others_transformer
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from utils.usefull import spinner


embedder = others_transformer.ollama_embeder

@spinner
def retrieve_docs(query:str, 
                  documents: list) -> list[Document]:
    """
    
    """

    # Create a FAISS vector store and save embeddings
    vector_store = FAISS.from_documents(documents, embedder) # type: ignore
    # Create a retriever
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7,
            },
        )
    # Retrieve documents
    retrieved_docs = retriever.invoke(query)
    
    return retrieved_docs