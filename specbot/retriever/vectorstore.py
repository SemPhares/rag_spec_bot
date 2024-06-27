from .spliter import text_splitter
from utils.usefull import spinner, timer
from langchain_core.documents import Document
from .doc_transformer import others_transformer
from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.jina_rerank import JinaRerank

# Get the embedder
embedder = others_transformer.ollama_embeder


def filter_and_split(documents: list[Document]) -> list[Document]:
    """
    """
    # Filter out documents with complex metadata
    documents = filter_complex_metadata(documents)
    documents = text_splitter.split_documents(documents)
    return documents


def get_retriever(documents: list[Document]) -> VectorStoreRetriever:
    """
    """
    documents = filter_and_split(documents)
    # Create a FAISS vector store and save embeddings
    vector_store = FAISS.from_documents(documents, embedder) # type: ignore
    # Create a retriever
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 10,
                "score_threshold": 0.7,
            },
        )
    globals()["retriever"] = retriever # type: ignore
    return retriever


def rerank_docs(retriever:VectorStoreRetriever,
                query: str) -> list[Document]:
    """
    """
    # Compress the retrieved documents
    compressor = JinaRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever= retriever)

    compressed_docs = compression_retriever.invoke(query)
    return compressed_docs


@timer
@spinner
def retrieve_docs(query:str, 
                  documents: list) -> list[Document]:
    """
    
    """

    documents = filter_and_split(documents)
    # Create a retriever
    retriever = get_retriever(documents)
    # Rerank the documents
    retrieved_docs = rerank_docs(retriever, query)
    # Retrieve documents
    retrieved_docs = retriever.invoke(query)
    
    return retrieved_docs


