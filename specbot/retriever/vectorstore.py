from pathlib import Path
from utils.log import logger
from .doc_spliter import text_splitter
from utils.usefull import spinner, timer
from config.model_config import ApiConfig
from config.global_config import GlobalConfig
from langchain_core.documents import Document
from .doc_transformer import others_transformer, custom_transformer
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.vectorstores.utils import filter_complex_metadata
from prompter.prompt_tuner import rewrite_query, extract_category
from langchain_community.document_compressors.jina_rerank import JinaRerank

# Get the embedder
embedder = others_transformer.ollama_embeder
# embedder = custom_transformer.gemini_embeder()



def filter_and_split(documents: list[Document]) -> list[Document]:
    """
    """
    # Filter out documents with complex metadata
    documents = filter_complex_metadata(documents)
    documents = text_splitter.split_documents(documents)
    return documents


def doc_ids(documents: list[Document]) -> str:
    """
    Construit un identifiant unique sur la base des metadata de tous des documents
    """
    unique_source = set()
    for doc in documents:
        unique_source.add(doc.metadata["source"])
    
    _id = "_".join(list(unique_source))

    docs_id = f"{len(documents)}-{_id}.db"

    logger.info(f"Documents ID created: {docs_id}")
    return docs_id


def get_vector_store(documents: list[Document]) -> Chroma:
    """
    """
    documents = filter_and_split(documents)
    # Create a Chroma vector store and save embeddings
    store_id = Path("specbot/store/vectorstore/", doc_ids(documents))
    logger.info(f"Vector store ID: {store_id}")
    
    if not store_id.exists():
        store_id.mkdir(parents=True)
        # store_id.touch()
        vector_store = Chroma.from_documents(documents = documents,
                                             embedding = embedder,
                                             persist_directory = str(store_id))
        globals()[f"vector_store_{store_id}"] = vector_store
    
    elif f"retriever_{store_id}" in globals():
        logger.info(f"Vector store already exists at {store_id} in globals")
        vector_store = globals()[f"vector_store_{store_id}"]
    
    else:
        # PAS NECESSAIRE MAIS BON, VOILA QUAND MEME
        logger.info(f"Vector store already exists at {store_id}")
        vector_store = Chroma(persist_directory = str(store_id), 
                              embedding_function = embedder)
        
    return vector_store


def get_retriever(documents: list[Document]) -> VectorStoreRetriever:
    """
    """

    # Get the vector store
    vector_store = get_vector_store(documents)
    # Create a retriever
    retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": GlobalConfig.RETRIEVER_TOP_K,
                "score_threshold": GlobalConfig.RETRIEVER_SCORE_THRESHOLD,
            },
                )   
    return retriever


def rerank_docs(retriever:VectorStoreRetriever,
                query: str) -> list[Document]:
    """
    """
    # Compress the retrieved documents
    compressor = JinaRerank(jina_api_key = ApiConfig.JINA_API_KEY)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever= retriever)

    compressed_docs = compression_retriever.invoke(query)
    return compressed_docs


@timer
@spinner
def retrieve_docs(query:str,
                  documents: list[Document]) -> list[Document]:
    """
    
    """
    query_category = extract_category(query)
    logger.info(f"Extracted category: {query_category}")

    if query_category == "SUMMARY":
        return documents
    
    else:
        # Get the retriever
        retriever = get_retriever(documents)
        # Rewrite the query
        query = rewrite_query(query)
        # Rerank the documents
        retrieved_docs = rerank_docs(retriever, query)
        # Retrieve documents
        retrieved_docs = retriever.invoke(query)
        
        return retrieved_docs


