from langchain.prompts import PromptTemplate
from typing import Union

from specbot.config import Config
from langchain_core.documents import Document


def doc_to_str(chunks: Union[Document, str]) -> str:
    """
    """
    if isinstance(chunks, Document):
        retrieved_chunks = ""
        for doc in chunks:
            retrieved_chunks+= f"\n {doc.page_content}"
        return retrieved_chunks
    else:
        return chunks


def build_rag_prompt(query : str, 
                 retrieved_chunks : str) :
    with open(Config.context_template, "r", encoding='utf-8') as f:
        content = f.read()
    retrieved_chunks = doc_to_str(retrieved_chunks)
    content = content.replace('{context_str}', retrieved_chunks)
    content = content.replace('{query}', query)
    # context_prompt = PromptTemplate.from_template(content)
    return content


def build_sum_prompt(texts : str) :
    with open(Config.summer_template, "r", encoding='utf-8') as f:
        content = f.read()
    texts = doc_to_str(texts)
    content = content.replace('{text_to_summarize}', texts)
    # summarize_prompt = PromptTemplate.from_template(content)
    return content