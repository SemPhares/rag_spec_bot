from typing import Union
from .prompt_template.context import context_template
from .prompt_template.summarize import summarize_template
from langchain_core.documents import Document

config = {
    "context_template" : "app/specbot/prompter/template/context.txt",
    "summer_template" : "app/specbot/prompter/template/summarize.txt"
}


def doc_to_str(chunks: Union[Document, str]) -> str:
    """
    """
    if isinstance(chunks, list) :
        retrieved_chunks = ""
        for doc in chunks:
            if isinstance(doc, Document):
                retrieved_chunks+= f"\n {doc.page_content}"
            elif isinstance(doc, str):
                retrieved_chunks+= f"\n {doc}"
        return retrieved_chunks
    else:
        return chunks


def build_rag_prompt(query : str, 
                 retrieved_chunks : Union[Document, str]) :
    # with open(config['context_template'], "r", encoding='utf-8') as f:
    #     content = f.read()
    # retrieved_chunks = doc_to_str(retrieved_chunks)
    # content = content.replace('{context_str}', doc_to_str(retrieved_chunks))
    # content = content.replace('{query}', query)
    context_prompt = context_template.format(context_str=doc_to_str(retrieved_chunks), 
                                             query=query)

    return context_prompt


def build_sum_prompt(texts : str) :
    # with open(config['summer_template'], "r", encoding='utf-8') as f:
    #     content = f.read()
    # texts = doc_to_str(texts)
    # content = content.replace('{text_to_summarize}', doc_to_str(texts))
    # summarize_prompt = PromptTemplate.from_template(content)
    summarize_prompt = summarize_template.format(text_to_summarize=doc_to_str(texts))
    return summarize_prompt