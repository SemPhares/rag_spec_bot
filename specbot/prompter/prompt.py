from specbot.utils.usefull import Union, List
from langchain_core.documents import Document
from specbot.prompter.prompt_template.context import context_template
from specbot.prompter.prompt_template.summarize import summarize_template


def doc_to_str(chunks: Union[List[Document], str]) -> str:
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
                     retrieved_chunks : Union[List[Document], str]) :
    
    context_prompt = context_template.format(context_str=doc_to_str(retrieved_chunks), 
                                             query=query)
    return context_prompt


def build_sum_prompt(texts : str) :
    summarize_prompt = summarize_template.format(text_to_summarize=doc_to_str(texts))
    return summarize_prompt