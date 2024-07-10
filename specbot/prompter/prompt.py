from typing import Union, List
from langchain_core.documents import Document
from .prompt_template import (context_template, 
                              summarize_template, 
                              classify_template,
                              rewrite_template)
from .prompt_typing import (prompt_input, prompt_output,
                            classification_prompt_input)

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


def build_rag_prompt(input :prompt_input) -> prompt_output :
    
    context_prompt = context_template.format(context_str=doc_to_str(input.retrieved_chunks), 
                                             query=input.query)
    return context_prompt


def build_classification_prompt(input :classification_prompt_input) -> prompt_output :
      
    classification_template = classify_template.format(categories=doc_to_str(input.categories),
                                                       text_to_classify=input.query)
    return classification_template


def build_rewrite_prompt(query :str):
    rewrite_prompt = rewrite_template.format(query=query)
    return rewrite_prompt


def build_summary_prompt(input :prompt_input) -> prompt_output :
    summarize_prompt = summarize_template.format(text_to_summarize=doc_to_str(input.retrieved_chunks),
                                                 query=input.query)
    return summarize_prompt