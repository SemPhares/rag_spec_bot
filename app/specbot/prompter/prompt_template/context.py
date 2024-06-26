
from langchain.prompts import PromptTemplate

context_template = PromptTemplate.from_template(
    """[INST]
    Context information is below.

    ---------------------
    {context_str}
    ---------------------

    You are a helpful  assistant. Your task is to answer the question based on the given context. 
    Restrict the questions to the context information provided.

    question: {query}
    answer: [/INST]
    """)


