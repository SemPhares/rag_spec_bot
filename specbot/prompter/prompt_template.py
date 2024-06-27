from langchain.prompts import PromptTemplate

EXTRACT_IMAGE_PROMPT = "Describe the image in detail. Be specific about graphs, such as bar plots."

context_template = PromptTemplate.from_template(
    """[INST]
    Context information is below.

    ---------------------
    {context_str}
    ---------------------

    You are a helpful  assistant. Your task is to answer the question based on the given context. 
    Restrict the questions to the context information provided.

    question: {query}
    Answer: 
    [/INST]
    """)

summarize_template = PromptTemplate.from_template(
    """[INST]
    Summarize the following text:

    ---------------------
    {text_to_summarize}
    ---------------------

    Your summary should be clear, precise and cover the main points of the text. Try to condense the information without omitting crucial elements."
    Answer: 
    
    [/INST]
    """)


summarize_table= PromptTemplate.from_template(
    """[INST]
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table. 
    
    Table or text chunk is below:
    ---------------------
    {table}
    ---------------------
    Your summary should be clear, precise and cover the main points of the text."
    Answer: 
    [/INST]
    """)
