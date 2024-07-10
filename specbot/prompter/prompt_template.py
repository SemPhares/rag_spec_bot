from langchain.prompts import PromptTemplate

EXTRACT_IMAGE_PROMPT = "Describe the image in detail. Be specific about graphs, such as bar plots."


context_template = PromptTemplate.from_template(
    """

    Context information is below.

    ---------------------
    {context_str}
    ---------------------

    You are a helpful  assistant. Your task is to answer the question based on the given context. 
    Restrict the questions to the context information provided.

    question: {query}
    Answer:

    """)

summarize_template = PromptTemplate.from_template(
    """

    Summarize the following text: using this query as a guide: {query}

    ---------------------
    {text_to_summarize}
    ---------------------

    Your summary should be clear, precise and cover the main points of the text. Try to condense the information without omitting crucial elements."
    Answer: 
    
    """)


summarize_table= PromptTemplate.from_template(
    """

    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table. 
    
    Table or text chunk is below:
    ---------------------
    {table}
    ---------------------
    Your summary should be clear, precise and cover the main points of the text."
    Answer: 


    """)


classify_template = PromptTemplate.from_template(
    """

    Classify the following text into one of the following categories: {categories}

    Text to classificy is below:
    ---------------------
    {text_to_classify}
    ---------------------

    Your answer should strictily be one of the following categories: {categories}
    Answer: 


    """)


rewrite_template = PromptTemplate.from_template(
    """

    Provide a better search query for the given query.
    Ensure that your response is clear and concise and consiste only of the query, nothing more.
    If you don't have any suggestions, you can leave the answer and retun the original query itself.
    
    Query: {query} 

    Answer:

    """)
