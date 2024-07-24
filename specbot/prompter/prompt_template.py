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

    Your answer should strictily be one of the following categories: {categories}.
    You shoudl be strictly in your classification and ensure that the text fits the category provided.

    Here are somme examples :

     - 'Please resuem the text' -> 'SUMMARY'
     - 'Explain me the purspose of the document?' -> 'DISCUSSION'
     - 'What is the main idea of the text?' -> 'SUMMARY'
     - 'Explain the images of the document' -> 'IMAGE_EXTRACTION'

    Answer: 


    """)


rewrite_template = PromptTemplate.from_template(
    """

    Rewrite the following query to improve its clarity and specificity. 
    The goal is to refine the query so it yields more accurate and relevant search results. 
    Consider the context and aim for precision without altering the original intent. 
    If multiple refined queries can be derived, list them using bullet points. 
    If you don't have any room for improvement, simply return the original query.

    Original Query: {query}

    Refined Query:


    """)
