from langchain.prompts import PromptTemplate

summarize_template = PromptTemplate.from_template(
    """[INST]
    Please summarize the following text:

    ---------------------
    {text_to_summarize}
    ---------------------

    Your summary should be clear, precise and cover the main points of the text. Try to condense the information without omitting crucial elements."
    Answer: [/INST]
    """)







