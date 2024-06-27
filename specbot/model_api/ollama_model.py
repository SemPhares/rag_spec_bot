from model_api.model_utils import prompt_func
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.ollama import ChatOllama
from .llm_typing import llm_input, llm_output, llm_image_input


def ollama_caption_image(query:llm_image_input) -> llm_output:
    """
    """
    model = ChatOllama(model=query.model_name,
                       temperature=0.3,
                       top_k=30,
                       num_ctx = 512,
                       num_gpu = -1,
                       repeat_penalty = 1.2,
                       top_p = 0.7)
        
    # Create the chain with the prompt function, model, and output parser
    chain = prompt_func | model | StrOutputParser()

    # Invoke the chain with the text and image data
    response = chain.invoke({"text": query.input, "image_path": query.image_path})
    output = llm_output(response = str(response), 
                        model_name = query.model_name)
    return output


def ask_ollama(query:llm_input) -> llm_output:
    """
    """
    model = ChatOllama(model=query.model_name,
                       temperature=0.3,
                       top_k=30,
                       num_ctx = 512,
                       # The number of GPUs to use. 
                       # On macOS it defaults to 1 to enable metal support, 0 to disable.
                       num_gpu = -1,
                       repeat_penalty = 1.2,
                       top_p = 0.7)
    
    chain =  model | StrOutputParser()
    response = chain.invoke(query.input)
    output = llm_output(response = str(response), 
                        model_name = query.model_name)
    return output

