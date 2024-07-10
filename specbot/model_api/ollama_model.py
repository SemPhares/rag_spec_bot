from config.global_config import GlobalConfig
from model_api.model_utils import prompt_func
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.chat_models.ollama import ChatOllama
from .llm_typing import llm_input, llm_output, llm_image_input


def ollama_caption_image(query:llm_image_input) -> llm_output:
    """
    """
    model = ChatOllama(model=query.llm_name,
                       temperature=GlobalConfig.TEMPERATURE,
                       top_k=50,
                       num_ctx = GlobalConfig.CONTEXT_WINDOW,
                       num_gpu = GlobalConfig.NUM_GPU,
                       repeat_penalty = 1.2,
                       top_p = 0.7)
        
    # Create the chain with the prompt function, model, and output parser
    chain = prompt_func | model | StrOutputParser()

    # Invoke the chain with the text and image data
    response = chain.invoke({"text": query.input, "image_path": query.image_path})
    output = llm_output(response = str(response), 
                        llm_name = query.llm_name)
    return output


def clasify_with_ollama(query:llm_input) -> llm_output:
    """
    """
    model = ChatOllama(model=query.llm_name,
                       temperature=GlobalConfig.CONSERVATIVE_TEMPERATURE,
                       top_k=30,
                       num_ctx = GlobalConfig.CONTEXT_WINDOW,
                       num_gpu = GlobalConfig.NUM_GPU,
                       repeat_penalty = 1.2,
                       top_p = 0.7)
    
    chain =  model | StrOutputParser()
    response = chain.invoke(query.input)
    output = llm_output(response = response, 
                        llm_name = query.llm_name)
    
    return output


def ask_ollama(query:llm_input) -> llm_output:
    """
    """
    model = ChatOllama(model=query.llm_name,
                       temperature=GlobalConfig.TEMPERATURE,
                       top_k=30,
                       num_ctx = GlobalConfig.CONTEXT_WINDOW,
                       # The number of GPUs to use. 
                       # On macOS it defaults to 1 to enable metal support, 0 to disable.
                       num_gpu = GlobalConfig.NUM_GPU,
                       repeat_penalty = 1.2,
                       top_p = 0.7)
    
    chain =  model | StrOutputParser()
    response = chain.invoke(query.input)
    response = response.replace("[/INST]", "")
    output = llm_output(response = response, 
                        llm_name = query.llm_name)
    return output

