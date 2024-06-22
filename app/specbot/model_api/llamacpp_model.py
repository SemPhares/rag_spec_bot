from config import Config
from llama_cpp import Llama
from utils.log import logger
from .llm_typing import llam_cpp_local_input, llm_output


llama_cpp_config = {
    "n_gpu_layers": -1,
    "n_threads" : 4,
    "temperature": 0.75,
    "n_ctx": 1024,
    "split_mode": 0,

    "main_gpu": 2,
    "verbose": False}


def llamacpp_from_pretrained(repo_id:str,
                             filename:str) -> Llama:
    """
    """
    try:
        model = Llama.from_pretrained(
            repo_id= repo_id,
            filename=filename,
            **llama_cpp_config)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = Llama.from_pretrained(
            repo_id= repo_id,
            filename=filename,
            verbose=False) 
           
    return model


def llamacpp_for_embedding(model_path:str = "") -> Llama:
    """
    """
    
    model = llamacpp_from_pretrained(Config.EMBEDDING_MODEL_REPO_ID,
                                     Config.EMBEDDING_MODEL_FILENAME)
    return model


def ask_llmcpp(query:llam_cpp_local_input) -> llm_output:
    
    llama_cpp = Llama(model_path= query.model_path,
                      **llama_cpp_config)
    
    # llama_cpp = llamacpp_from_pretrained(query.repo_id,
    #                                      query.filename)

    output = llama_cpp.create_completion(query.input) 
    output = output["choices"][0]["text"] # type: ignore
    output = llm_output(response=output, model_name=query.model_name)
    return output

