from llama_cpp import Llama
from utils.log import logger
from config import ModelConfig
from .llm_typing import llam_cpp_local_input, llm_output, llama_cpp_image_input


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


def llamacpp_for_embedding() -> Llama:
    """
    """
    
    model = llamacpp_from_pretrained(ModelConfig.EMBEDDING_MODEL_REPO_ID,
                                     ModelConfig.EMBEDDING_MODEL_FILENAME)
    return model


def llamacpp_for_caption(query:llama_cpp_image_input) -> llm_output:
    """
    """
    
    model = llamacpp_from_pretrained(query.repo_id,
                                     query.filename)
    
    response = model.create_chat_completion(
        messages = [
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text": query.input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query.image_bs4}" } }
                ]
            }
        ],
        temperature=0.2,
    )
    caption:str = response["choices"][0]["message"]['content'] # type: ignore
    output = llm_output(response=caption, model_name=query.model_name)
    return output


def ask_llmcpp(query:llam_cpp_local_input) -> llm_output:
    
    llama_cpp = Llama(model_path= query.model_path,
                      **llama_cpp_config)
    
    # llama_cpp = llamacpp_from_pretrained(query.repo_id,
    #                                      query.filename)

    output = llama_cpp.create_completion(query.input) 
    output = output["choices"][0]["text"] # type: ignore
    output = llm_output(response=output, model_name=query.model_name)
    return output

