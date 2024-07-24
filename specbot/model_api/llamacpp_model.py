from time import time
from llama_cpp import Llama
from utils.log import logger
from config.global_config import GlobalConfig
from .model_utils import encode_image
from .llm_typing import llama_cpp_local_input, llm_output, llama_cpp_image_input


llama_cpp_config = {
    "n_gpu_layers": -1,
    "n_threads" : 4,
    "temperature": GlobalConfig.TEMPERATURE,
    "n_ctx": GlobalConfig.CONTEXT_WINDOW,
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


def llamacpp_embedder(llm_path:str) -> Llama:
    """
    """
    kwargs = llama_cpp_config.copy()
    kwargs.update({"embedding": True})
    model = Llama(model_path= llm_path,
                  **llama_cpp_config)    
    return model


def llamacpp_for_caption(query:llama_cpp_image_input) -> llm_output:
    """
    """
    
    model = llamacpp_from_pretrained(query.repo_id,
                                     query.filename)
    image_bs4 = encode_image(query.image_path)

    start = time()
    response = model.create_chat_completion(
        messages = [
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text": query.input},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bs4}" } }
                ]
            }
        ],
        temperature=GlobalConfig.TEMPERATURE
    )
    end = time()

    caption = response["choices"][0]["message"]['content']

    output = llm_output(response=caption, 
                        llm_name=query.llm_name,
                        generation_time= end - start)
    return output


def ask_llmcpp(query:llama_cpp_local_input) -> llm_output:
    
    llama_cpp = Llama(model_path= query.llm_path,
                      **llama_cpp_config)

    start = time()
    output = llama_cpp.create_completion(query.input) 
    end = time()
    
    output = output["choices"][0]["text"]
    output = llm_output(response=output,
                        llm_name=query.llm_name,
                        generation_time= end - start)
    return output

