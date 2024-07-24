from utils.usefull import spinner
from typing import Literal
from config.model_config import ModelConfig
from .llm_typing import llm_input, llm_output, llama_cpp_local_input

@spinner
def ask_llm(framework:Literal["llamacpp", "ollama", "gemini"], 
            request:str,
            conservative_mode:bool = False) -> llm_output:
    """
    """
    
    if framework not in ["llamacpp", "ollama", "gemini"]:
        raise ValueError("Invalid framework")
    
    elif framework == "llamacpp":
        from .llamacpp_model import ask_llmcpp
        return ask_llmcpp(llama_cpp_local_input(
            llm_path=ModelConfig.LLAMA_CPP_BASE_MODEL_PATH,
            llm_name=ModelConfig.LLAMA_CPP_BASE_MODEL_NAME,
            input=request))
    
    elif framework == "gemini":
        from .gemini_model import ask_gemini
        return ask_gemini(llm_input(llm_name=ModelConfig.GEMINI_MODEL_NAME,
                                    input=request),
                                    conservative_mode)
    
    else:
        from .ollama_model import ask_ollama
        return ask_ollama(llm_input(llm_name=ModelConfig.OLLAMA_BASE_MODEL_NAME,
                                    input=request),
                                    conservative_mode)