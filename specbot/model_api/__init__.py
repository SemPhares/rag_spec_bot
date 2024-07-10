from .llm_typing import llm_input, llm_output, llama_cpp_local_input
from config.model_config import ModelConfig
from utils.usefull import spinner

@spinner
def ask_llm(ccp_or_ollama:str, 
            request:str) -> llm_output:
    """
    """
    if ccp_or_ollama not in ["llamacpp", "ollama", "gemini"]:
        raise ValueError("Invalid model name")
    
    elif ccp_or_ollama == "llamacpp":
        from .llamacpp_model import ask_llmcpp
        return ask_llmcpp(llama_cpp_local_input(
            llm_path=ModelConfig.LLAMA_CPP_BASE_MODEL_PATH,
            llm_name=ModelConfig.LLAMA_CPP_BASE_MODEL_NAME,
            input=request))
    
    elif ccp_or_ollama == "gemini":
        from .gemini_model import ask_gemini
        return ask_gemini(llm_input(llm_name=ModelConfig.GEMINI_MODEL_NAME,
                                    input=request))
    
    else:
        from .ollama_model import ask_ollama
        return ask_ollama(llm_input(llm_name=ModelConfig.OLLAMA_BASE_MODEL_NAME,
                                    input=request))