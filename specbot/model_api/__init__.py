from specbot.model_api.llm_typing import llm_input, llm_output, llam_cpp_local_input
from specbot.config import ModelConfig
from specbot.utils.usefull import spinner

@spinner
def ask_llm(ccp_or_ollama:str, 
            request:str) -> llm_output:
    """
    """
    if ccp_or_ollama not in ["llamacpp", "ollama"]:
        raise ValueError("Invalid model name")
    
    elif ccp_or_ollama == "llamacpp":
        from .llamacpp_model import ask_llmcpp
        return ask_llmcpp(llam_cpp_local_input(
            model_path=ModelConfig.LLAMA_CPP_BASE_MODEL_PATH,
            model_name=ModelConfig.LLAMA_CPP_BASE_MODEL_NAME,
            input=request))
    else:
        from .ollama_model import ask_ollama
        return ask_ollama(llm_input(model_name=ccp_or_ollama,
                                    input=request))