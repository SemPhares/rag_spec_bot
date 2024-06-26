from langchain_community.chat_models.ollama import ChatOllama
from .llm_typing import llm_input, llm_output


def ask_ollama(query:llm_input) -> llm_output:
    """
    """
    model = ChatOllama(model=query.model_name,
                       temperature=0.3,
                       top_k=30,
                       num_ctx = 512,
                       # The number of GPUs to use. 
                       # On macOS it defaults to 1 to enable metal support, 0 to disable.
                       num_gpu = 1,
                       repeat_penalty = 1.2,
                       top_p = 0.7)
    
    output = model.invoke(query.input)
    output = llm_output(response = str(output.content), 
                        model_name = query.model_name)
    return output
