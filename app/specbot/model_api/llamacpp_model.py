from llama_cpp import Llama as llamacpp

config = {
    "n_gpu_layers" : 1,
    "n_batch" : 512,
    "repo_id":"TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
    "mutiimodal_repo_id":"vikhyatk/moondream2",
    "max_tokens": 32,

}


def ask_llmcpp(query:str) -> str:
    
    llama_cpp = llamacpp(model_path="app/specbot/model_api/models_w/capybarahermes-2.5-mistral-7b.Q2_K.gguf",
                         n_gpu_layers = config["n_gpu_layers"], 
                         temperature=0.75,
                         max_tokens=2000,
                         top_p=1,verbose=True)
    
    # output = llama_cpp(
    #   prompt = query,
    #   max_tokens=config["max_tokens"], 
    #   stop=["Q:", "\n"],
    #   echo=True) 
    
    return llama_cpp.invoke(query)

