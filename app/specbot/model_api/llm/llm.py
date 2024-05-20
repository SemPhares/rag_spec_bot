# from langchain_community.llms.llamacpp import LlamaCpp

from llama_cpp import Llama
from llm_typing import LlmOutput
import numpy as np

config = {
    "n_gpu_layers" : 1,
    "n_batch" : 512,
    "repo_id":"TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
    "repo_id":"vikhyatk/moondream2"

}

llm = Llama(n_gpu_layers = config["n_gpu_layers"],
            verbose=True).from_pretrained(repo_id=config["repo_id"])


def ask_llm(llm_input,
            model_name) -> LlmOutput:
    return llm.ask_question(model_name=model_name, llm_input=llm_input)



# #mutiimodal
# llm = Llama.from_pretrained(
#   repo_id="vikhyatk/moondream2",
#   filename="*text-model*",
#   chat_handler=chat_handler,
#   n_ctx=2048, # n_ctx should be increased to accommodate the image embedding
# )