from model_api import ask_llm
from config.model_config import ModelConfig
from config.global_config import GlobalConfig
from deepeval.models.base_model import DeepEvalBaseLLM


class evaluation_model(DeepEvalBaseLLM) :
    def __init__(self,
                 model_name:str=""):
        self.model_name = self.name_of_the_model(model_name)

    def name_of_the_model(self, 
                          model_name:str):
        
        if model_name == "":
            return ""
        
        elif model_name == "ollama":
            return ModelConfig.OLLAMA_BASE_MODEL_NAME
        
        elif model_name == "llamacpp":
            return ModelConfig.LLAMA_CPP_BASE_MODEL_NAME
        
        elif model_name == "gemini":
            return ModelConfig.GEMINI_MODEL_NAME
        
        else:
            raise ValueError("Invalid model framework")


    def load_model(self):
        pass
        # return self.model

    def generate(self, prompt: str) -> str:
        model_response = ask_llm(GlobalConfig.MAIN_ASK_FRAMEWORK, prompt)
        return model_response.response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name