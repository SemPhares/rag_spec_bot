from deepeval.models.base_model import DeepEvalBaseLLM
from config.global_config import GlobalConfig
from model_api import ask_llm

class evaluation_model(DeepEvalBaseLLM) :
    def __init__(self,
                 model_name:str=""):
        self.model_name = model_name

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