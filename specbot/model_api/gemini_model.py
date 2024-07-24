from time import time
from utils.usefull import timer
from google import generativeai as genai
from config.model_config import ApiConfig
from config.global_config import GlobalConfig
from .llm_typing import llm_input, llm_output


# Gemini Api Instance
genai.configure(api_key=ApiConfig.GOOGLE_API_KEY)


def ask_gemini(query:llm_input, 
               conservative_mode:bool = False) -> llm_output:
    if conservative_mode:
        model = genai.GenerativeModel(model_name = query.llm_name,
                                      generation_config = {"temperature" : GlobalConfig.CONSERVATIVE_TEMPERATURE})
    else:
        model = genai.GenerativeModel(model_name = query.llm_name,
                                      generation_config = {"temperature" : GlobalConfig.TEMPERATURE})
    start = time()   
    response = model.generate_content(query.input)
    end = time()

    output = llm_output(response = str(response.text), 
                        llm_name = query.llm_name,
                        generation_time = end - start)
    return output