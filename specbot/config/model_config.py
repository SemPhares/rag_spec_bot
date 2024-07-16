from utils.log import logger

try:
    from .secret import GOOGLE_API_KEY, JINA_API_KEY
except ImportError:
    GOOGLE_API_KEY = ""
    JINA_API_KEY = ""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = "specbot/config/.env", verbose=True, override=True)
  

class ApiConfig():

    GOOGLE_API_KEY :str = GOOGLE_API_KEY
    JINA_API_KEY :str = JINA_API_KEY


class PromptConfig():

    PROMPT_CATEGORIES :list = str(os.getenv("PROMPT_CATEGORIES")).split("#")
    REWRITE_QUERY :bool = bool(int(str(os.getenv("REWRITE_QUERY"))))


class ModelConfig():

    MISTRAL_7B_MODEL_NAME :str = os.getenv("MISTRAL_7B_MODEL_NAME") or ""
    MISTRAL_7B_PATH :str = os.getenv("MISTRAL_7B_PATH") or ""
    MISTRAL_7B_REPO_ID :str = os.getenv("MISTRAL_7B_REPO_ID") or ""

    IMAGE_MODEL_NAME :str = os.getenv("IMAGE_MODEL_NAME") or ""
    IMAGE_MODEL_REPO_ID :str = os.getenv("IMAGE_MODEL_REPO_ID") or ""
    IMAGE_MODEL_FILENAME :str = os.getenv("IMAGE_MODEL_FILENAME") or ""

    GEMINI_MODEL_NAME :str = os.getenv("GEMINI_MODEL_NAME") or ""

    OLLAMA_BASE_MODEL_NAME :str = os.getenv("OLLAMA_BASE_MODEL_NAME") or ""
    LLAMA_CPP_BASE_MODEL_NAME :str = os.getenv("LLAMA_CPP_BASE_MODEL_NAME") or ""
    LLAMA_CPP_BASE_MODEL_REPO_ID :str = os.getenv("LLAMA_CPP_BASE_MODEL_REPO_ID") or ""
    LLAMA_CPP_BASE_MODEL_FILENAME :str = os.getenv("LLAMA_CPP_BASE_MODEL_FILENAME") or ""
    LLAMA_CPP_BASE_MODEL_PATH :str = os.getenv("LLAMA_CPP_BASE_MODEL_PATH") or ""

    EMBEDDING_MODEL_NAME :str = os.getenv("EMBEDDING_MODEL_NAME") or ""
    EMBEDDING_MODEL_REPO_ID :str = os.getenv("EMBEDDING_MODEL_REPO_ID") or ""
    EMBEDDING_MODEL_FILENAME :str = os.getenv("EMBEDDING_MODEL_FILENAME") or ""

    logger.info(f"""Configuration : 
                EMBEDDING_MODEL_NAME : {EMBEDDING_MODEL_NAME},
                IMAGE_MODEL_NAME : {IMAGE_MODEL_NAME}
                """)
    

# class MistralModelConfig():

#     MISTRAL_7B_MODEL_NAME :str = os.getenv("MISTRAL_7B_MODEL_NAME") or ""
#     MISTRAL_7B_PATH :str = os.getenv("MISTRAL_7B_PATH") or ""
#     MISTRAL_7B_REPO_ID :str = os.getenv("MISTRAL_7B_REPO_ID") or ""


# class ImageModelConfig():

#     IMAGE_MODEL_NAME :str = os.getenv("IMAGE_MODEL_NAME") or ""
#     IMAGE_MODEL_REPO_ID :str = os.getenv("IMAGE_MODEL_REPO_ID") or ""
#     IMAGE_MODEL_FILENAME :str = os.getenv("IMAGE_MODEL_FILENAME") or ""


# class GeminiModelConfig():
#     GEMINI_MODEL_NAME :str = os.getenv("GEMINI_MODEL_NAME") or ""


# class LlamaModelConfig():

#     LLAMA3_MODEL_NAME :str = os.getenv("LLAMA3_MODEL_NAME") or ""
#     LLAMA3_INSTRUCT_REPO_ID :str = os.getenv("LLAMA3_INSTRUCT_REPO_ID") or ""
#     LLAMA3_INSTRUCT_FILENAME :str = os.getenv("LLAMA3_INSTRUCT_FILENAME") or ""
#     LLAMA3_MODEL_PATH :str = os.getenv("LLAMA3_MODEL_PATH") or ""


# class EmbeddingModelConfig():

#     EMBEDDING_MODEL_NAME :str = os.getenv("EMBEDDING_MODEL_NAME") or ""
#     EMBEDDING_MODEL_REPO_ID :str = os.getenv("EMBEDDING_MODEL_REPO_ID") or ""
#     EMBEDDING_MODEL_FILENAME :str = os.getenv("EMBEDDING_MODEL_FILENAME") or ""

