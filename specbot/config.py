from utils.log import logger

try:
    from secret import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = ""

from prompter.prompt_template import simple_prompt

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = "specbot/.env", verbose=True, override=True)


class GlobalConfig():

    CHUNCK_SIZE :int = int(os.getenv("CHUNCK_SIZE")) # type: ignore
    CHUNK_OVERLAP :int = int(os.getenv("CHUNK_OVERLAP")) # type: ignore
    NUM_GPU :int = int(os.getenv("NUM_GPU")) # type: ignore

class ModelConfig():

    PHI3_MODEL_NAME :str = os.getenv("SUMMARIZE_MODEL") or ""
    PHI3_INSTRUCT_REPO_ID :str = os.getenv("PHI3_INSTRUCT_REPO_ID") or ""
    PHI3_INSTRUCT_FILENAME :str = os.getenv("PHI3_INSTRUCT_FILENAME") or ""

    MISTRAL_7B_MODEL_NAME :str = os.getenv("MISTRAL_7B_MODEL_NAME") or ""
    MISTRAL_7B_PATH :str = os.getenv("MISTRAL_7B_PATH") or ""

    IMAGE_MODEL_NAME :str = os.getenv("IMAGE_MODEL_NAME") or ""
    IMAGE_MODEL_REPO_ID :str = os.getenv("IMAGE_MODEL_REPO_ID") or ""
    IMAGE_MODEL_FILENAME :str = os.getenv("IMAGE_MODEL_FILENAME") or ""

    GEMINI_API_KEY :str = GEMINI_API_KEY
    GEMINI_API_ENDPOINT :str = os.getenv("GEMINI_API_ENDPOINT") or ""

    OLLAMA_BASE_MODEL_NAME :str = os.getenv("OLLAMA_BASE_MODEL_NAME") or ""
    LLAMA_CPP_BASE_MODEL_NAME :str = os.getenv("LLAMA_CPP_BASE_MODEL_NAME") or ""
    LLAMA_CPP_BASE_MODEL_REPO_ID :str = os.getenv("LLAMA_CPP_BASE_MODEL_REPO_ID") or ""
    LLAMA_CPP_BASE_MODEL_FILENAME :str = os.getenv("LLAMA_CPP_BASE_MODEL_FILENAME") or ""
    LLAMA_CPP_BASE_MODEL_PATH :str = os.getenv("LLAMA_CPP_BASE_MODEL_PATH") or ""
    
    logger.info(f"""BASE MODELS: {OLLAMA_BASE_MODEL_NAME}, {LLAMA_CPP_BASE_MODEL_REPO_ID}""")

    EMBEDDING_MODEL_NAME :str = os.getenv("EMBEDDING_MODEL_NAME") or ""
    EMBEDDING_MODEL_REPO_ID :str = os.getenv("EMBEDDING_MODEL_REPO_ID") or ""
    EMBEDDING_MODEL_FILENAME :str = os.getenv("EMBEDDING_MODEL_FILENAME") or ""

    EXTRACTED_IMAGE_PROMPT :str = simple_prompt.EXTRACTED_IMAGE_PROMPT