from utils.log import logger

try:
    from secret import GEMINI_API_KEY, JINA_API_KEY
except ImportError:
    GEMINI_API_KEY = ""
    JINA_API_KEY = ""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path = "specbot/.env", verbose=True, override=True)


class GlobalConfig():

    CHUNCK_SIZE :int = int(os.getenv("CHUNCK_SIZE")) # type: ignore
    CHUNK_OVERLAP :int = int(os.getenv("CHUNK_OVERLAP")) # type: ignore
    NUM_GPU :int = int(os.getenv("NUM_GPU")) # type: ignore

    IMAGES_EXTENSIONS = str(os.getenv("IMAGES_EXTENSIONS")).split("#") # type: ignore
    ACCEPTED_EXTENSION = str(os.getenv("ACCEPTED_EXTENSION")).split("#")
    
    EXTRACT_IMG :bool = bool(os.getenv("EXTRACT_IMG")) 
    EXTRACTED_IMG_DIR = os.getenv("EXTRACTED_IMG_DIR") or ""


class ModelConfig():

    MISTRAL_7B_MODEL_NAME :str = os.getenv("MISTRAL_7B_MODEL_NAME") or ""
    MISTRAL_7B_PATH :str = os.getenv("MISTRAL_7B_PATH") or ""
    MISTRAL_7B_REPO_ID :str = os.getenv("MISTRAL_7B_REPO_ID") or ""

    IMAGE_MODEL_NAME :str = os.getenv("IMAGE_MODEL_NAME") or ""
    IMAGE_MODEL_REPO_ID :str = os.getenv("IMAGE_MODEL_REPO_ID") or ""
    IMAGE_MODEL_FILENAME :str = os.getenv("IMAGE_MODEL_FILENAME") or ""

    GEMINI_API_KEY :str = GEMINI_API_KEY
    GEMINI_API_ENDPOINT :str = os.getenv("GEMINI_API_ENDPOINT") or ""

    JINA_API_KEY :str = JINA_API_KEY

    OLLAMA_BASE_MODEL_NAME :str = os.getenv("OLLAMA_BASE_MODEL_NAME") or ""
    LLAMA_CPP_BASE_MODEL_NAME :str = os.getenv("LLAMA_CPP_BASE_MODEL_NAME") or ""
    LLAMA_CPP_BASE_MODEL_REPO_ID :str = os.getenv("LLAMA_CPP_BASE_MODEL_REPO_ID") or ""
    LLAMA_CPP_BASE_MODEL_FILENAME :str = os.getenv("LLAMA_CPP_BASE_MODEL_FILENAME") or ""
    LLAMA_CPP_BASE_MODEL_PATH :str = os.getenv("LLAMA_CPP_BASE_MODEL_PATH") or ""

    EMBEDDING_MODEL_NAME :str = os.getenv("EMBEDDING_MODEL_NAME") or ""
    EMBEDDING_MODEL_REPO_ID :str = os.getenv("EMBEDDING_MODEL_REPO_ID") or ""
    EMBEDDING_MODEL_FILENAME :str = os.getenv("EMBEDDING_MODEL_FILENAME") or ""
