
import os
from utils.log import logger
from dotenv import load_dotenv
load_dotenv(dotenv_path = "specbot/config/.env", verbose=True, override=True)


class GlobalConfig():

    CHUNCK_SIZE :int = int(os.getenv("CHUNCK_SIZE")) 
    CHUNK_OVERLAP :int = int(os.getenv("CHUNK_OVERLAP")) 
    NUM_GPU :int = int(os.getenv("NUM_GPU")) 
    TEMPERATURE :float = float(os.getenv("TEMPERATURE")) 
    CONTEXT_WINDOW :int = int(os.getenv("CONTEXT_WINDOW")) 
    CONSERVATIVE_TEMPERATURE :float = float(os.getenv("CONSERVATIVE_TEMPERATURE")) 
    RETRIEVER_TOP_K :int = int(os.getenv("RETRIEVER_TOP_K")) 
    RETRIEVER_SCORE_THRESHOLD :float = float(os.getenv("RETRIEVER_SCORE_THRESHOLD")) 

    IMAGES_EXTENSIONS :list = str(os.getenv("IMAGES_EXTENSIONS")).split("#") 
    ACCEPTED_EXTENSION :list = str(os.getenv("ACCEPTED_EXTENSION")).split("#")
    
    EXTRACT_IMG :bool = bool(int(os.getenv("EXTRACT_IMG"))) 
    EXTRACTED_IMG_DIR = os.getenv("EXTRACTED_IMG_DIR") or ""

    MAIN_ASK_FRAMEWORK :str = os.getenv("MAIN_ASK_FRAMEWORK") or ""
    MAIN_CLASSIFICATION_MODEL :str = os.getenv("MAIN_CLASSIFICATION_MODEL") or ""

    EVALUATE_RAG :bool = bool(int(str(os.getenv("EVALUATE_RAG")))) 

    logger.info(f"""Configuration : 
                 EXTRACT_IMG : {EXTRACT_IMG},
                 EXTRACTED_IMG_DIR : {EXTRACTED_IMG_DIR}""")
    