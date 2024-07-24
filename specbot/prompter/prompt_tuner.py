from utils.log import logger
from model_api import ask_llm
from utils.usefull import timer
from config.model_config import PromptConfig
from config.global_config import GlobalConfig
from .prompt import (build_classification_prompt,
                     build_rewrite_prompt)
from .prompt_typing import classification_prompt_input


def classify_user_query(query: str) -> str:
    """
    """

    classification_prompt = build_classification_prompt(
        classification_prompt_input(query=query,
                                    categories=PromptConfig.PROMPT_CATEGORIES))
    
    output = ask_llm(GlobalConfig.MAIN_CLASSIFICATION_MODEL, 
                     classification_prompt,
                     conservative_mode = True)

    return output.response


@timer
def extract_category(query: str) -> str:
    """
    """

    output = classify_user_query(query)

    # Extract the category from the response
    found_category = [category for category in PromptConfig.PROMPT_CATEGORIES if category.upper() in output.upper()]

    if not found_category:
        logger.info(f"Category not found in response: {output}")
        return "Unknown"
    
    elif len(found_category) > 1:
        logger.warning(f"Multiple categories found in response: {found_category}")
        return "Unknown"
    
    else:
        return found_category[0]


def rewrite_query(query: str) -> str:
    """
    """
    if PromptConfig.REWRITE_QUERY:
        rewrite_prompt = build_rewrite_prompt(query=query)
        
        # use classify_with_ollama instead of ask_gemini to rewrite the query using CONSERVATIVE_TEMPERATURE
        output = ask_llm(GlobalConfig.MAIN_REWRITING_MODEL,
                         rewrite_prompt,
                         conservative_mode = True)
        
        logger.info(f"Query rewritten to: {output.response}")
        
        return output.response
    
    else:
        logger.info("Query rewriting is disabled")
        return query
