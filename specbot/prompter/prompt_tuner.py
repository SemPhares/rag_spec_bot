from utils.log import logger
from utils.usefull import timer
from model_api.llm_typing import llm_input
from config.model_config import PromptConfig, ModelConfig
from .prompt import (build_classification_prompt,
                     build_rewrite_prompt)
from .prompt_typing import (prompt_input, prompt_output,
                            classification_prompt_input)

# from model_api.gemini_model import clasify_with_gemini, ask_gemini
from model_api.ollama_model import clasify_with_ollama


def classify_user_query(query: str) -> str:
    """
    """

    classification_prompt = build_classification_prompt(
        classification_prompt_input(query=query,
                                    categories=PromptConfig.PROMPT_CATEGORIES)
                                    )

    # output = clasify_with_gemini(llm_input(llm_name=ModelConfig.GEMINI_MODEL_NAME,
    #                                          input=classification_prompt))
    
    output = clasify_with_ollama(llm_input(llm_name=ModelConfig.MISTRAL_7B_MODEL_NAME,
                                  input=classification_prompt))

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
        # outout = ask_gemini(llm_input(llm_name=ModelConfig.GEMINI_MODEL_NAME,
        #                             input=rewrite_prompt))
        
        # use classify_with_ollama instead of ask_gemini to rewrite the query using CONSERVATIVE_TEMPERATURE
        outout = clasify_with_ollama(llm_input(llm_name=ModelConfig.MISTRAL_7B_MODEL_NAME,
                                      input=rewrite_prompt))
        logger.info(f"Query rewritten to: {outout.response}")
        
        return outout.response
    
    else:
        logger.info("Query rewriting is disabled")
        return query
