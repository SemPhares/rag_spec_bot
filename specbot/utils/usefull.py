import time
import streamlit as st
from .log import logger
from typing import Union, List


# Créer un décorateur pour mesurer le temps d'exécution d'une fonction
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Executed {func.__name__} in {elapsed_time} seconds")
        return result
    return wrapper


def spinner(func):
    def wrapper(*args, **kwargs):
        with st.spinner("Wait for it... Porcessing"):
            result = func(*args, **kwargs)
        return result
    return wrapper