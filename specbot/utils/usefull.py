import os
import time
import shutil
import streamlit as st
from .log import logger


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


def stream_data(text_to_stream:str):
    for word in text_to_stream.split(" "):
        yield word + " "
        time.sleep(0.02)




def supprimer_contenu_dossier(chemin_dossier):
    for nom in os.listdir(chemin_dossier):
        chemin_complet = os.path.join(chemin_dossier, nom)
        if os.path.isdir(chemin_complet):
            shutil.rmtree(chemin_complet)
        else:
            os.remove(chemin_complet)