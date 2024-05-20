import logging
import time

# Créer un logger
logger = logging.getLogger("RAG")
logger.setLevel(logging.INFO)

# Créer un gestionnaire de log pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Créer un gestionnaire de log pour un fichier
file_handler = logging.FileHandler('logfile.log')
file_handler.setLevel(logging.INFO)

# Créer un formateur de log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ajouter le formateur aux gestionnaires de log
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Ajouter les gestionnaires de log au logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


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

