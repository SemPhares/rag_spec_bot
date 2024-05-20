from langchain_community.chat_models.ollama import ChatOllama


def ask_ollama(query:str) -> str:
    model = ChatOllama(model="phi3",
                       temperature=0.5,
                       top_k=0.3,)
    output = model.invoke(query)
    return output.content