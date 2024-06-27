import base64
from langchain_core.messages import HumanMessage


def encode_image(image_path):
    ''' Getting the base64 string '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def prompt_func(data):
    text = data["text"]
    image_bs4 = encode_image(data["image_path"])

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image_bs4}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]