import requests
from config import ModelConfig


def generate_text(prompt, 
                  max_tokens=100):
    headers = {
        'Authorization': f'Bearer {ModelConfig.GEMINI_API_KEY}',
        'Content-Type': 'application/json'}
    
    data = {
        'prompt': prompt,
        'max_tokens': max_tokens }
    
    response = requests.post(ModelConfig.GEMINI_API_ENDPOINT, 
                             headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['generated_text']
    else:
        return f"Error: {response.status_code}, {response.text}"


def ask_gemini(query:str) -> str:
    return generate_text(query)

