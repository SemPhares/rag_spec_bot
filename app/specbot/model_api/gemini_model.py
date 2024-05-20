import requests

# Replace with your actual API key and endpoint
API_KEY = 'AIzaSyC0APq90xsuS34jTPwArK-Rjp_gWoHSxp4'
API_ENDPOINT = 'https://api.gemini.com/v1/text'

def generate_text(prompt, max_tokens=100):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'prompt': prompt,
        'max_tokens': max_tokens
    }
    
    response = requests.post(API_ENDPOINT, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['generated_text']
    else:
        return f"Error: {response.status_code}, {response.text}"


def ask_gemini(query:str) -> str:
    return generate_text(query)
