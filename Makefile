install_ollama:
	curl -fsSL https://ollama.com/install.sh | sh
	/usr/local/bin/ollama serve

download_base_models:
	ollama pull llama3
	ollama pull llava:7b
	ollama pull mistral:7b

install_env:
	pip install -qr requirements.txt
	pip install -q "unstructured[docx,pdf,xlsx]"

run:
	streamlit run specbot/spec_app.py 