install_ollama:
	curl -fsSL https://ollama.com/install.sh | sh

serve_ollama:
	/usr/local/bin/ollama serve

install_env:
	pip install -qr requirements.txt
	pip install -q "unstructured[all-docs]"

run:
	streamlit run specbot/spec_app.py 