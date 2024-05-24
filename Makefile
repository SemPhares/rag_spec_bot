# download_model:
#     curl -o llama3_model.tar.gz https://example.com/path/to/llama3/model
#     tar -xzf llama3_model.tar.gz
#     rm llama3_model.tar.gz

install:
	pip install -r requirements.txt

dev:
	streamlit run app/specbot/spec_app.py