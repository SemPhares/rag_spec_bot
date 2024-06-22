first_install:
	pip install -r requirements.txt

# download_models_gguf:
# 	python app/specbot/model_api/models_w/download-models.py

run:
	streamlit run app/specbot/spec_app.py 