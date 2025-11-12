#!/bin/bash
cd "$(dirname "$0")"
echo "Iniciando NLP Explorer..."
echo "Ativando ambiente virtual..."
source ../.venv/bin/activate
echo "Executando Streamlit..."
streamlit run app.py
