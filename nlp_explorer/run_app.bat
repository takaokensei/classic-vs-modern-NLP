@echo off
cd /d %~dp0
echo Iniciando NLP Explorer...
echo Verificando ambiente virtual...
if not exist "..\.venv\Scripts\python.exe" (
    echo ERRO: Ambiente virtual nao encontrado em ..\.venv
    echo Certifique-se de que o .venv existe na raiz do projeto.
    pause
    exit /b 1
)
echo Ambiente virtual encontrado!
echo Executando Streamlit com Python do .venv...
..\.venv\Scripts\python.exe -m streamlit run app.py
pause

