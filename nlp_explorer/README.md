# NLP Explorer

Aplicação Streamlit interativa para explorar e comparar métodos clássicos (TF-IDF) e modernos (Embeddings) de NLP para classificação e clustering de textos.

## 🚀 Funcionalidades

- 📊 **Upload de Dados**: Suporte para CSV, JSON e dataset pré-processado (20 Newsgroups)
- 🔍 **Classificação**: Múltiplos algoritmos (Naive Bayes, KNN, Decision Tree, Logistic Regression)
- 🎯 **Clustering**: K-Means e DBSCAN com visualizações interativas
- 📈 **Visualizações**: Gráficos interativos de métricas e clusters
- 💡 **Explicabilidade**: Explicações automáticas dos resultados
- 💾 **Exportação**: Exportação de resultados em CSV e gráficos em PNG

## 📦 Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
pip install streamlit
```

2. Execute a aplicação:
```bash
streamlit run nlp_explorer/app.py
```

## 🎯 Uso

1. **Upload de Dados**: Faça upload do seu dataset ou use o dataset pré-processado
2. **Escolha o Método**: Selecione TF-IDF, Embeddings (Sentence Transformers) ou Embeddings (Google Gemini)
3. **Configure API (se usar Gemini)**: Insira sua chave de API do Google na sidebar
4. **Execute Análise**: Escolha classificação ou clustering
5. **Explore Resultados**: Visualize métricas e gráficos
6. **Exporte**: Baixe resultados em CSV ou PNG

## 📁 Estrutura

```
nlp_explorer/
├── app.py                 # Aplicação principal
├── pages/                 # Páginas da aplicação
│   ├── home.py           # Página inicial
│   ├── data_upload.py    # Upload de dados
│   ├── classification.py # Classificação
│   ├── clustering.py     # Clustering
│   └── results.py        # Resultados e exportação
└── utils/                 # Utilitários
    ├── session_state.py  # Gerenciamento de estado
    ├── config.py         # Configurações
    └── data_processing.py # Processamento de dados
```

## 🔧 Requisitos

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- sentence-transformers (para embeddings)
- umap-learn ou t-SNE (para visualização)

## 📝 Notas

- Para usar embeddings locais, instale `sentence-transformers`
- Para usar embeddings via API, instale `google-generativeai` e configure sua chave de API
- Para visualizações melhores, instale `umap-learn` (requer Python < 3.14)
- O dataset pré-processado deve ser gerado executando o notebook `01_preprocessing.ipynb` primeiro

## 🔑 API do Google Gemini

Para usar embeddings do Google Gemini:

1. Obtenha sua chave de API gratuita em: https://makersuite.google.com/app/apikey
2. Configure a chave na sidebar ao selecionar "Embeddings (Google Gemini)"
3. **Importante**: A API gratuita tem limites restritivos (quota muito baixa ou zero para embeddings)
4. Para testes rápidos, recomendamos usar Sentence Transformers que não requer API

