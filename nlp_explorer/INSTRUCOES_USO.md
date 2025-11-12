# 📖 Instruções de Uso - NLP Explorer

## 🚀 Como Executar

### Opção 1: Via linha de comando
```bash
# Na raiz do projeto
streamlit run nlp_explorer/app.py
```

### Opção 2: Scripts de execução
- **Windows**: Execute `run_app.bat` dentro da pasta `nlp_explorer`
- **Linux/Mac**: Execute `run_app.sh` dentro da pasta `nlp_explorer`

## 📋 Fluxo de Uso

### 1. Upload de Dados
1. Acesse a página **📊 Upload de Dados**
2. Escolha entre:
   - **Upload de arquivo**: Faça upload de CSV, JSON ou PKL
   - **Dataset pré-processado**: Use o dataset 20 Newsgroups já processado

### 2. Classificação
1. Acesse a página **🔍 Classificação**
2. Configure:
   - Método de vetorização (TF-IDF, Embeddings (Sentence Transformers) ou Embeddings (Google Gemini))
   - Parâmetros do método escolhido
   - Se usar Google Gemini: insira sua chave de API na sidebar
   - Classificadores a serem testados
3. Clique em **🚀 Executar Classificação**
4. Explore as métricas e visualizações

### 3. Clustering
1. Acesse a página **🎯 Clustering**
2. Configure:
   - Método de vetorização (TF-IDF, Embeddings (Sentence Transformers) ou Embeddings (Google Gemini))
   - Se usar Google Gemini: insira sua chave de API na sidebar
   - Algoritmos de clustering (K-Means, DBSCAN)
   - Parâmetros de redução dimensional
3. Clique em **🚀 Executar Clustering**
4. Visualize os clusters e métricas

### 4. Exportação
1. Acesse a página **📈 Resultados & Exportação**
2. Exporte:
   - Métricas em CSV
   - Gráficos em PNG
   - Explicações dos resultados

## 💡 Dicas

- **TF-IDF**: Método clássico, rápido e eficiente para muitos casos
- **Embeddings (Sentence Transformers)**: Método moderno local, rápido e eficiente
- **Embeddings (Google Gemini)**: Requer API key, pode ser lento devido aos limites da API gratuita
- **Visualizações**: Use UMAP para visualizações mais rápidas, t-SNE para mais precisão (mas mais lento)
- **Exportação**: Todos os resultados podem ser exportados para análise posterior

## 🔑 Usando Google Gemini API

Para usar embeddings via API do Google:

1. **Obtenha a chave**: Acesse https://makersuite.google.com/app/apikey e crie uma chave gratuita
2. **Configure na aplicação**: Insira a chave na sidebar ao selecionar "Embeddings (Google Gemini)"
3. **Limitações**: 
   - A API gratuita tem quota muito restritiva para embeddings
   - Pode exigir aguardar 24h para resetar quota
   - Para testes rápidos, use Sentence Transformers
4. **Performance**: Processamento sequencial com delay de 1 segundo entre requisições para respeitar limites

## ⚠️ Requisitos

Certifique-se de ter instalado:
- `streamlit`
- `sentence-transformers` (para embeddings locais)
- `google-generativeai` (para embeddings via API do Google, opcional)
- `umap-learn` ou `scikit-learn` (para visualizações)

Instale com: `pip install -r requirements.txt`

