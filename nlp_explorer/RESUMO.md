# 📋 Resumo - NLP Explorer

## ✅ Funcionalidades Implementadas

### 🏠 Página Inicial
- Apresentação da aplicação
- Guia rápido de uso
- Visão geral das funcionalidades

### 📊 Upload de Dados
- Upload de arquivos CSV, JSON, PKL
- Carregamento do dataset pré-processado (20 Newsgroups)
- Seleção de colunas de texto e rótulos
- Pré-processamento automático
- Visualização de amostra dos dados

### 🔍 Classificação
- Suporte a TF-IDF e Embeddings (Sentence Transformers e Google Gemini)
- Múltiplos classificadores:
  - MultinomialNB / GaussianNB
  - KNN
  - Decision Tree
  - Logistic Regression
- Validação cruzada
- Visualizações:
  - Tabela de métricas
  - Gráficos de barras comparativos
  - Matrizes de confusão
- Seleção inteligente de classificadores baseado no método de vetorização

### 🎯 Clustering
- Suporte a TF-IDF e Embeddings (Sentence Transformers e Google Gemini)
- Algoritmos:
  - K-Means (configurável k)
  - DBSCAN (configurável eps e min_samples)
- Redução dimensional:
  - PCA (opcional, configurável)
  - UMAP ou t-SNE para visualização 2D
- Métricas de avaliação:
  - Silhouette Score
  - Davies-Bouldin Index
- Visualizações:
  - Gráficos de clusters em 2D
  - Comparação com rótulos verdadeiros (se disponível)
  - Distribuição dos clusters

### 📈 Resultados & Exportação
- Explicação básica dos resultados (preparado para integração com LLM)
- Exportação de métricas em CSV
- Exportação de gráficos em PNG
- Resumo dos melhores resultados

## 🎨 Interface
- Layout amplo e responsivo
- Sidebar com navegação e configurações
- Cores e ícones para melhor UX
- Mensagens de erro e sucesso claras
- Barras de progresso e spinners

## 🔧 Configurações
- Arquivo de configuração do Streamlit (`.streamlit/config.toml`)
- Scripts de execução para Windows e Linux/Mac
- Integração com módulos existentes do projeto (`src/`)

## 📝 Arquivos Criados

```
nlp_explorer/
├── app.py                    # Aplicação principal
├── pages/
│   ├── home.py              # Página inicial
│   ├── data_upload.py       # Upload de dados
│   ├── classification.py    # Classificação
│   ├── clustering.py        # Clustering
│   └── results.py           # Resultados e exportação
├── utils/
│   ├── session_state.py     # Gerenciamento de estado
│   ├── config.py            # Configurações
│   └── data_processing.py   # Processamento de dados
├── .streamlit/
│   └── config.toml          # Configuração do Streamlit
├── README.md                 # Documentação
├── INSTRUCOES_USO.md        # Instruções de uso
├── run_app.bat              # Script Windows
└── run_app.sh               # Script Linux/Mac
```

## 🚀 Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute a aplicação:
```bash
streamlit run nlp_explorer/app.py
```

Ou use os scripts:
- Windows: `nlp_explorer/run_app.bat`
- Linux/Mac: `nlp_explorer/run_app.sh`

## 💡 Melhorias Futuras

- [x] Integração com API de LLM (Gemini) para embeddings
- [ ] Integração completa com API de LLM (OpenAI, Gemini) para explicações detalhadas
- [ ] Suporte a mais formatos de arquivo (Excel, Parquet)
- [ ] Histórico de execuções
- [ ] Comparação lado a lado entre TF-IDF e Embeddings
- [ ] Métricas avançadas de clustering
- [ ] Exportação de modelos treinados

## 📚 Dependências

- streamlit>=1.28.0
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- sentence-transformers (para embeddings locais)
- google-generativeai (para embeddings via API do Google)
- umap-learn ou t-SNE (para visualização)

Todas as dependências estão listadas em `requirements.txt`.

## 🔑 Configuração de API

Para usar embeddings do Google Gemini:
1. Obtenha sua chave de API em: https://makersuite.google.com/app/apikey
2. Configure a chave na sidebar da aplicação ao selecionar "Embeddings (Google Gemini)"
3. A API gratuita tem limites restritivos - considere usar Sentence Transformers para testes rápidos

