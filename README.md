# 🧭 Projeto — Classificação e Clustering de Textos (20 Newsgroups - 6 classes)

**Disciplina:** ELE606 — Tópicos em IA  
**Professor:** José Alfredo F. Costa  
**Aluno:** Cauã Vitor  
**Instituição:** UFRN — DEE — 2025.2

---

## 📋 Descrição

Este projeto realiza classificação e clustering de textos da base **20 Newsgroups** utilizando duas abordagens:

1. **Vetorização clássica (TF-IDF)** com classificadores e algoritmos de clustering tradicionais
2. **Embeddings modernos (Google Gemini)** com os mesmos algoritmos para comparação

O objetivo é comparar o desempenho entre métodos clássicos e modernos de NLP em tarefas de classificação e clustering.

---

## 📁 Estrutura de Diretórios

```
ClassicVsModernNLP/
│
├── data/
│   ├── raw/              # Dados brutos (se necessário)
│   └── processed/        # Dados pré-processados e vetorizados
│
├── notebooks/            # Notebooks Jupyter em ordem sequencial
│   ├── 01_preprocessing.ipynb
│   ├── 02_vectorization_tfidf.ipynb
│   ├── 03_classification_tfidf.ipynb
│   ├── 04_clustering_tfidf.ipynb
│   ├── 05_embeddings_gemini.ipynb
│   ├── 06_classification_llm_embeddings.ipynb
│   ├── 07_classification_embeddings.ipynb
│   └── 08_clustering_embeddings.ipynb
│
├── src/                  # Módulos Python reutilizáveis
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── classification.py
│   └── clustering.py
│
├── reports/
│   ├── figures/          # Figuras (matrizes de confusão, UMAP, etc.)
│   └── metrics/          # Métricas salvas em CSV
│
├── requirements.txt
└── README.md
```

---

## 🔧 Instalação e Configuração

### 1. Criar e ativar ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

**Nota sobre Python 3.14:** O `requirements.txt` agora tem suporte automático para diferentes versões do Python:
- **Python 3.8-3.13**: `umap-learn` será instalado automaticamente
- **Python 3.14+**: `umap-learn` não será instalado (mas o código usa t-SNE como fallback)

**Instalação:**
```bash
pip install -r requirements.txt
```

**Se quiser UMAP no Python 3.14+** (opcional):
```bash
# 1. Instale primeiro o numba beta
pip install numba==0.63.0b1

# 2. Depois instale o umap-learn
pip install umap-learn
```

O código está preparado para usar **t-SNE automaticamente** como fallback quando UMAP não está disponível. Os notebooks funcionam perfeitamente com t-SNE!

### 3. Configurar chave da API do Google Gemini

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "SUA_CHAVE_AQUI"
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

**Alternativa:** Criar arquivo `.env` na raiz do projeto:
```
GEMINI_API_KEY=SUA_CHAVE_AQUI
```

> **Nota:** Você pode obter uma chave de API gratuita em [Google AI Studio](https://makersuite.google.com/app/apikey).

---

## 🚀 Execução

Execute os notebooks na ordem sequencial:

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### Descrição dos Notebooks

1. **01_preprocessing.ipynb**: Carrega e pré-processa os dados das 6 classes selecionadas
2. **02_vectorization_tfidf.ipynb**: Gera vetorização TF-IDF
3. **03_classification_tfidf.ipynb**: Classifica textos usando TF-IDF
4. **04_clustering_tfidf.ipynb**: Realiza clustering usando TF-IDF
5. **05_embeddings_gemini.ipynb**: Gera embeddings usando Google Gemini API ou Sentence-Transformers
6. **06_classification_llm_embeddings.ipynb**: Classifica textos usando embeddings via API com input dinâmico
7. **07_classification_embeddings.ipynb**: Classifica textos usando embeddings pré-gerados
8. **08_clustering_embeddings.ipynb**: Realiza clustering usando embeddings

---

## 📊 Classes Utilizadas

- `rec.sport.baseball`
- `rec.sport.hockey`
- `talk.politics.mideast`
- `talk.politics.guns`
- `rec.autos`
- `sci.space`

---

## 📈 Métricas e Resultados

Os resultados são salvos automaticamente em:

- **Métricas**: `/reports/metrics/*.csv`
- **Figuras**: `/reports/figures/*.png`

### Métricas de Classificação:
- Accuracy
- Macro F1-Score
- Validação cruzada (k=5)
- Matrizes de confusão

### Métricas de Clustering:
- Silhouette Score
- Davies-Bouldin Index
- Visualizações UMAP 2D

---

## 🧰 Bibliotecas Principais

- **scikit-learn**: Classificação, clustering e pré-processamento
- **google-generativeai**: Geração de embeddings via API
- **umap-learn**: Redução dimensional para visualização
- **pandas/numpy**: Manipulação de dados
- **matplotlib/seaborn**: Visualização

---

## 📝 Notas Importantes

1. **Reprodutibilidade**: Todos os processos usam `random_state=42` para garantir resultados reproduzíveis
2. **Rate Limiting**: O notebook `05_embeddings_gemini.ipynb` inclui delays entre requisições para evitar rate limiting
3. **Armazenamento**: Dados intermediários são salvos em pickle para facilitar reprocessamento
4. **Comparação**: Os resultados permitem comparar diretamente TF-IDF vs Embeddings

---

## 🔮 Extensões Futuras

- Explicação automática dos clusters via LLM (Groq ou Gemini)
- Sumarização por tópico com prompts curtos
- Protótipo Streamlit para interação com parâmetros e visualizações dinâmicas

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte da disciplina ELE606 — Tópicos em IA.

---

## 👤 Autor

**Cauã Vitor**  
UFRN — Departamento de Engenharia Elétrica  
2025.2

