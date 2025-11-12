# 🧭 Text Classification & Clustering: Classical vs Modern NLP

A complete comparison between classical TF-IDF vectorization and modern embeddings for text classification and clustering tasks using the 20 Newsgroups dataset.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This project implements and compares two approaches in Natural Language Processing (NLP):

1. **Classical Methods**: TF-IDF with traditional ML algorithms
2. **Modern Methods**: Neural embeddings (Google Gemini API) with the same algorithms

The goal is to show performance differences between classical and modern NLP techniques in classification and clustering.

---

## ✨ Key Features

* **Dual Vectorization Pipeline**: TF-IDF vs neural embeddings
* **Classifiers**: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting
* **Clustering**: K-Means, Hierarchical, DBSCAN with full metrics
* **Visualizations**: Confusion matrices, UMAP/t-SNE projections, performance comparisons
* **Reproducibility**: Fixed seeds, versioned dependencies
* **Modular Code**: Reusable Python modules

---

## 📁 Project Structure

```
ClassicVsModernNLP/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_vectorization_tfidf.ipynb
│   ├── 03_classification_tfidf.ipynb
│   ├── 04_clustering_tfidf.ipynb
│   ├── 05_embeddings_gemini.ipynb
│   ├── 06_classification_llm_embeddings.ipynb
│   ├── 07_classification_embeddings.ipynb
│   └── 08_clustering_embeddings.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── classification.py
│   └── clustering.py
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔧 Installation

### Prerequisites

* Python 3.8+
* Google Gemini API Key

### Setup

```bash
git clone https://github.com/takaokensei/classic-vs-modern-NLP.git
cd classic-vs-modern-NLP
```

**Create and activate virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Configure API key**

```env
GEMINI_API_KEY=your_real_api_key
```

---

## 🚀 Usage

Run notebooks sequentially:

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### As a library

```python
from src.preprocessing import load_and_preprocess
from src.vectorization import create_tfidf_vectors
from src.classification import train_classifiers

X_train, X_test, y_train, y_test = load_and_preprocess()
X_train_vec, X_test_vec = create_tfidf_vectors(X_train, X_test)
results = train_classifiers(X_train_vec, y_train, X_test_vec, y_test)
```

---

## 📊 Dataset

**Source**: [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)

**Selected Classes**:

* `rec.sport.baseball`
* `rec.sport.hockey`
* `talk.politics.mideast`
* `talk.politics.guns`
* `rec.autos`
* `sci.space`

---

## 📈 Evaluation Metrics

**Classification**: Accuracy 85–92%, Macro F1 0.84–0.91, 5-fold CV, confusion matrices
**Clustering**: Silhouette Score 0.48–0.62, Davies-Bouldin 0.35–0.55, UMAP/t-SNE 2D projections

---

## 🔬 Results

* Modern embeddings improved F1 ~5% vs TF-IDF
* TF-IDF performs well for sports, worse for politics and space
* Clustering with embeddings produces more coherent visual groups
* TF-IDF ~2x faster, embeddings provide richer representations

---

## 🔮 Future Work

* Automatic cluster interpretation with LLM
* Interactive Streamlit dashboard
* Expand to full 20 Newsgroups dataset
* Integrate Sentence-BERT and OpenAI embeddings
* Hyperparameter optimization pipeline

---

## 🤝 Contributing

Pull Requests welcome. For major changes, open an issue first.

---

## 📄 License

MIT License – see [LICENSE](LICENSE)

---

## 📚 Citation

```bibtex
@software{classicvsmodernnlp2025,
  author = {Cauã Vitor},
  title = {Text Classification & Clustering: Classical vs Modern NLP},
  year = {2025},
  url = {https://github.com/takaokensei/classic-vs-modern-NLP}
}
```

---

## 📧 Contact

**Cauã Vitor**

* GitHub: [@takaokensei](https://github.com/takaokensei)
* Email: [cauavitorfigueredo@gmail.com](mailto:cauavitorfigueredo@gmail.com)
* LinkedIn: [Cauã Vitor](https://www.linkedin.com/in/cau%C3%A3-vitor-7bb072286/)
