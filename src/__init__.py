"""
Módulos utilitários para classificação e clustering de textos.

Este pacote contém funções reutilizáveis para:
- Pré-processamento de textos
- Vetorização (TF-IDF)
- Classificação
- Clustering
"""

from .preprocessing import preprocess_text, preprocess_batch
from .vectorization import create_tfidf_vectorizer, vectorize_texts, get_top_features_by_class
from .classification import get_default_classifiers, train_and_evaluate, generate_classification_report
from .clustering import (
    reduce_dimensions_pca_umap,
    kmeans_clustering,
    dbscan_clustering,
    evaluate_clustering
)

__all__ = [
    # Preprocessing
    'preprocess_text',
    'preprocess_batch',
    # Vectorization
    'create_tfidf_vectorizer',
    'vectorize_texts',
    'get_top_features_by_class',
    # Classification
    'get_default_classifiers',
    'train_and_evaluate',
    'generate_classification_report',
    # Clustering
    'reduce_dimensions_pca_umap',
    'kmeans_clustering',
    'dbscan_clustering',
    'evaluate_clustering',
]

