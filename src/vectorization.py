"""
Módulo de vetorização de textos.

Funções para gerar representações vetoriais de textos usando TF-IDF.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.95):
    """
    Cria um TfidfVectorizer configurado.
    
    Args:
        max_features: Número máximo de features
        ngram_range: Tupla com range de n-gramas (ex: (1,2) para unigramas e bigramas)
        min_df: Frequência mínima de documentos
        max_df: Frequência máxima de documentos (proporção)
    
    Returns:
        TfidfVectorizer configurado
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=True
    )


def vectorize_texts(texts, vectorizer=None, max_features=5000, ngram_range=(1, 2), 
                    min_df=5, max_df=0.95):
    """
    Vetoriza textos usando TF-IDF.
    
    Args:
        texts: Lista de textos a serem vetorizados
        vectorizer: TfidfVectorizer pré-configurado (opcional)
        max_features: Número máximo de features (se vectorizer não fornecido)
        ngram_range: Range de n-gramas (se vectorizer não fornecido)
        min_df: Frequência mínima (se vectorizer não fornecido)
        max_df: Frequência máxima (se vectorizer não fornecido)
    
    Returns:
        Tupla (matriz_sparse, vectorizer)
    """
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer(max_features, ngram_range, min_df, max_df)
    
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def get_top_features_by_class(X, y, vectorizer, target_names, top_n=20):
    """
    Obtém as top features (palavras) por classe baseado nos scores TF-IDF médios.
    
    Args:
        X: Matriz TF-IDF (sparse ou dense)
        y: Rótulos das classes
        vectorizer: TfidfVectorizer usado
        target_names: Nomes das classes
        top_n: Número de top features por classe
    
    Returns:
        Dicionário {class_name: [(feature, score), ...]}
    """
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    feature_names = vectorizer.get_feature_names_out()
    top_features = {}
    
    for class_idx, class_name in enumerate(target_names):
        # Índices dos documentos desta classe
        class_mask = y == class_idx
        # Média dos scores TF-IDF para esta classe
        class_tfidf = X_dense[class_mask].mean(axis=0)
        
        # Obter top N palavras
        top_indices = np.argsort(class_tfidf)[::-1][:top_n]
        top_words = [(feature_names[i], class_tfidf[i]) for i in top_indices]
        top_features[class_name] = top_words
    
    return top_features

