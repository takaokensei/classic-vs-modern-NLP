"""
Sistema de cache para salvar e carregar resultados processados.
Evita reprocessar dados do dataset 20 Newsgroups com 6 classes.
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import streamlit as st


CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "20news_6classes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(filename: str) -> Path:
    """Retorna o caminho completo para um arquivo de cache."""
    return CACHE_DIR / filename


def save_to_cache(data: Any, filename: str) -> bool:
    """
    Salva dados no cache.
    
    Args:
        data: Dados a serem salvos
        filename: Nome do arquivo (será salvo como .pkl)
    
    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        cache_path = get_cache_path(filename)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Erro ao salvar cache {filename}: {e}")
        return False


def load_from_cache(filename: str) -> Optional[Any]:
    """
    Carrega dados do cache.
    
    Args:
        filename: Nome do arquivo (será carregado como .pkl)
    
    Returns:
        Dados carregados ou None se não existir
    """
    try:
        cache_path = get_cache_path(filename)
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Erro ao carregar cache {filename}: {e}")
        return None


def save_metadata(metadata: Dict[str, Any]) -> bool:
    """Salva metadados do dataset em JSON."""
    try:
        metadata_path = get_cache_path("metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erro ao salvar metadata: {e}")
        return False


def load_metadata() -> Optional[Dict[str, Any]]:
    """Carrega metadados do dataset."""
    try:
        metadata_path = get_cache_path("metadata.json")
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Erro ao carregar metadata: {e}")
        return None


def is_20news_6classes_dataset(texts, labels, target_names) -> bool:
    """
    Verifica se o dataset carregado é o 20 Newsgroups com 6 classes.
    
    Args:
        texts: Lista de textos
        labels: Array de labels
        target_names: Lista de nomes das classes
    
    Returns:
        True se for o dataset esperado
    """
    # Verificar número de classes
    if len(target_names) != 6:
        return False
    
    # Verificar se os nomes das classes correspondem ao 20 Newsgroups (6 classes usadas)
    expected_classes = [
        'rec.autos',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.space',
        'talk.politics.guns',
        'talk.politics.mideast'
    ]
    
    # Verificar se todas as classes esperadas estão presentes
    target_names_set = set(target_names)
    expected_set = set(expected_classes)
    
    return target_names_set == expected_set


def cache_embeddings(vectors: np.ndarray, method: str, model_name: Optional[str] = None) -> bool:
    """
    Salva embeddings no cache.
    
    Args:
        vectors: Matriz de embeddings
        method: Método usado ('tfidf', 'sentence_transformer', 'gemini')
        model_name: Nome do modelo (opcional)
    
    Returns:
        True se salvou com sucesso
    """
    if method == 'tfidf':
        filename = "embeddings_tfidf.pkl"
    elif method == 'sentence_transformer':
        filename = f"embeddings_sentence_transformer_{model_name or 'default'}.pkl"
    elif method == 'gemini':
        filename = "embeddings_gemini.pkl"
    else:
        return False
    
    return save_to_cache(vectors, filename)


def load_cached_embeddings(method: str, model_name: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Carrega embeddings do cache.
    
    Args:
        method: Método usado ('tfidf', 'sentence_transformer', 'gemini')
        model_name: Nome do modelo (opcional)
    
    Returns:
        Matriz de embeddings ou None se não existir
    """
    if method == 'tfidf':
        filename = "embeddings_tfidf.pkl"
    elif method == 'sentence_transformer':
        filename = f"embeddings_sentence_transformer_{model_name or 'default'}.pkl"
    elif method == 'gemini':
        filename = "embeddings_gemini.pkl"
    else:
        return None
    
    return load_from_cache(filename)


def _get_classifiers_key(classifiers: Dict[str, Any]) -> str:
    """
    Gera uma chave única baseada nos nomes dos classificadores.
    
    Args:
        classifiers: Dicionário de classificadores
    
    Returns:
        String com nomes dos classificadores ordenados e concatenados
    """
    sorted_names = sorted(classifiers.keys())
    return "_".join(sorted_names).replace(" ", "_").replace("(", "").replace(")", "")


def cache_classification_results(results: Dict[str, Any], method: str, classifiers: Dict[str, Any]) -> bool:
    """
    Salva resultados de classificação no cache com chave baseada na combinação de vetorização + classificadores.
    
    Args:
        results: Dicionário com resultados de classificação
        method: Método de vetorização usado ('tfidf', 'sentence_transformer', 'gemini')
        classifiers: Dicionário de classificadores usados
    
    Returns:
        True se salvou com sucesso
    """
    classifiers_key = _get_classifiers_key(classifiers)
    filename = f"classification_{method}_{classifiers_key}.pkl"
    return save_to_cache(results, filename)


def load_cached_classification_results(method: str, classifiers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Carrega resultados de classificação do cache baseado na combinação de vetorização + classificadores.
    Tenta primeiro o novo formato (combinado), depois o formato antigo (compatibilidade).
    
    Args:
        method: Método de vetorização usado ('tfidf', 'sentence_transformer', 'gemini')
        classifiers: Dicionário de classificadores usados
    
    Returns:
        Dicionário com resultados ou None se não existir
    """
    # Tentar novo formato primeiro (combinado)
    classifiers_key = _get_classifiers_key(classifiers)
    filename = f"classification_{method}_{classifiers_key}.pkl"
    results = load_from_cache(filename)
    
    if results is not None:
        return results
    
    # Se não encontrou, tentar formato antigo (compatibilidade)
    old_filename = f"classification_results_{method}.pkl"
    old_results = load_from_cache(old_filename)
    
    if old_results is not None:
        # Filtrar resultados do cache antigo para apenas os classificadores selecionados
        cached_results = old_results.get('results', {})
        cached_cv_results = old_results.get('cv_results', {})
        cached_predictions = old_results.get('predictions', {})
        
        # Verificar se todos os classificadores selecionados estão no cache antigo
        selected_names = set(classifiers.keys())
        available_names = set(cached_results.keys())
        
        if selected_names.issubset(available_names):
            # Todos os classificadores selecionados estão disponíveis
            filtered_results = {
                'results': {name: cached_results[name] for name in selected_names},
                'cv_results': {name: cached_cv_results.get(name, {}) for name in selected_names},
                'predictions': {name: cached_predictions.get(name, []) for name in selected_names},
                'X_test': old_results.get('X_test'),
                'y_test': old_results.get('y_test')
            }
            return filtered_results
        else:
            # Alguns classificadores não estão no cache antigo
            # Retornar None para processar normalmente
            return None
    
    return None


def cache_clustering_results(results: Dict[str, Any], method: str) -> bool:
    """
    Salva resultados de clustering no cache.
    
    Args:
        results: Dicionário com resultados de clustering
        method: Método de vetorização usado ('tfidf', 'embeddings')
    
    Returns:
        True se salvou com sucesso
    """
    filename = f"clustering_results_{method}.pkl"
    return save_to_cache(results, filename)


def load_cached_clustering_results(method: str) -> Optional[Dict[str, Any]]:
    """
    Carrega resultados de clustering do cache.
    
    Args:
        method: Método de vetorização usado ('tfidf', 'embeddings')
    
    Returns:
        Dicionário com resultados ou None se não existir
    """
    filename = f"clustering_results_{method}.pkl"
    return load_from_cache(filename)


def cache_reduction_data(X_pca: np.ndarray, X_vis: np.ndarray, 
                        reduction_method: str, pca_components: int) -> bool:
    """
    Salva dados de redução dimensional no cache.
    
    Args:
        X_pca: Dados após PCA
        X_vis: Dados após redução para visualização (UMAP/t-SNE)
        reduction_method: Método usado ('umap', 'tsne')
        pca_components: Número de componentes do PCA
    
    Returns:
        True se salvou com sucesso
    """
    data = {
        'X_pca': X_pca,
        'X_vis': X_vis,
        'reduction_method': reduction_method,
        'pca_components': pca_components
    }
    filename = f"reduction_{reduction_method}_pca{pca_components}.pkl"
    return save_to_cache(data, filename)


def load_cached_reduction_data(reduction_method: str, pca_components: int) -> Optional[Dict[str, Any]]:
    """
    Carrega dados de redução dimensional do cache.
    
    Args:
        reduction_method: Método usado ('umap', 'tsne')
        pca_components: Número de componentes do PCA
    
    Returns:
        Dicionário com dados ou None se não existir
    """
    filename = f"reduction_{reduction_method}_pca{pca_components}.pkl"
    return load_from_cache(filename)


def clear_cache() -> bool:
    """
    Limpa todo o cache.
    
    Returns:
        True se limpou com sucesso
    """
    try:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Erro ao limpar cache: {e}")
        return False

