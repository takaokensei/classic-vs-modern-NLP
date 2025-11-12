"""
Módulo de clustering de textos.

Funções utilitárias para realizar clustering e redução dimensional.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from typing import Optional, Callable

# Tentar importar UMAP, se não disponível usar TSNE
# Nota: umap-learn requer Python < 3.14 devido ao numba
# Se você está no Python 3.14+, o código usará t-SNE automaticamente
# (Erros de linting sobre 'umap' podem ser ignorados)
try:
    import umap
    USE_UMAP = True
except ImportError:
    USE_UMAP = False


def reduce_dimensions_pca_umap(X, pca_components=50, umap_components=2, 
                               umap_neighbors=15, umap_min_dist=0.1, random_state=42,
                               progress_callback: Optional[Callable] = None):
    """
    Aplica redução dimensional: PCA seguido de UMAP.
    
    Args:
        X: Matriz de features
        pca_components: Número de componentes do PCA
        umap_components: Número de componentes do UMAP (geralmente 2 para visualização)
        umap_neighbors: Número de vizinhos para UMAP
        umap_min_dist: Distância mínima para UMAP
        random_state: Seed para reprodutibilidade
        progress_callback: Função callback para atualizar progresso (recebe (current, total, message))
    
    Returns:
        Tupla (X_pca, X_umap, pca_model, umap_model)
    """
    # Converter sparse matrix para dense se necessário
    if progress_callback:
        progress_callback(0, 3, "Convertendo matriz para formato denso...")
    
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # PCA
    if progress_callback:
        progress_callback(1, 3, f"Aplicando PCA ({pca_components} componentes)...")
    
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_dense)
    
    # Redução dimensional (UMAP ou TSNE)
    if progress_callback:
        method = "UMAP" if USE_UMAP else "t-SNE"
        progress_callback(2, 3, f"Aplicando {method} para visualização 2D...")
    
    if USE_UMAP:
        reducer = umap.UMAP(
            n_components=umap_components,
            random_state=random_state,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist
        )
        X_vis = reducer.fit_transform(X_pca)
    else:
        reducer = TSNE(
            n_components=umap_components,
            random_state=random_state,
            perplexity=30,
            max_iter=1000
        )
        X_vis = reducer.fit_transform(X_pca)
    
    if progress_callback:
        progress_callback(3, 3, "Redução dimensional concluída!")
    
    return X_pca, X_vis, pca, reducer


def kmeans_clustering(X, n_clusters=6, random_state=42, n_init=10,
                     progress_callback: Optional[Callable] = None):
    """
    Aplica clustering K-Means.
    
    Args:
        X: Matriz de features
        n_clusters: Número de clusters
        random_state: Seed para reprodutibilidade
        n_init: Número de inicializações
        progress_callback: Função callback para atualizar progresso (recebe (current, total, message))
    
    Returns:
        Tupla (labels, kmeans_model)
    """
    if progress_callback:
        progress_callback(0, 1, f"Executando K-Means com {n_clusters} clusters ({n_init} inicializações)...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(X)
    
    if progress_callback:
        progress_callback(1, 1, "K-Means concluído!")
    
    return labels, kmeans


def dbscan_clustering(X, eps_values=None, min_samples_values=None, 
                     metric='euclidean', random_state=42,
                     target_n_clusters: Optional[int] = None,
                     progress_callback: Optional[Callable] = None):
    """
    Aplica clustering DBSCAN com busca automática de parâmetros.
    
    Args:
        X: Matriz de features
        eps_values: Lista de valores de eps para testar
        min_samples_values: Lista de valores de min_samples para testar
        metric: Métrica de distância
        random_state: Seed para reprodutibilidade (não usado diretamente, mas para consistência)
        target_n_clusters: Número de clusters desejado (opcional, usado para otimização)
        progress_callback: Função callback para atualizar progresso (recebe (current, total, message))
    
    Returns:
        Tupla (labels, dbscan_model, best_eps, best_min_samples)
    """
    if eps_values is None:
        eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    if min_samples_values is None:
        min_samples_values = [5, 10, 15]
    
    total_combinations = len(eps_values) * len(min_samples_values)
    current_combination = 0
    
    best_dbscan = None
    best_score = -1
    best_eps = None
    best_min_samples = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            current_combination += 1
            if progress_callback:
                progress_callback(
                    current_combination, 
                    total_combinations, 
                    f"Testando DBSCAN: eps={eps}, min_samples={min_samples} ({current_combination}/{total_combinations})..."
                )
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            noise_ratio = n_noise / len(labels) if len(labels) > 0 else 1.0
            
            # Só calcular métricas se houver pelo menos 2 clusters e não houver muito ruído
            if n_clusters >= 2 and noise_ratio < 0.5:  # Máximo 50% de ruído
                try:
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 0:
                        silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                        
                        # Calcular score combinado
                        # Penalizar muitos clusters pequenos e muito ruído
                        score = silhouette
                        
                        # Se temos um número alvo de clusters, penalizar desvios
                        if target_n_clusters is not None:
                            # Penalizar se o número de clusters está muito longe do alvo
                            cluster_diff = abs(n_clusters - target_n_clusters)
                            # Penalização suave: reduzir score em até 0.2 se estiver muito longe
                            penalty = min(0.2, cluster_diff * 0.05)
                            score = score * (1 - penalty)
                        
                        # Penalizar muitos clusters pequenos (mais de 2x o número esperado)
                        if target_n_clusters is not None and n_clusters > target_n_clusters * 2:
                            # Penalização adicional para muitos clusters
                            excess_clusters = n_clusters - target_n_clusters * 2
                            penalty = min(0.3, excess_clusters * 0.05)
                            score = score * (1 - penalty)
                        
                        # Penalizar muito ruído
                        if noise_ratio > 0.3:
                            score = score * (1 - (noise_ratio - 0.3) * 0.5)  # Penalizar ruído acima de 30%
                        
                        if score > best_score:
                            best_score = score
                            best_dbscan = dbscan
                            best_eps = eps
                            best_min_samples = min_samples
                except:
                    pass
    
    if progress_callback:
        progress_callback(total_combinations, total_combinations, "DBSCAN concluído!")
    
    if best_dbscan is not None:
        labels = best_dbscan.fit_predict(X)
        return labels, best_dbscan, best_eps, best_min_samples
    else:
        # Usar valores padrão
        if progress_callback:
            progress_callback(1, 1, "Usando parâmetros padrão do DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=10, metric=metric)
        labels = dbscan.fit_predict(X)
        return labels, dbscan, 0.5, 10


def evaluate_clustering(X, labels, metrics=['silhouette', 'davies_bouldin']):
    """
    Avalia qualidade do clustering.
    
    Args:
        X: Matriz de features
        labels: Rótulos dos clusters
        metrics: Lista de métricas a calcular
    
    Returns:
        Dicionário com métricas calculadas
    """
    results = {}
    
    # Remover pontos de ruído (-1) se houver
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) == 0:
        return {'error': 'Todos os pontos são ruído'}
    
    X_clean = X[non_noise_mask]
    labels_clean = labels[non_noise_mask]
    
    n_clusters = len(set(labels_clean))
    if n_clusters < 2:
        return {'error': 'Menos de 2 clusters encontrados'}
    
    if 'silhouette' in metrics:
        try:
            results['silhouette'] = silhouette_score(X_clean, labels_clean)
        except:
            results['silhouette'] = -1
    
    if 'davies_bouldin' in metrics:
        try:
            results['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
        except:
            results['davies_bouldin'] = float('inf')
    
    results['n_clusters'] = n_clusters
    results['n_noise'] = np.sum(~non_noise_mask)
    
    return results

