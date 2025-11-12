"""
Página de clustering de textos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import time

# Adicionar paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.vectorization import vectorize_texts
from src.clustering import (
    reduce_dimensions_pca_umap,
    kmeans_clustering,
    dbscan_clustering,
    evaluate_clustering
)
from src.llm_analysis import (
    name_cluster_with_llm,
    summarize_cluster_with_llm,
    get_top_terms_for_cluster,
    check_llm_availability,
    get_api_keys
)

# Tentar importar sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def check_sentence_transformers():
    """Verifica dinamicamente se sentence_transformers está disponível."""
    # Primeira tentativa: importação direta
    try:
        from sentence_transformers import SentenceTransformer
        return True, None
    except ImportError as e:
        original_error = str(e)
        
        # Segunda tentativa: encontrar e adicionar o .venv ao path
        venv_path = Path(__file__).parent.parent.parent / '.venv'
        
        if not venv_path.exists():
            # Tentar caminho alternativo (se estiver rodando da raiz)
            venv_path = Path(__file__).parent.parent.parent.parent / '.venv'
        
        if venv_path.exists():
            # Determinar caminho do site-packages baseado no OS
            if sys.platform.startswith('win'):
                venv_site_packages = venv_path / 'Lib' / 'site-packages'
            else:
                # Para Linux/Mac, encontrar a versão do Python
                python_dirs = list(venv_path.glob('lib/python*/site-packages'))
                if python_dirs:
                    venv_site_packages = python_dirs[0]
                else:
                    venv_site_packages = None
            
            if venv_site_packages and venv_site_packages.exists():
                venv_path_str = str(venv_site_packages.resolve())
                
                # Adicionar ao path se não estiver lá
                if venv_path_str not in sys.path:
                    sys.path.insert(0, venv_path_str)
                
                # Tentar importar novamente
                try:
                    from sentence_transformers import SentenceTransformer
                    return True, None
                except ImportError:
                    # Adicionar também os caminhos pai (caso haja dependências)
                    venv_lib = venv_site_packages.parent
                    if venv_lib.exists() and str(venv_lib) not in sys.path:
                        sys.path.insert(0, str(venv_lib))
                    try:
                        from sentence_transformers import SentenceTransformer
                        return True, None
                    except ImportError:
                        pass
        
        # Se chegou aqui, não conseguiu importar
        error_msg = f"No module named 'sentence_transformers'\n"
        error_msg += f"Python usado: {sys.executable}\n"
        error_msg += f"Caminhos verificados: {venv_path if venv_path.exists() else 'não encontrado'}\n"
        error_msg += f"sys.path contém {len(sys.path)} diretórios"
        
        return False, error_msg


def render_clustering():
    """Renderiza a página de clustering."""
    from utils.icons import icon_text
    
    st.markdown(
        f'<h1 style="display: inline-flex; align-items: center; gap: 10px;">{icon_text("target", "Clustering de Textos", size=32)}</h1>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Verificar se dados estão carregados
    if not st.session_state.get('data_loaded', False):
        st.warning("Por favor, carregue dados primeiro na página Upload de Dados.")
        return
    
    texts = st.session_state.get('texts')
    labels = st.session_state.get('labels')
    target_names = st.session_state.get('target_names', [])
    
    # Sidebar - Configurações
    st.sidebar.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("settings", "Configurações", size=20)}</h2>',
        unsafe_allow_html=True
    )
    
    vectorization_method = st.sidebar.selectbox(
        "Método de Vetorização:",
        ["TF-IDF", "Embeddings (Sentence Transformers)", "Embeddings (Google Gemini)"],
        key="cluster_vectorization"
    )
    
    # Configurações específicas
    if vectorization_method == "TF-IDF":
        max_features = st.sidebar.slider("Max Features", 1000, 10000, 5000, 500)
        ngram_range = st.sidebar.selectbox(
            "N-gram Range",
            ["(1,1)", "(1,2)", "(1,3)"],
            index=1,
            format_func=lambda x: {"(1,1)": "Unigramas", "(1,2)": "Unigramas + Bigramas", "(1,3)": "Até Trigramas"}[x]
        )
        ngram_range = eval(ngram_range)
    elif vectorization_method == "Embeddings (Sentence Transformers)":
        embedding_model = st.sidebar.selectbox(
            "Modelo de Embedding:",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            help="Modelos de sentence-transformers"
        )
        # Verificação dinâmica
        is_available, error_msg = check_sentence_transformers()
        if not is_available:
            st.sidebar.error("sentence-transformers não está disponível!")
            st.sidebar.info("Certifique-se de que o ambiente virtual (.venv) está ativado.")
            st.sidebar.info("Instale com: pip install sentence-transformers")
            if error_msg:
                with st.sidebar.expander("Detalhes do erro"):
                    st.code(error_msg)
    else:  # Embeddings (Google Gemini)
        st.sidebar.markdown("### 🔑 Configuração da API")
        api_key = st.sidebar.text_input(
            "Google API Key:",
            type="password",
            value=st.session_state.get('google_api_key', ''),
            help="Insira sua chave de API do Google Generative AI"
        )
        st.session_state['google_api_key'] = api_key
        
        if not api_key:
            st.sidebar.warning("Insira uma chave de API para usar Google Gemini")
        
        st.sidebar.info("Obtenha sua chave em: https://makersuite.google.com/app/apikey")
    
    # Seleção de algoritmos de clustering
    st.sidebar.markdown("### Algoritmos de Clustering")
    
    use_kmeans = st.sidebar.checkbox("K-Means", value=True)
    if use_kmeans:
        kmeans_k = st.sidebar.slider("Número de Clusters (K)", 2, 20, 6, 1)
    
    use_dbscan = st.sidebar.checkbox("DBSCAN", value=True)
    if use_dbscan:
        dbscan_eps = st.sidebar.slider("EPS", 0.1, 2.0, 0.5, 0.1)
        dbscan_min_samples = st.sidebar.slider("Min Samples", 2, 50, 10, 1)
    
    # Redução dimensional
    st.sidebar.markdown("### Redução Dimensional")
    use_pca = st.sidebar.checkbox("Usar PCA antes de UMAP/t-SNE", value=True)
    if use_pca:
        pca_components = st.sidebar.slider("Componentes PCA", 10, 100, 50, 10)
    
    reduction_method = st.sidebar.selectbox(
        "Método de Visualização 2D:",
        ["UMAP", "t-SNE"],
        help="Método para reduzir dimensões para visualização 2D"
    )
    
    # Botão para executar clustering
    if st.button("Executar Clustering", type="primary"):
        with st.spinner("Processando..."):
            try:
                from utils.cache import (
                    is_20news_6classes_dataset,
                    load_cached_embeddings,
                    cache_embeddings,
                    load_cached_clustering_results,
                    cache_clustering_results,
                    load_cached_reduction_data,
                    cache_reduction_data
                )
                
                # Verificar se é dataset 20news com 6 classes
                is_20news = st.session_state.get('is_20news_6classes', False)
                if not is_20news:
                    is_20news = is_20news_6classes_dataset(texts, labels if labels is not None else [], target_names)
                
                # Determinar método de vetorização para cache
                if vectorization_method == "TF-IDF":
                    cache_method = 'tfidf'
                elif vectorization_method == "Embeddings (Sentence Transformers)":
                    cache_method = 'sentence_transformer'
                else:
                    cache_method = 'gemini'
                
                # Tentar carregar do cache primeiro (apenas para 20news)
                vectors = None
                X_pca = None
                X_vis = None
                clustering_results = None
                
                if is_20news:
                    # Tentar carregar embeddings do cache
                    if cache_method == 'tfidf':
                        vectors = load_cached_embeddings('tfidf')
                    elif cache_method == 'sentence_transformer':
                        vectors = load_cached_embeddings('sentence_transformer', embedding_model)
                    elif cache_method == 'gemini':
                        vectors = load_cached_embeddings('gemini')
                    
                    # Tentar carregar redução dimensional do cache
                    if vectors is not None and use_pca:
                        reduction_data = load_cached_reduction_data(
                            reduction_method.lower().replace('-', ''), 
                            pca_components
                        )
                        if reduction_data is not None:
                            X_pca = reduction_data['X_pca']
                            X_vis = reduction_data['X_vis']
                    
                    # Tentar carregar resultados de clustering do cache
                    if vectors is not None and X_pca is not None:
                        clustering_results = load_cached_clustering_results(cache_method)
                        if clustering_results is not None:
                            st.info("💾 Carregando resultados do cache...")
                            st.session_state['vectors'] = vectors
                            st.session_state['vectorization_type'] = 'tfidf' if cache_method == 'tfidf' else 'embeddings'
                            if cache_method == 'sentence_transformer':
                                st.session_state['embeddings_model'] = embedding_model
                            elif cache_method == 'gemini':
                                st.session_state['embeddings_model'] = 'gemini-embedding-001'
                            st.session_state['clustering_results'] = clustering_results
                            st.session_state['X_vis'] = X_vis
                            st.session_state['X_pca'] = X_pca
                            st.session_state['reduction_method'] = reduction_method
                            st.success("✅ Clustering carregado do cache!")
                            # Forçar rerun para renderizar os resultados
                            st.rerun()
                
                # Se não encontrou no cache, processar normalmente
                if vectors is None:
                    # Vetorização
                    if vectorization_method == "TF-IDF":
                        vectors, vectorizer = vectorize_texts(
                            texts,
                            max_features=max_features,
                            ngram_range=ngram_range
                        )
                        st.session_state['vectorizer'] = vectorizer
                        st.session_state['vectorization_type'] = 'tfidf'
                        # Salvar no cache
                        if is_20news:
                            cache_embeddings(vectors, 'tfidf')
                    elif vectorization_method == "Embeddings (Sentence Transformers)":
                        # Verificação dinâmica antes de usar
                        is_available, error_msg = check_sentence_transformers()
                        if not is_available:
                            st.error("sentence-transformers não está disponível!")
                            st.info("Certifique-se de executar a aplicação com o ambiente virtual ativado.")
                            if error_msg:
                                with st.expander("Detalhes do erro"):
                                    st.code(error_msg)
                            return
                        # Gerar embeddings com barra de progresso
                        progress_emb = st.progress(0)
                        status_emb = st.empty()
                        
                        def update_emb_progress(current, total, message):
                            """Callback para atualizar progresso dos embeddings."""
                            progress = min(current / total, 1.0)
                            progress_emb.progress(progress)
                            status_emb.text(f"🔄 {message}")
                        
                        vectors = generate_embeddings(texts, embedding_model, progress_callback=update_emb_progress)
                        
                        progress_emb.empty()
                        status_emb.empty()
                        st.session_state['vectorization_type'] = 'embeddings'
                        st.session_state['embeddings_model'] = embedding_model
                        # Salvar no cache
                        if is_20news:
                            cache_embeddings(vectors, 'sentence_transformer', embedding_model)
                    else:  # Embeddings (Google Gemini)
                        api_key = st.session_state.get('google_api_key', '')
                        if not api_key:
                            st.error("Chave de API do Google não configurada!")
                            st.info("Configure sua chave de API na sidebar.")
                            return
                        vectors = generate_gemini_embeddings(texts, api_key)
                        st.session_state['vectorization_type'] = 'embeddings'
                        st.session_state['embeddings_model'] = 'gemini-embedding-001'
                        # Salvar no cache
                        if is_20news:
                            cache_embeddings(vectors, 'gemini')
                
                st.session_state['vectors'] = vectors
                
                # Redução dimensional com barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, message):
                    """Callback para atualizar progresso."""
                    progress = min(current / total, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 {message}")
                
                # Redução dimensional usando função com callback
                if use_pca:
                    X_pca, X_vis, pca_model, reducer = reduce_dimensions_pca_umap(
                        vectors,
                        pca_components=pca_components,
                        umap_components=2,
                        umap_neighbors=15,
                        umap_min_dist=0.1,
                        random_state=42,
                        progress_callback=update_progress
                    )
                    # Salvar no cache
                    if is_20news:
                        cache_reduction_data(X_pca, X_vis, reduction_method.lower().replace('-', ''), pca_components)
                else:
                    # Aplicar redução diretamente (sem PCA)
                    if hasattr(vectors, 'toarray'):
                        X_dense = vectors.toarray()
                    else:
                        X_dense = vectors
                    
                    X_pca = X_dense  # Para clustering, usar dados completos
                    
                    # Redução para visualização
                    update_progress(0, 3, "Aplicando redução dimensional para visualização...")
                    if reduction_method == "UMAP":
                        try:
                            import umap
                            reducer = umap.UMAP(n_components=2, random_state=42)
                            X_vis = reducer.fit_transform(X_dense)
                        except ImportError:
                            from sklearn.manifold import TSNE
                            st.warning("UMAP não disponível, usando t-SNE")
                            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                            X_vis = reducer.fit_transform(X_dense)
                    else:
                        from sklearn.manifold import TSNE
                        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                        X_vis = reducer.fit_transform(X_dense)
                    update_progress(3, 3, "Redução dimensional concluída!")
                
                # Clustering com barras de progresso
                clustering_results = {}
                
                if use_kmeans:
                    clusters_kmeans, kmeans_model = kmeans_clustering(
                        X_pca,
                        n_clusters=kmeans_k,
                        random_state=42,
                        progress_callback=update_progress
                    )
                    metrics_kmeans = evaluate_clustering(X_pca, clusters_kmeans)
                    clustering_results['K-Means'] = {
                        'labels': clusters_kmeans,
                        'model': kmeans_model,
                        'metrics': metrics_kmeans
                    }
                
                if use_dbscan:
                    # Usar número de clusters do K-Means como referência (se disponível)
                    target_n_clusters = kmeans_k if use_kmeans else None
                    clusters_dbscan, dbscan_model, best_eps, best_min_samples = dbscan_clustering(
                        X_pca,
                        eps_values=[dbscan_eps],
                        min_samples_values=[dbscan_min_samples],
                        random_state=42,
                        target_n_clusters=target_n_clusters,
                        progress_callback=update_progress
                    )
                    metrics_dbscan = evaluate_clustering(X_pca, clusters_dbscan)
                    clustering_results['DBSCAN'] = {
                        'labels': clusters_dbscan,
                        'model': dbscan_model,
                        'metrics': metrics_dbscan,
                        'eps': best_eps,
                        'min_samples': best_min_samples
                    }
                
                progress_bar.empty()
                status_text.empty()
                
                # Salvar no cache
                if is_20news:
                    cache_clustering_results(clustering_results, cache_method)
                
                st.session_state['clustering_results'] = clustering_results
                st.session_state['X_vis'] = X_vis
                st.session_state['X_pca'] = X_pca
                st.session_state['reduction_method'] = reduction_method
                
                st.success("Clustering concluído!")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Mostrar resultados
    if st.session_state.get('clustering_results') is not None:
        st.markdown("---")
        clustering_results = st.session_state['clustering_results']
        X_vis = st.session_state.get('X_vis')
        reduction_method = st.session_state.get('reduction_method', 'UMAP')
        
        # Tabs para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs([
            "Métricas",
            "Visualização",
            "Análise",
            "Análise LLM"
        ])
        
        with tab1:
            show_clustering_metrics(clustering_results)
        
        with tab2:
            show_clustering_visualization(clustering_results, X_vis, labels, target_names, reduction_method)
        
        with tab3:
            show_clustering_analysis(clustering_results, X_vis, labels, target_names, reduction_method)
        
        with tab4:
            show_llm_cluster_analysis(clustering_results, st.session_state.get('texts', []))


def generate_embeddings(texts, model_name="all-MiniLM-L6-v2", progress_callback=None):
    """
    Gera embeddings usando sentence-transformers.
    
    Args:
        texts: Lista de textos
        model_name: Nome do modelo
        progress_callback: Função callback para atualizar progresso (recebe (current, total, message))
    """
    if progress_callback:
        progress_callback(0, 2, f"Carregando modelo {model_name}...")
    
    model = SentenceTransformer(model_name)
    
    if progress_callback:
        progress_callback(1, 2, f"Gerando embeddings com {model_name}...")
    
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    if progress_callback:
        progress_callback(2, 2, "Embeddings gerados!")
    
    return embeddings


def generate_gemini_embeddings(texts, api_key, batch_size=1, delay=1.0):
    """
    Gera embeddings usando Google Gemini API.
    
    Args:
        texts: Lista de textos
        api_key: Chave de API do Google
        batch_size: Tamanho do lote (recomendado: 1 para free tier)
        delay: Delay em segundos entre requisições
    
    Returns:
        Array numpy com embeddings
    """
    try:
        import google.generativeai as genai
        import time
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        raise ImportError(
            "google-generativeai não está instalado!\n"
            "Instale com: pip install google-generativeai"
        )
    
    # Configurar API
    genai.configure(api_key=api_key)
    model_name = "models/embedding-001"
    
    embeddings = []
    n_texts = len(texts)
    progress_bar = st.progress(0)
    
    st.info(f"Gerando embeddings com Google Gemini para {n_texts} textos...")
    st.info("Este processo pode demorar devido aos limites da API gratuita.")
    
    max_retries = 3
    
    for i in range(0, n_texts, batch_size):
        batch = texts[i:i+batch_size]
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Gerar embeddings para o lote
                result = genai.embed_content(
                    model=model_name,
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                # Extrair embeddings
                if isinstance(result, dict):
                    if 'embedding' in result:
                        batch_embeddings = result['embedding']
                        if isinstance(batch_embeddings, list):
                            if len(batch_embeddings) > 0 and isinstance(batch_embeddings[0], list):
                                embeddings.extend(batch_embeddings)
                            else:
                                embeddings.extend([batch_embeddings])
                        else:
                            embeddings.append(batch_embeddings)
                    else:
                        batch_embeddings = list(result.values())[0] if result else []
                        if isinstance(batch_embeddings, list):
                            embeddings.extend(batch_embeddings if isinstance(batch_embeddings[0], list) else [batch_embeddings])
                elif isinstance(result, list):
                    embeddings.extend(result)
                else:
                    embeddings.append(result)
                
                success = True
                
            except google_exceptions.ResourceExhausted as e:
                error_msg = str(e)
                if "free_tier" in error_msg.lower() or "limit: 0" in error_msg:
                    progress_bar.empty()
                    raise Exception("Quota da API gratuita excedida. Consulte https://ai.google.dev/gemini-api/docs/rate-limits")
                
                wait_time = delay * (2 ** retry_count)
                time.sleep(wait_time)
                retry_count += 1
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    wait_time = delay * (2 ** retry_count)
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    raise e
        
        if not success:
            raise Exception(f"Erro persistente após {max_retries} tentativas")
        
        # Delay entre lotes
        if i + batch_size < n_texts:
            time.sleep(delay)
        
        # Atualizar progresso
        progress = min((i + batch_size) / n_texts, 1.0)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return np.array(embeddings)


def show_clustering_metrics(clustering_results):
    """Mostra métricas de clustering."""
    from utils.icons import icon_text
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("bar-chart", "Métricas de Clustering", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    metrics_data = {}
    for name, result in clustering_results.items():
        metrics = result['metrics']
        if 'error' not in metrics:
            metrics_data[name] = {
                'Silhouette Score': metrics.get('silhouette', -1),
                'Davies-Bouldin Index': metrics.get('davies_bouldin', float('inf')),
                'Número de Clusters': metrics.get('n_clusters', 0),
                'Pontos de Ruído': metrics.get('n_noise', 0)
            }
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data).T
        
        # Formatando para exibição sem usar st.dataframe/st.table (evita pyarrow)
        # Exibir tabela como markdown
        st.markdown("### Métricas Detalhadas:")
        
        # Criar tabela em markdown
        columns = list(df_metrics.columns)
        markdown_table = "| Algoritmo | " + " | ".join(columns) + " |\n"
        markdown_table += "|" + "|".join(["---" for _ in range(len(columns) + 1)]) + "|\n"
        
        for idx, row in df_metrics.iterrows():
            values = []
            for col in columns:
                val = row[col]
                if np.isinf(val) or np.isnan(val):
                    values.append("N/A")
                elif col in ['Número de Clusters', 'Pontos de Ruído']:
                    values.append(f"{val:.0f}")
                else:
                    values.append(f"{val:.4f}")
            markdown_table += f"| {idx} | " + " | ".join(values) + " |\n"
        
        st.markdown(markdown_table)
        
        # Destacar melhores resultados
        st.markdown("**📌 Melhores resultados:**")
        if 'Silhouette Score' in df_metrics.columns:
            max_idx = df_metrics['Silhouette Score'].idxmax()
            max_val = df_metrics.loc[max_idx, 'Silhouette Score']
            st.markdown(f"- **Silhouette Score** (maior é melhor): `{max_idx}` com **{max_val:.4f}**")
        if 'Davies-Bouldin Index' in df_metrics.columns:
            valid_db = df_metrics['Davies-Bouldin Index'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_db) > 0:
                min_idx = valid_db.idxmin()
                min_val = df_metrics.loc[min_idx, 'Davies-Bouldin Index']
                st.markdown(f"- **Davies-Bouldin Index** (menor é melhor): `{min_idx}` com **{min_val:.4f}**")
        
        # Gráficos de barras com Plotly (cores harmoniosas e valores exatos)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Silhouette Score por Algoritmo', 'Davies-Bouldin Index por Algoritmo'),
            horizontal_spacing=0.15
        )
        
        # Paleta harmoniosa moderna
        colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6']
        
        # Silhouette Score
        silhouette_vals = df_metrics['Silhouette Score'].fillna(0)
        fig.add_trace(
            go.Bar(
                x=df_metrics.index,
                y=silhouette_vals,
                name='Silhouette Score',
                marker_color=colors[0],
                text=[f'{val:.4f}' if not np.isnan(val) else 'N/A' for val in df_metrics['Silhouette Score']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Silhouette Score: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Davies-Bouldin Index (menor é melhor, então usar cor diferente)
        db_scores = df_metrics['Davies-Bouldin Index'].replace([float('inf'), np.inf], np.nan)
        db_valid = db_scores.fillna(0)
        fig.add_trace(
            go.Bar(
                x=df_metrics.index,
                y=db_valid,
                name='Davies-Bouldin Index',
                marker_color=colors[2],
                text=[f'{val:.4f}' if not np.isnan(val) else 'N/A' for val in db_scores],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Davies-Bouldin Index: %{customdata}<extra></extra>',
                customdata=[f'{val:.4f}' if not np.isnan(val) else 'N/A' for val in db_scores]
            ),
            row=1, col=2
        )
        
        # Atualizar layout
        fig.update_xaxes(title_text="Algoritmo", row=1, col=1, tickangle=-45)
        fig.update_xaxes(title_text="Algoritmo", row=1, col=2, tickangle=-45)
        
        # Determinar range do Silhouette Score (geralmente entre -1 e 1)
        silhouette_max = silhouette_vals.max()
        silhouette_min = max(-0.1, silhouette_vals.min())
        fig.update_yaxes(title_text="Silhouette Score", range=[silhouette_min - 0.1, silhouette_max + 0.1], row=1, col=1)
        
        # Determinar range do Davies-Bouldin (pode variar muito)
        db_max = db_valid.max()
        db_min = max(0, db_valid.min())
        fig.update_yaxes(title_text="Davies-Bouldin Index (menor é melhor)", range=[0, db_max * 1.15], row=1, col=2)
        
        fig.update_layout(
            height=450,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Cormorant Garamond, 'Times New Roman', Times, serif", size=16),  # Aumentado ~30%
            margin=dict(l=50, r=50, t=60, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        
        # Salvar visualização
        st.session_state['visualizations']['clustering_metrics'] = fig


def show_clustering_visualization(clustering_results, X_vis, labels, target_names, reduction_method):
    """Mostra visualização dos clusters com Plotly (interativa e bonita)."""
    from utils.icons import icon_text
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("palette", "Visualização dos Clusters", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    n_algorithms = len(clustering_results)
    has_labels = labels is not None
    
    # Paleta harmoniosa moderna (10 cores para clusters)
    colors_palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    # Calcular layout
    if has_labels:
        cols = 2  # clusters, labels verdadeiros (2 colunas por linha)
        rows = n_algorithms
    else:
        cols = 2
        rows = n_algorithms
    
    # Criar subplots com Plotly
    subplot_titles = []
    for name in clustering_results.keys():
        subplot_titles.append(f'{name} Clusters')
        if has_labels:
            subplot_titles.append('Rótulos Verdadeiros')
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
    )
    
    plot_idx = 0
    for algo_idx, (name, result) in enumerate(clustering_results.items()):
        cluster_labels = result['labels']
        
        row = algo_idx + 1
        
        # Visualização dos clusters
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_name = f"Ruído" if cluster_id == -1 else f"Cluster {cluster_id}"
            color = '#9ca3af' if cluster_id == -1 else colors_palette[cluster_id % len(colors_palette)]
            
            fig.add_trace(
                go.Scatter(
                    x=X_vis[mask, 0],
                    y=X_vis[mask, 1],
                    mode='markers',
                    name=cluster_name,
                    marker=dict(
                        size=6,
                        color=color,
                        opacity=0.7,
                        line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                    ),
                    hovertemplate=f'<b>{cluster_name}</b><br>{reduction_method} 1: %{{x:.3f}}<br>{reduction_method} 2: %{{y:.3f}}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=1
            )
        
        fig.update_xaxes(title_text=f"{reduction_method} 1", row=row, col=1)
        fig.update_yaxes(title_text=f"{reduction_method} 2", row=row, col=1)
        
        # Labels verdadeiros (se disponível)
        if has_labels:
            unique_labels = np.unique(labels)
            label_names = target_names if target_names else [f"Classe {i}" for i in unique_labels]
            
            for label_id in unique_labels:
                mask = labels == label_id
                label_name = label_names[label_id] if label_id < len(label_names) else f"Classe {label_id}"
                color = colors_palette[label_id % len(colors_palette)]
                
                fig.add_trace(
                    go.Scatter(
                        x=X_vis[mask, 0],
                        y=X_vis[mask, 1],
                        mode='markers',
                        name=label_name,
                        marker=dict(
                            size=6,
                            color=color,
                            opacity=0.7,
                            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                        ),
                        hovertemplate=f'<b>{label_name}</b><br>{reduction_method} 1: %{{x:.3f}}<br>{reduction_method} 2: %{{y:.3f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=2
                )
            
            fig.update_xaxes(title_text=f"{reduction_method} 1", row=row, col=2)
            fig.update_yaxes(title_text=f"{reduction_method} 2", row=row, col=2)
    
    fig.update_layout(
        height=450 * rows,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Cormorant Garamond, 'Times New Roman', Times, serif", size=14),  # Aumentado ~30%
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
    # Salvar visualização
    st.session_state['visualizations']['clustering_visualization'] = fig


def show_clustering_analysis(clustering_results, X_vis, labels, target_names, reduction_method):
    """Mostra análise detalhada dos clusters."""
    from utils.icons import icon_text
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("chart-line", "Análise dos Clusters", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    for name, result in clustering_results.items():
        st.markdown(f"### {name}")
        
        cluster_labels = result['labels']
        metrics = result['metrics']
        
        # Estatísticas dos clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in cluster_labels else 0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Número de Clusters", n_clusters)
        with col2:
            st.metric("Silhouette Score", f"{metrics.get('silhouette', -1):.4f}")
        with col3:
            db_score = metrics.get('davies_bouldin', float('inf'))
            st.metric("Davies-Bouldin", f"{db_score:.4f}" if db_score != float('inf') else "Inf")
        with col4:
            n_noise = metrics.get('n_noise', 0)
            st.metric("Pontos de Ruído", n_noise)
        
        # Distribuição dos clusters (usando markdown para evitar pyarrow)
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        st.markdown("**Distribuição dos Clusters:**")
        
        # Criar tabela em markdown
        markdown_table = "| Cluster | Tamanho |\n"
        markdown_table += "|---------|--------|\n"
        
        for cluster_id, count in cluster_counts.items():
            cluster_name = "Ruído" if cluster_id == -1 else f"Cluster {cluster_id}"
            markdown_table += f"| {cluster_name} | {count} |\n"
        
        st.markdown(markdown_table)


def show_llm_cluster_analysis(clustering_results, texts):
    """Mostra análise de clusters usando LLM (naming e sumarização)."""
    from utils.icons import icon_text
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("robot", "Análise de Clusters com LLM", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    # Seção de configuração de API Keys
    with st.expander("Configurar Chaves de API", expanded=False):
        st.markdown("""
        **Configure suas chaves de API LLM:** (as chaves serão ocultas por segurança)
        
        **Recomendação:** Use Groq (gratuito e rápido)
        - Obtenha sua chave em: https://console.groq.com/
        """)
        
        # Campos de input com tipo password
        col1, col2, col3 = st.columns(3)
        
        with col1:
            groq_key = st.text_input(
                "Groq API Key",
                value=st.session_state.get('api_keys', {}).get('groq', ''),
                type="password",
                help="Chave da API Groq",
                key="groq_key_input"
            )
        
        with col2:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get('api_keys', {}).get('openai', ''),
                type="password",
                help="Chave da API OpenAI",
                key="openai_key_input"
            )
        
        with col3:
            gemini_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.get('api_keys', {}).get('gemini', ''),
                type="password",
                help="Chave da API Google Gemini",
                key="gemini_key_input"
            )
        
        # Salvar chaves no session state
        if groq_key or openai_key or gemini_key:
            if 'api_keys' not in st.session_state:
                st.session_state['api_keys'] = {}
            
            if groq_key:
                st.session_state['api_keys']['groq'] = groq_key
            if openai_key:
                st.session_state['api_keys']['openai'] = openai_key
            if gemini_key:
                st.session_state['api_keys']['gemini'] = gemini_key
        
        if groq_key or openai_key or gemini_key:
            st.success("Chaves configuradas!")
    
    # Verificar disponibilidade de LLMs (agora incluindo session state)
    session_keys = st.session_state.get('api_keys', {})
    availability = check_llm_availability(session_keys)
    available_providers = [p for p, avail in availability.items() if avail]
    
    if not available_providers:
        st.warning("Nenhuma API de LLM configurada!")
        st.info("""
        Configure pelo menos uma chave de API no painel acima.
        
        **Onde obter chaves:**
        - **Groq**: https://console.groq.com/ (gratuito e recomendado)
        - **OpenAI**: https://platform.openai.com/api-keys
        - **Gemini**: https://ai.google.dev/
        """)
        return
    
    # Seleção de provedor
    provider = st.selectbox(
        "Escolha o provedor de LLM:",
        available_providers,
        help="Groq é recomendado por ser rápido e gratuito"
    )
    
    # Seleção de algoritmo de clustering
    algo_names = list(clustering_results.keys())
    selected_algo = st.selectbox(
        "Escolha o algoritmo de clustering para analisar:",
        algo_names
    )
    
    result = clustering_results[selected_algo]
    cluster_labels = result['labels']
    
    # Gerar nomes e descrições dos clusters
    if st.button("Gerar Nomes e Descrições dos Clusters", type="primary"):
        with st.spinner("Gerando nomes e descrições dos clusters com LLM..."):
            try:
                # Obter vectorizer se disponível (para top terms)
                # Se não estiver disponível, criar um TF-IDF apenas para análise de termos
                vectorizer = st.session_state.get('vectorizer')
                if vectorizer is None and texts:
                    # Criar vectorizer TF-IDF apenas para análise de top-terms
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                    vectorizer.fit(texts)  # Apenas fit, não transform
                
                unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
                
                cluster_info = {}
                progress_bar = st.progress(0)
                total_clusters = len(unique_clusters)
                
                # Adicionar delay entre requisições para respeitar rate limits
                # Gemini Free Tier: 2 req/min, então precisamos de pelo menos 30s entre cada par
                delay_between_clusters = 35.0 if provider == 'gemini' else 1.0  # Delay maior para Gemini
                
                for idx, cluster_id in enumerate(unique_clusters):
                    progress_bar.progress((idx) / total_clusters)
                    
                    # Textos deste cluster
                    mask = cluster_labels == cluster_id
                    cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
                    
                    if not cluster_texts:
                        continue
                    
                    # Obter top termos (se vectorizer disponível)
                    top_terms = []
                    if vectorizer is not None:
                        top_terms = get_top_terms_for_cluster(cluster_texts, vectorizer, n_terms=20)
                    
                    # Obter chave de API do session state
                    api_key = session_keys.get(provider) if session_keys else None
                    
                    # Status atual
                    status_text = st.empty()
                    status_text.info(f"Processando Cluster {cluster_id} ({idx+1}/{total_clusters})...")
                    
                    # Containers para streaming de resposta
                    desc_container = st.empty()
                    summary_container = st.empty()
                    
                    try:
                        # Callback para streaming (otimizado: uma única chamada)
                        response_stream = ""
                        def stream_callback(chunk):
                            nonlocal response_stream
                            response_stream += chunk
                            desc_container.markdown(f"**Gerando análise...**\n```\n{response_stream}\n```")
                        
                        # Gerar nome, descrição e sumário em UMA ÚNICA chamada (otimizado)
                        name, description, summary = name_cluster_with_llm(
                            cluster_texts,
                            cluster_id,
                            top_terms,
                            provider=provider,
                            api_key=api_key,
                            stream_callback=stream_callback,
                            include_summary=True  # Incluir sumário na mesma chamada
                        )
                        
                        # Mostrar resultados
                        desc_container.markdown(f"**Nome:** {name}\n\n**Descrição:** {description}")
                        if summary:
                            summary_container.markdown(f"**Sumário:** {summary}")
                        else:
                            summary_container.markdown("*Sumário não disponível*")
                        
                        cluster_info[cluster_id] = {
                            'name': name,
                            'description': description,
                            'summary': summary or "",
                            'size': len(cluster_texts),
                            'top_terms': top_terms[:15]  # Top 15 termos para análise
                        }
                        
                        status_text.success(f"Cluster {cluster_id} concluído!")
                        
                    except Exception as e:
                        error_msg = str(e)
                        # Se for quota error, mostrar aviso especial
                        if 'quota' in error_msg.lower() or '429' in error_msg:
                            cluster_info[cluster_id] = {
                                'name': f"Cluster {cluster_id}",
                                'description': f"Rate limit excedido. Aguarde alguns minutos ou use Groq/OpenAI.",
                                'summary': "",
                                'size': len(cluster_texts),
                                'top_terms': top_terms[:10]
                            }
                            status_text.warning(f"Cluster {cluster_id}: Rate limit atingido")
                        else:
                            cluster_info[cluster_id] = {
                                'name': f"Cluster {cluster_id}",
                                'description': f"Erro: {error_msg[:100]}...",
                                'summary': "",
                                'size': len(cluster_texts),
                                'top_terms': top_terms[:10]
                            }
                            status_text.error(f"Cluster {cluster_id} falhou")
                    
                    # Delay entre clusters (importante para Gemini)
                    if idx < total_clusters - 1 and provider == 'gemini':
                        status_text.info(f"Aguardando {delay_between_clusters:.0f}s (rate limit Gemini)...")
                        time.sleep(delay_between_clusters)
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                # Salvar no session state
                if 'cluster_names_llm' not in st.session_state:
                    st.session_state['cluster_names_llm'] = {}
                st.session_state['cluster_names_llm'][selected_algo] = cluster_info
                
                st.success(f"Análise concluída para {len(cluster_info)} clusters!")
                
            except Exception as e:
                st.error(f"Erro ao gerar análise: {str(e)}")
                import traceback
                with st.expander("Detalhes do erro"):
                    st.code(traceback.format_exc())
    
    # Mostrar resultados se disponíveis
    cluster_info = st.session_state.get('cluster_names_llm', {}).get(selected_algo, {})
    
    if cluster_info:
        st.markdown("---")
        from utils.icons import icon_text
        st.markdown(
            f'<h3 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("clipboard", "Clusters Analisados", size=20)}</h3>',
            unsafe_allow_html=True
        )
        
        for cluster_id in sorted(cluster_info.keys()):
            info = cluster_info[cluster_id]
            
            with st.expander(f"{info['name']} (ID: {cluster_id}, {info['size']} documentos)", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Descrição (LLM):**")
                    st.markdown(info['description'])
                    
                    if info.get('summary'):
                        st.markdown("**📄 Sumário do Tópico (LLM):**")
                        st.markdown(info['summary'])
                
                with col2:
                    st.markdown("**🔤 Top Termos TF-IDF:**")
                    if info.get('top_terms'):
                        # Mostrar top termos como badges
                        terms_text = " ".join([f"`{term}`" for term in info['top_terms'][:10]])
                        st.markdown(terms_text)
                        st.caption(f"*Top {len(info['top_terms'])} termos mais relevantes calculados via TF-IDF*")
                    else:
                        st.info("Termos não disponíveis")
                    
                    st.markdown(f"**📊 Tamanho:** {info['size']} documentos")
                
                # Análise de coerência (conforme orientação do professor)
                if info.get('top_terms') and info.get('description'):
                    st.markdown("---")
                    st.markdown("**🔍 Análise de Coerência:**")
                    st.caption(
                        "Comparação entre a descrição gerada pelo LLM e os termos mais relevantes (TF-IDF). "
                        "Se os termos aparecem na descrição, há coerência entre o agrupamento e a interpretação."
                    )
    else:
        st.info("Clique no botão acima para gerar nomes e descrições dos clusters usando LLM.")

