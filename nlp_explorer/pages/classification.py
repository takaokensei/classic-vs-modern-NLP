"""
Página de classificação de textos.
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

# Adicionar paths
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from src.vectorization import vectorize_texts
from src.classification import train_and_evaluate, get_default_classifiers
from sklearn.model_selection import train_test_split
from utils.progress import StreamlitProgress

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


def render_classification():
    """Renderiza a página de classificação."""
    from utils.icons import icon_text
    
    st.markdown(
        f'<h1 style="display: inline-flex; align-items: center; gap: 10px;">{icon_text("search", "Classificação de Textos", size=32)}</h1>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Verificar se dados estão carregados
    if not st.session_state.get('data_loaded', False):
        st.warning("Por favor, carregue dados primeiro na página Upload de Dados.")
        return
    
    if st.session_state.get('labels') is None:
        st.warning("Dados carregados não possuem rótulos. Classificação requer dados rotulados.")
        return
    
    texts = st.session_state.get('texts')
    labels = np.array(st.session_state.get('labels'))
    target_names = st.session_state.get('target_names', [])
    
    # Sidebar - Configurações
    st.sidebar.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("settings", "Configurações", size=20)}</h2>',
        unsafe_allow_html=True
    )
    
    vectorization_method = st.sidebar.selectbox(
        "Método de Vetorização:",
        ["TF-IDF", "Embeddings (Sentence Transformers)", "Embeddings (Google Gemini)"],
        key="class_vectorization"
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
    
    # Seleção de classificadores
    st.sidebar.markdown("### Classificadores")
    default_classifiers = get_default_classifiers()
    
    selected_classifiers = {}
    for name, model in default_classifiers.items():
        # Ajustar classificadores baseado no método de vetorização
        if vectorization_method == "TF-IDF" and name == "GaussianNB":
            continue  # GaussianNB não funciona bem com TF-IDF sparse
        if vectorization_method != "TF-IDF" and name == "MultinomialNB":
            continue  # MultinomialNB requer dados não-negativos
        # SVM funciona com ambos, mas pode ser lento com datasets grandes
        if "SVM" in name:
            st.sidebar.caption(f"⚠️ {name} pode ser lento com muitos dados")
        
        if st.sidebar.checkbox(name, value=True, key=f"classifier_{name}"):
            selected_classifiers[name] = model
    
    if len(selected_classifiers) == 0:
        st.warning("⚠️ Selecione pelo menos um classificador.")
        return
    
    # Botão para executar classificação
    if st.button("🚀 Executar Classificação", type="primary"):
        with st.spinner("🔄 Processando..."):
            try:
                from utils.cache import (
                    is_20news_6classes_dataset, 
                    load_cached_embeddings, 
                    cache_embeddings,
                    load_cached_classification_results,
                    cache_classification_results
                )
                
                # Verificar se é dataset 20news com 6 classes
                is_20news = st.session_state.get('is_20news_6classes', False)
                if not is_20news:
                    is_20news = is_20news_6classes_dataset(texts, labels, target_names)
                
                # Determinar método de vetorização para cache
                if vectorization_method == "TF-IDF":
                    cache_method = 'tfidf'
                elif vectorization_method == "Embeddings (Sentence Transformers)":
                    cache_method = 'sentence_transformer'
                else:
                    cache_method = 'gemini'
                
                # Tentar carregar do cache primeiro (apenas para 20news)
                vectors = None
                results = None
                
                if is_20news:
                    # Tentar carregar embeddings do cache
                    if cache_method == 'tfidf':
                        vectors = load_cached_embeddings('tfidf')
                    elif cache_method == 'sentence_transformer':
                        vectors = load_cached_embeddings('sentence_transformer', embedding_model)
                    elif cache_method == 'gemini':
                        vectors = load_cached_embeddings('gemini')
                    
                    # Tentar carregar resultados de classificação do cache (combinando vetorização + classificadores)
                    if vectors is not None:
                        results = load_cached_classification_results(cache_method, selected_classifiers)
                        if results is not None:
                            # Verificar se os resultados têm os classificadores esperados
                            cached_classifier_names = set(results.get('results', {}).keys())
                            selected_classifier_names = set(selected_classifiers.keys())
                            
                            if selected_classifier_names.issubset(cached_classifier_names):
                                st.info("💾 Carregando resultados do cache...")
                                st.session_state['vectors'] = vectors
                                st.session_state['vectorization_type'] = 'tfidf' if cache_method == 'tfidf' else 'embeddings'
                                if cache_method == 'sentence_transformer':
                                    st.session_state['embeddings_model'] = embedding_model
                                elif cache_method == 'gemini':
                                    st.session_state['embeddings_model'] = 'gemini-embedding-001'
                                st.session_state['classification_results'] = results
                                st.session_state['classification_models'] = selected_classifiers
                                st.success("✅ Classificação carregada do cache!")
                                # Forçar rerun para renderizar os resultados
                                st.rerun()
                            else:
                                # Cache encontrado mas não tem todos os classificadores selecionados
                                missing = selected_classifier_names - cached_classifier_names
                                st.warning(f"⚠️ Cache encontrado, mas faltam classificadores: {', '.join(missing)}. Processando apenas os faltantes...")
                                # Continuar para processar os faltantes
                        else:
                            # Cache não encontrado - mostrar informação
                            st.info("ℹ️ Cache não encontrado para esta combinação. Processando...")
                
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
                            st.error("❌ sentence-transformers não está disponível!")
                            st.info("💡 Certifique-se de executar a aplicação com o ambiente virtual ativado.")
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
                            st.error("❌ Chave de API do Google não configurada!")
                            st.info("💡 Configure sua chave de API na sidebar.")
                            return
                        vectors = generate_gemini_embeddings(texts, api_key)
                        st.session_state['vectorization_type'] = 'embeddings'
                        st.session_state['embeddings_model'] = 'gemini-embedding-001'
                        # Salvar no cache
                        if is_20news:
                            cache_embeddings(vectors, 'gemini')
                
                st.session_state['vectors'] = vectors
                
                # Classificação com barra de progresso
                n_classifiers = len(selected_classifiers)
                total_steps = n_classifiers * 2  # Treino + CV para cada
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, message):
                    """Callback para atualizar progresso."""
                    progress = min(current / total, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 {message}")
                
                # Preparar vetores para treinamento (converter esparso para denso se necessário)
                vectors_for_training = vectors
                if cache_method == 'tfidf' and hasattr(vectors, 'toarray'):
                    # Verificar se tem classificadores que precisam de dados densos
                    needs_dense = any('SVM' in name for name in selected_classifiers.keys())
                    if needs_dense:
                        vectors_for_training = vectors.toarray()
                
                # Treinar apenas com os classificadores selecionados pelo usuário
                results = train_and_evaluate(
                    vectors_for_training,
                    labels,
                    classifiers=selected_classifiers,
                    test_size=0.2,
                    random_state=42,
                    stratify=True,
                    progress_callback=update_progress
                )
                
                # Salvar no cache (apenas para 20news) - salva combinação específica de vetorização + classificadores
                if is_20news:
                    cache_classification_results(results, cache_method, selected_classifiers)
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['classification_results'] = results
                st.session_state['classification_models'] = selected_classifiers
                
                st.success("✅ Classificação concluída!")
                
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Mostrar resultados
    if st.session_state.get('classification_results') is not None:
        st.markdown("---")
        results = st.session_state['classification_results']
        
        # Tabs para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["📊 Métricas", "🔢 Matriz de Confusão", "📈 Comparação"])
        
        with tab1:
            show_classification_metrics(results, target_names)
        
        with tab2:
            show_confusion_matrices(results, target_names)
        
        with tab3:
            show_classification_comparison(results)


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


def show_classification_metrics(results, target_names=None):
    """Mostra métricas de classificação."""
    from utils.icons import icon_text
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("bar-chart", "Métricas de Desempenho", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    # Criar DataFrame com resultados
    results_dict = results['results']
    cv_results = results['cv_results']
    
    # Extrair métricas (com fallback para compatibilidade)
    df_metrics_data = {
        'Accuracy': [r.get('accuracy', 0) for r in results_dict.values()],
        'F1 Macro': [r.get('f1_macro', 0) for r in results_dict.values()],
        'Precision Macro': [r.get('precision_macro', 0) for r in results_dict.values()],
        'Recall Macro': [r.get('recall_macro', 0) for r in results_dict.values()],
        'F1 CV (Mean)': [cv['mean'] for cv in cv_results.values()],
        'F1 CV (Std)': [cv['std'] for cv in cv_results.values()]
    }
    
    df_metrics = pd.DataFrame(df_metrics_data, index=results_dict.keys())
    
    # Formatando para exibição sem usar st.dataframe/st.table (evita pyarrow)
    # Exibir tabela como markdown
    st.markdown("### Métricas Detalhadas:")
    
    # Criar tabela em markdown
    markdown_table = "| Classificador | Accuracy | Precision | Recall | F1 Macro | F1 CV (Mean) | F1 CV (Std) |\n"
    markdown_table += "|" + "|".join(["---" for _ in range(7)]) + "|\n"
    
    for idx, row in df_metrics.iterrows():
        accuracy = f"{row['Accuracy']:.4f}"
        precision = f"{row['Precision Macro']:.4f}"
        recall = f"{row['Recall Macro']:.4f}"
        f1_macro = f"{row['F1 Macro']:.4f}"
        f1_cv_mean = f"{row['F1 CV (Mean)']:.4f}"
        f1_cv_std = f"{row['F1 CV (Std)']:.4f}"
        markdown_table += f"| {idx} | {accuracy} | {precision} | {recall} | {f1_macro} | {f1_cv_mean} | {f1_cv_std} |\n"
    
    st.markdown(markdown_table)
    
    # Destacar valores máximos
    st.markdown("**📌 Melhores resultados:**")
    for col in ['Accuracy', 'Precision Macro', 'Recall Macro', 'F1 Macro', 'F1 CV (Mean)']:
        if col in df_metrics.columns:
            max_idx = df_metrics[col].idxmax()
            max_val = df_metrics.loc[max_idx, col]
            st.markdown(f"- **{col}**: `{max_idx}` com **{max_val:.4f}**")
    
    # Gráficos de barras com Plotly (cores harmoniosas e valores exatos)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy por Classificador', 'F1 Macro por Classificador', 
                       'Precision Macro por Classificador', 'Recall Macro por Classificador'),
        horizontal_spacing=0.15,
        vertical_spacing=0.2
    )
    
    # Paleta harmoniosa moderna
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6']
    
    # Accuracy
    fig.add_trace(
        go.Bar(
            x=df_metrics.index,
            y=df_metrics['Accuracy'],
            name='Accuracy',
            marker_color=colors[0],
            text=[f'{val:.4f}' for val in df_metrics['Accuracy']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # F1 Macro
    fig.add_trace(
        go.Bar(
            x=df_metrics.index,
            y=df_metrics['F1 Macro'],
            name='F1 Macro',
            marker_color=colors[2],
            text=[f'{val:.4f}' for val in df_metrics['F1 Macro']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>F1 Macro: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Precision Macro
    fig.add_trace(
        go.Bar(
            x=df_metrics.index,
            y=df_metrics['Precision Macro'],
            name='Precision Macro',
            marker_color=colors[4],
            text=[f'{val:.4f}' for val in df_metrics['Precision Macro']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Precision Macro: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Recall Macro
    fig.add_trace(
        go.Bar(
            x=df_metrics.index,
            y=df_metrics['Recall Macro'],
            name='Recall Macro',
            marker_color=colors[5],
            text=[f'{val:.4f}' for val in df_metrics['Recall Macro']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Recall Macro: %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Atualizar layout
    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(title_text="Classificador", row=row, col=col, tickangle=-45)
        fig.update_yaxes(range=[0, 1.05], row=row, col=col)
    
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="F1 Macro", row=1, col=2)
    fig.update_yaxes(title_text="Precision Macro", row=2, col=1)
    fig.update_yaxes(title_text="Recall Macro", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Cormorant Garamond, 'Times New Roman', Times, serif", size=16),  # Aumentado ~30%
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
    # Salvar referência
    st.session_state['visualizations']['classification_metrics'] = fig
    
    # Adicionar seção de métricas por classe
    if target_names:
        st.markdown("---")
        show_per_class_metrics(results, target_names)


def show_per_class_metrics(results, target_names):
    """Mostra métricas de Precision, Recall e F1 por classe para cada classificador."""
    from utils.icons import icon_text
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    st.markdown(
        f'<h3 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("grid", "Métricas por Classe", size=20)}</h3>',
        unsafe_allow_html=True
    )
    
    results_dict = results['results']
    y_test = results['y_test']
    predictions = results['predictions']
    
    # Selecionar classificador para análise detalhada
    if len(results_dict) > 0:
        selected_classifier = st.selectbox(
            "Selecione um classificador para ver métricas detalhadas por classe:",
            list(results_dict.keys()),
            key="per_class_classifier"
        )
        
        y_pred = predictions[selected_classifier]
        result = results_dict[selected_classifier]
        
        # Obter relatório de classificação se disponível
        if 'classification_report' in result:
            report = result['classification_report']
            
            # Extrair métricas por classe
            classes_data = []
            for class_idx, class_name in enumerate(target_names):
                if str(class_idx) in report:
                    class_metrics = report[str(class_idx)]
                    classes_data.append({
                        'Classe': class_name,
                        'Precision': class_metrics.get('precision', 0),
                        'Recall': class_metrics.get('recall', 0),
                        'F1-Score': class_metrics.get('f1-score', 0),
                        'Support': class_metrics.get('support', 0)
                    })
            
            if classes_data:
                df_per_class = pd.DataFrame(classes_data)
                
                # Tabela de métricas por classe
                st.markdown(f"### Métricas Detalhadas - {selected_classifier}")
                
                markdown_table = "| Classe | Precision | Recall | F1-Score | Support |\n"
                markdown_table += "|" + "|".join(["---" for _ in range(5)]) + "|\n"
                
                for _, row in df_per_class.iterrows():
                    precision = f"{row['Precision']:.4f}"
                    recall = f"{row['Recall']:.4f}"
                    f1 = f"{row['F1-Score']:.4f}"
                    support = f"{int(row['Support'])}"
                    markdown_table += f"| {row['Classe']} | {precision} | {recall} | {f1} | {support} |\n"
                
                st.markdown(markdown_table)
                
                # Gráfico de métricas por classe
                fig = go.Figure()
                
                x_classes = df_per_class['Classe']
                
                fig.add_trace(go.Bar(
                    x=x_classes,
                    y=df_per_class['Precision'],
                    name='Precision',
                    marker_color='#6366f1',
                    text=[f'{val:.3f}' for val in df_per_class['Precision']],
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    x=x_classes,
                    y=df_per_class['Recall'],
                    name='Recall',
                    marker_color='#8b5cf6',
                    text=[f'{val:.3f}' for val in df_per_class['Recall']],
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    x=x_classes,
                    y=df_per_class['F1-Score'],
                    name='F1-Score',
                    marker_color='#ec4899',
                    text=[f'{val:.3f}' for val in df_per_class['F1-Score']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f'Métricas por Classe - {selected_classifier}',
                    xaxis_title='Classe',
                    yaxis_title='Score',
                    barmode='group',
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Cormorant Garamond, 'Times New Roman', Times, serif", size=14),
                    margin=dict(l=50, r=50, t=80, b=100),
                    xaxis=dict(tickangle=-45)
                )
                
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                
                # Salvar visualização
                st.session_state['visualizations'][f'per_class_metrics_{selected_classifier}'] = fig
        else:
            # Fallback: calcular métricas manualmente
            st.info("Calculando métricas por classe...")
            try:
                from sklearn.metrics import precision_recall_fscore_support
                
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred, labels=range(len(target_names)), zero_division=0
                )
                
                classes_data = []
                for class_idx, class_name in enumerate(target_names):
                    if class_idx < len(precision):
                        classes_data.append({
                            'Classe': class_name,
                            'Precision': precision[class_idx],
                            'Recall': recall[class_idx],
                            'F1-Score': f1[class_idx],
                            'Support': support[class_idx]
                        })
                
                if classes_data:
                    df_per_class = pd.DataFrame(classes_data)
                    
                    st.markdown(f"### Métricas Detalhadas - {selected_classifier}")
                    
                    markdown_table = "| Classe | Precision | Recall | F1-Score | Support |\n"
                    markdown_table += "|" + "|".join(["---" for _ in range(5)]) + "|\n"
                    
                    for _, row in df_per_class.iterrows():
                        precision = f"{row['Precision']:.4f}"
                        recall = f"{row['Recall']:.4f}"
                        f1 = f"{row['F1-Score']:.4f}"
                        support = f"{int(row['Support'])}"
                        markdown_table += f"| {row['Classe']} | {precision} | {recall} | {f1} | {support} |\n"
                    
                    st.markdown(markdown_table)
            except Exception as e:
                st.warning(f"Não foi possível calcular métricas por classe: {str(e)}")


def show_confusion_matrices(results, target_names):
    """Mostra matrizes de confusão com Plotly."""
    from utils.icons import icon_text
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("hash", "Matrizes de Confusão", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    from sklearn.metrics import confusion_matrix
    y_test = results['y_test']
    predictions = results['predictions']
    
    n_classifiers = len(predictions)
    cols = 2
    rows = (n_classifiers + 1) // 2
    
    # Criar subplots com Plotly
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[name for name in predictions.keys()],
        horizontal_spacing=0.15,
        vertical_spacing=0.15,
        specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Paleta harmoniosa para heatmap
    colorscale = [[0, '#f0f9ff'], [0.5, '#3b82f6'], [1, '#1e40af']]
    
    plot_idx = 0
    for idx, (name, y_pred) in enumerate(predictions.items()):
        # Garantir que a matriz de confusão use a mesma ordem de target_names
        # Isso garante que linha i corresponde à classe target_names[i]
        # e coluna j corresponde à classe target_names[j]
        unique_labels = sorted(set(y_test) | set(y_pred))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        # Criar matriz de confusão com labels ordenados
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Mapear labels numéricos para nomes das classes na mesma ordem
        # Garantir que target_names e a matriz estejam na mesma ordem
        labels_sorted = sorted(set(y_test) | set(y_pred))
        class_names_ordered = [target_names[label] if label < len(target_names) else f"Classe {label}" 
                                for label in labels_sorted]
        
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        # Criar texto para anotação (valores formatados)
        text = [[f'{val:.2f}' for val in row] for row in cm_normalized]
        
        # A diagonal principal agora deve estar correta:
        # z[i, j] = proporção de classe i (real) predita como classe j (predito)
        # Quando i == j, temos acertos (diagonal principal)
        fig.add_trace(
            go.Heatmap(
                z=cm_normalized,
                x=class_names_ordered,  # Predito (colunas) - mesma ordem
                y=class_names_ordered,  # Real (linhas) - mesma ordem
                text=text,
                texttemplate='%{text}',
                textfont={"size": 15},  # Aumentado ~30% para melhor legibilidade
                colorscale=colorscale,
                showscale=(idx == 0),  # Mostrar colorbar apenas no primeiro
                colorbar=dict(title="Proporção", len=0.6, y=0.5) if idx == 0 else None,
                hovertemplate='<b>Real: %{y}</b><br>Predito: %{x}<br>Proporção: %{z:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Inverter eixo Y para que a primeira classe apareça no topo (padrão visual)
        fig.update_yaxes(
            title_text="Real",
            row=row,
            col=col,
            autorange="reversed",  # Inverter Y para que diagonal principal fique de cima-esquerda para baixo-direita
            title_font=dict(size=14),  # Aumentado ~30% (correto: title_font, não titlefont)
            tickfont=dict(size=12)  # Aumentado ~30%
        )
        fig.update_xaxes(
            title_text="Predito", 
            row=row, 
            col=col, 
            tickangle=-45,
            title_font=dict(size=14),  # Aumentado ~30% (correto: title_font, não titlefont)
            tickfont=dict(size=12)  # Aumentado ~30%
        )
    
    fig.update_layout(
        height=400 * rows,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Cormorant Garamond, 'Times New Roman', Times, serif", size=14),  # Aumentado ~30%
        margin=dict(l=80, r=50, t=60, b=80)
    )
    
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
    # Salvar visualização
    st.session_state['visualizations']['confusion_matrices'] = fig


def show_classification_comparison(results):
    """Mostra comparação entre classificadores."""
    from utils.icons import icon_text
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("chart-line", "Comparação de Classificadores", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    results_dict = results['results']
    cv_results = results['cv_results']
    
    # Preparar dados para gráfico
    models = list(results_dict.keys())
    accuracy = [r['accuracy'] for r in results_dict.values()]
    f1 = [r['f1_macro'] for r in results_dict.values()]
    f1_cv_mean = [cv['mean'] for cv in cv_results.values()]
    f1_cv_std = [cv['std'] for cv in cv_results.values()]
    
    # Gráfico de comparação com Plotly (interativo e com valores)
    fig = go.Figure()
    
    # Paleta harmoniosa moderna
    colors = ['#6366f1', '#8b5cf6', '#ec4899']
    
    x_positions = np.arange(len(models))
    width = 0.25
    
    # Accuracy
    fig.add_trace(go.Bar(
        x=x_positions - width,
        y=accuracy,
        name='Accuracy',
        marker_color=colors[0],
        text=[f'{val:.4f}' for val in accuracy],
        textposition='outside',
        hovertemplate='<b>%{text}</b><br>Classificador: %{customdata}<extra></extra>',
        customdata=models,
        width=width
    ))
    
    # F1 Macro
    fig.add_trace(go.Bar(
        x=x_positions,
        y=f1,
        name='F1 Macro',
        marker_color=colors[1],
        text=[f'{val:.4f}' for val in f1],
        textposition='outside',
        hovertemplate='<b>%{text}</b><br>Classificador: %{customdata}<extra></extra>',
        customdata=models,
        width=width
    ))
    
    # F1 CV (Mean) com barras de erro
    fig.add_trace(go.Bar(
        x=x_positions + width,
        y=f1_cv_mean,
        name='F1 CV (Mean)',
        marker_color=colors[2],
        error_y=dict(type='data', array=f1_cv_std, visible=True),
        text=[f'{val:.4f}' for val in f1_cv_mean],
        textposition='outside',
        hovertemplate='<b>%{text}</b><br>Classificador: %{customdata}<br>Std: %{error_y.array:.4f}<extra></extra>',
        customdata=models,
        width=width
    ))
    
    fig.update_xaxes(
        tickvals=x_positions,
        ticktext=models,
        title_text="Classificadores",
        tickangle=-45
    )
    fig.update_yaxes(title_text="Score", range=[0, 1.05])
    
    fig.update_layout(
        title='Comparação de Métricas por Classificador',
        barmode='group',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Cormorant Garamond, 'Times New Roman', Times, serif", size=16),  # Aumentado ~30%
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=120)
    )
    
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
    # Salvar visualização
    st.session_state['visualizations']['classification_comparison'] = fig

