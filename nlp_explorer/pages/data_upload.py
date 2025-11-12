"""
Página de upload e pré-processamento de dados.
"""

import streamlit as st
import pandas as pd
import pickle
import os
from pathlib import Path

# Importar módulos do projeto
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing import preprocess_batch
from utils.icons import icon_text


def render_data_upload():
    """Renderiza a página de upload de dados."""
    st.markdown(
        f'<h1 style="display: inline-flex; align-items: center; gap: 10px;">{icon_text("upload", "Upload de Dados", size=32)}</h1>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Opções de carregamento
    load_option = st.radio(
        "Escolha como carregar os dados:",
        ["📁 Upload de arquivo", "📦 Dataset pré-processado (20 Newsgroups)"],
        key="load_option"
    )
    
    if load_option == "📁 Upload de arquivo":
        st.markdown(
            f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("upload", "Upload de Dataset", size=24)}</h2>',
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Faça upload do seu dataset",
            type=['csv', 'json', 'pkl'],
            help="Formatos suportados: CSV, JSON, PKL"
        )
        
        if uploaded_file is not None:
            try:
                # Detectar formato
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Arquivo CSV carregado! Shape: {df.shape}")
                    
                    # Selecionar colunas
                    if len(df.columns) > 0:
                        text_col = st.selectbox(
                            "Selecione a coluna de texto:",
                            df.columns.tolist(),
                            key="text_col"
                        )
                        
                        label_col = st.selectbox(
                            "Selecione a coluna de rótulos (opcional):",
                            ["Nenhuma"] + df.columns.tolist(),
                            key="label_col"
                        )
                        
                        if st.button("Processar Dados"):
                            process_custom_data(df, text_col, label_col if label_col != "Nenhuma" else None)
                
                elif file_extension == 'json':
                    import json
                    data = json.load(uploaded_file)
                    st.info("ℹ️ Formato JSON detectado. Implementação em desenvolvimento.")
                
                elif file_extension == 'pkl':
                    data = pickle.load(uploaded_file)
                    st.success("✅ Arquivo PKL carregado!")
                    if isinstance(data, dict) and 'text' in data:
                        process_preprocessed_data(data)
                    else:
                        st.error("❌ Formato PKL não reconhecido. Esperado: dicionário com 'text' e 'target'.")
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
    
    else:
        # Dataset pré-processado
        st.markdown(
            f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("database", "Dataset 20 Newsgroups", size=24)}</h2>',
            unsafe_allow_html=True
        )
        st.info("""
        Usando o dataset pré-processado do 20 Newsgroups com 6 classes:
        - rec.autos
        - rec.sport.baseball
        - rec.sport.hockey
        - sci.space
        - talk.politics.guns
        - talk.politics.mideast
        """)
        
        data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "20news_preprocessed.pkl"
        
        if data_path.exists():
            if st.button("Carregar Dataset Pré-processado"):
                try:
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    process_preprocessed_data(data)
                except Exception as e:
                    st.error(f"❌ Erro ao carregar dataset: {str(e)}")
        else:
            st.warning(f"⚠️ Dataset não encontrado em: {data_path}")
            st.info("ℹ️ Execute o notebook 01_preprocessing.ipynb primeiro para gerar o dataset.")
    
    # Mostrar dados carregados
    if st.session_state.get('data_loaded', False):
        st.markdown("---")
        st.markdown(
            f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("check", "Dados Carregados", size=24)}</h2>',
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Documentos", len(st.session_state.get('texts', [])))
        with col2:
            st.metric("Classes", len(st.session_state.get('target_names', [])))
        with col3:
            if st.session_state.get('labels') is not None:
                st.metric("Documentos com Rótulos", 
                         sum(st.session_state['labels'] is not None for _ in st.session_state['texts']))
        
        # Mostrar amostra (usando markdown para evitar pyarrow)
        if st.checkbox("Mostrar amostra dos dados"):
            # Preparar dados
            texts_sample = [text[:100] + "..." if len(text) > 100 else text 
                           for text in st.session_state['texts'][:10]]
            labels_sample = [st.session_state.get('target_names', [])[label] 
                            if st.session_state.get('labels') is not None and label is not None and label < len(st.session_state.get('target_names', []))
                            else "N/A" 
                            for label in (st.session_state.get('labels', [])[:10] if st.session_state.get('labels') is not None else [None]*10)]
            
            # Criar tabela em markdown
            st.markdown("**Amostra dos Dados (primeiros 10 documentos):**")
            markdown_table = "| # | Texto | Rótulo |\n"
            markdown_table += "|---|-------|--------|\n"
            
            for i, (text, label) in enumerate(zip(texts_sample, labels_sample), 1):
                # Escapar pipes no texto para markdown
                text_escaped = text.replace('|', '\\|')
                markdown_table += f"| {i} | {text_escaped} | {label} |\n"
            
            st.markdown(markdown_table)


def process_custom_data(df, text_col, label_col=None):
    """Processa dados customizados."""
    with st.spinner("Processando dados..."):
        # Extrair textos
        texts = df[text_col].astype(str).tolist()
        
        # Pré-processar
        texts_processed = preprocess_batch(texts)
        
        # Remover vazios
        valid_indices = [i for i, text in enumerate(texts_processed) if len(text.strip()) > 0]
        texts_processed = [texts_processed[i] for i in valid_indices]
        
        # Extrair rótulos se disponível
        labels = None
        target_names = None
        if label_col:
            labels_raw = df[label_col].iloc[valid_indices].tolist()
            unique_labels = sorted(set(labels_raw))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [label_to_idx[label] for label in labels_raw]
            target_names = unique_labels
        
        # Salvar no session state
        st.session_state['data_loaded'] = True
        st.session_state['texts'] = texts_processed
        st.session_state['labels'] = labels
        st.session_state['target_names'] = target_names
        st.session_state['df'] = pd.DataFrame({
            'text': texts_processed,
            'label': labels if labels else [None] * len(texts_processed)
        })
        
        st.success(f"✅ {len(texts_processed)} documentos processados com sucesso!")
        
        if labels:
            st.info(f"📋 {len(target_names)} classes encontradas: {', '.join(target_names)}")


def process_preprocessed_data(data):
    """Processa dados pré-processados e verifica cache."""
    from utils.cache import is_20news_6classes_dataset, save_metadata
    
    with st.spinner("Carregando dados..."):
        texts = data['text']
        labels = data.get('target')
        target_names = data.get('target_names', [])
        
        # Verificar se é o dataset 20news com 6 classes
        is_20news = is_20news_6classes_dataset(texts, labels, target_names)
        
        if is_20news:
            # Salvar metadados para cache
            metadata = {
                'dataset': '20news_6classes',
                'n_texts': len(texts),
                'n_classes': len(target_names),
                'target_names': target_names,
                'classes': dict(zip(range(len(target_names)), target_names))
            }
            save_metadata(metadata)
            st.session_state['is_20news_6classes'] = True
        else:
            st.session_state['is_20news_6classes'] = False
        
        # Salvar no session state
        st.session_state['data_loaded'] = True
        st.session_state['texts'] = texts
        st.session_state['labels'] = labels
        st.session_state['target_names'] = target_names
        st.session_state['df'] = pd.DataFrame({
            'text': texts,
            'label': labels if labels is not None else [None] * len(texts)
        })
        
        st.success(f"✅ {len(texts)} documentos carregados com sucesso!")
        
        if labels is not None and target_names:
            st.info(f"📋 {len(target_names)} classes: {', '.join(target_names)}")
            if is_20news:
                st.info("💾 Cache disponível para este dataset - resultados serão carregados automaticamente se já processados!")

