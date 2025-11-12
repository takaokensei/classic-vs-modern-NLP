"""
Página inicial do NLP Explorer.
"""

import streamlit as st
from utils.icons import icon_text


def render_home():
    """Renderiza a página inicial."""
    st.markdown(
        f'<h1 style="display: inline-flex; align-items: center; gap: 10px;">{icon_text("brain", "NLP Explorer", size=32)}</h1>',
        unsafe_allow_html=True
    )
    st.markdown("### Bem-vindo ao Explorador de NLP Clássico vs Moderno")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("book", "Sobre a Aplicação", size=24)}</h2>',
            unsafe_allow_html=True
        )
        st.markdown("""
        Esta aplicação permite explorar e comparar métodos clássicos e modernos 
        de processamento de linguagem natural (NLP):
        
        - **TF-IDF**: Método tradicional baseado em frequência de termos
        - **Embeddings**: Representações modernas usando modelos de linguagem
        
        Você pode realizar:
        - **Upload e pré-processamento** de seus próprios dados
        - **Classificação** de textos com múltiplos algoritmos
        - **Clustering** com diferentes técnicas
        - **Visualização** interativa dos resultados
        - **Exportação** de resultados e gráficos
        """)
    
    with col2:
        st.markdown(
            f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("rocket", "Começando", size=24)}</h2>',
            unsafe_allow_html=True
        )
        st.markdown("""
        **Passo 1:** Faça upload do seu dataset na página **Upload de Dados**
        
        **Passo 2:** Escolha o método de vetorização (TF-IDF ou Embeddings)
        
        **Passo 3:** Execute classificação ou clustering na página correspondente
        
        **Passo 4:** Explore os resultados e exporte os dados
        """)
    
    st.markdown("---")
    
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("clipboard", "Funcionalidades", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown(
            f'<h3 style="display: inline-flex; align-items: center; gap: 6px;">{icon_text("upload", "Upload Flexível", size=20)}</h3>',
            unsafe_allow_html=True
        )
        st.markdown("""
        - Upload via arquivo CSV/JSON
        - Dataset pré-processado 20 Newsgroups
        - Pré-processamento automático dos textos
        """)
    
    with features_col2:
        st.markdown(
            f'<h3 style="display: inline-flex; align-items: center; gap: 6px;">{icon_text("microscope", "Análise Avançada", size=20)}</h3>',
            unsafe_allow_html=True
        )
        st.markdown("""
        - Classificação com múltiplos algoritmos
        - Clustering K-Means e DBSCAN
        - Métricas de avaliação detalhadas
        - Visualizações interativas
        """)
    
    with features_col3:
        st.markdown(
            f'<h3 style="display: inline-flex; align-items: center; gap: 6px;">{icon_text("lightbulb", "Explicabilidade", size=20)}</h3>',
            unsafe_allow_html=True
        )
        st.markdown("""
        - Explicações via LLM dos resultados
        - Interpretação de clusters e classificações
        - Insights automáticos sobre os dados
        """)
    
    st.markdown("---")
    
    if st.session_state.get('data_loaded', False):
        st.success("✅ Dados carregados! Você pode prosseguir para Classificação ou Clustering.")
    else:
        st.info("ℹ️ Faça upload de dados na página Upload de Dados para começar.")

