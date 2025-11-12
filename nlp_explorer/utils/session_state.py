"""
Utilitários para gerenciar o session state do Streamlit.
"""

import streamlit as st


def init_session_state():
    """Inicializa variáveis no session state."""
    defaults = {
        'data_loaded': False,
        'df': None,
        'texts': None,
        'labels': None,
        'target_names': None,
        'vectorization_type': None,
        'vectorizer': None,
        'vectors': None,
        'embeddings_model': None,
        'classification_results': None,
        'clustering_results': None,
        'classification_models': {},
        'clustering_models': {},
        'visualizations': {},
        'google_api_key': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

