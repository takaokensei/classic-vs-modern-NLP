"""
NLP Explorer - Aplicação Streamlit para Exploração de NLP Clássico vs Moderno

Aplicação interativa para comparar métodos clássicos (TF-IDF) e modernos (Embeddings)
para classificação e clustering de textos.
"""

import streamlit as st
import sys
import os

# Configurar kaleido otimizado para exportação (se disponível)
try:
    import kaleido
    # Usar Chrome do sistema para melhor performance
    os.environ['KALEIDO_BROWSER_EXECUTABLE'] = ''
except ImportError:
    pass

# Adicionar diretórios ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.session_state import init_session_state
from utils.config import setup_page_config
from utils.icons import icon_text

# Configurar página
setup_page_config()

# Inicializar session state
init_session_state()

# Sidebar - Navegação
st.sidebar.markdown(
    f'<h1 style="display: inline-flex; align-items: center; gap: 10px;">{icon_text("brain", "NLP Explorer", size=28)}</h1>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Opções de navegação (sem HTML, pois st.radio não renderiza HTML)
page_labels = [
    "🏠 Início",
    "📊 Upload de Dados",
    "🔍 Classificação",
    "🎯 Clustering",
    "📈 Resultados & Exportação"
]

page = st.sidebar.radio(
    "Navegar",
    page_labels,
    key="navigation"
)

# Mapear de volta para labels sem emoji para importação de páginas
page_map = {
    "🏠 Início": "Início",
    "📊 Upload de Dados": "Upload de Dados",
    "🔍 Classificação": "Classificação",
    "🎯 Clustering": "Clustering",
    "📈 Resultados & Exportação": "Resultados & Exportação"
}
page = page_map.get(page, page.replace("🏠 ", "").replace("📊 ", "").replace("🔍 ", "").replace("🎯 ", "").replace("📈 ", ""))

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<h3 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("info", "Sobre", size=18)}</h3>',
    unsafe_allow_html=True
)
st.sidebar.markdown("""
Esta aplicação permite explorar e comparar:
- **TF-IDF** (Método Clássico)
- **Embeddings** (Método Moderno)
  - Sentence Transformers (local)
  - Google Gemini (via API)

Para tarefas de classificação e clustering de textos.
""")

# Importar páginas
if page == "Início":
    from pages.home import render_home
    render_home()
elif page == "Upload de Dados":
    from pages.data_upload import render_data_upload
    render_data_upload()
elif page == "Classificação":
    from pages.classification import render_classification
    render_classification()
elif page == "Clustering":
    from pages.clustering import render_clustering
    render_clustering()
elif page == "Resultados & Exportação":
    from pages.results import render_results
    render_results()

