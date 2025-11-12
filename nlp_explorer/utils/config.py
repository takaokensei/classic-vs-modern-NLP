"""
Configurações da aplicação Streamlit.
"""

import streamlit as st


def setup_page_config():
    """Configura a página do Streamlit."""
    st.set_page_config(
        page_title="NLP Explorer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Adicionar CSS customizado para fonte elegante e suprimir warnings
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500;1,600;1,700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cormorant Garamond', 'Times New Roman', serif;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Cormorant Garamond', 'Times New Roman', serif;
        font-weight: 600;
    }
    
    .stSelectbox, .stSlider, .stButton, .stTextInput, .stTextArea {
        font-family: 'Cormorant Garamond', 'Times New Roman', serif;
    }
    
    code, pre {
        font-family: 'Courier New', monospace;
    }
    
    /* Suprimir warnings de cores vazias na sidebar */
    [data-testid="stSidebar"] {
        --widget-border-color: #262730 !important;
        --skeleton-background-color: #262730 !important;
        --widget-background-color: #262730 !important;
    }
    </style>
    <script>
    // Suprimir warnings de cores inválidas no console
    const originalWarn = console.warn;
    console.warn = function(...args) {
        if (args[0] && typeof args[0] === 'string' && 
            (args[0].includes('Invalid color') || args[0].includes('widgetBorderColor') || 
             args[0].includes('widgetBackgroundColor') || args[0].includes('skeletonBackgroundColor'))) {
            return; // Suprimir warnings de cores
        }
        originalWarn.apply(console, args);
    };
    </script>
    """, unsafe_allow_html=True)

