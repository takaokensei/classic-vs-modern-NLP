"""
Página de resultados e exportação.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from io import BytesIO
import base64
import sys
import os

# Configurar kaleido para ser mais rápido (se disponível)
try:
    import kaleido
    # Configurar variáveis de ambiente para otimizar kaleido
    os.environ['KALEIDO_BROWSER_EXECUTABLE'] = ''  # Usar Chrome padrão do sistema
except ImportError:
    pass  # kaleido não está instalado, tudo bem

# Adicionar paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm_analysis import (
    explain_results_with_llm,
    check_llm_availability,
    get_api_keys
)


def render_results():
    """Renderiza a página de resultados e exportação."""
    from utils.icons import icon_text
    
    st.markdown(
        f'<h1 style="display: inline-flex; align-items: center; gap: 10px;">{icon_text("chart", "Resultados & Exportação", size=32)}</h1>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Verificar se há resultados para exportar
    has_classification = st.session_state.get('classification_results') is not None
    has_clustering = st.session_state.get('clustering_results') is not None
    has_visualizations = bool(st.session_state.get('visualizations', {}))
    
    if not (has_classification or has_clustering):
        st.info("Execute classificação ou clustering primeiro para ver resultados e exportações disponíveis.")
        return
    
    # Explicação via LLM (se disponível)
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("lightbulb", "Explicação dos Resultados com LLM", size=24)}</h2>',
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
                key="groq_key_input_results"
            )
        
        with col2:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get('api_keys', {}).get('openai', ''),
                type="password",
                help="Chave da API OpenAI",
                key="openai_key_input_results"
            )
        
        with col3:
            gemini_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.get('api_keys', {}).get('gemini', ''),
                type="password",
                help="Chave da API Google Gemini",
                key="gemini_key_input_results"
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
    
    if available_providers:
        provider = st.selectbox(
            "Escolha o provedor de LLM:",
            available_providers,
            help="Groq é recomendado por ser rápido e gratuito"
        )
        
        if st.button("Gerar Explicação Detalhada", type="primary"):
            with st.spinner("Gerando explicação detalhada com LLM..."):
                try:
                    classification_results = st.session_state.get('classification_results')
                    clustering_results = st.session_state.get('clustering_results')
                    
                    # Obter chave de API do session state
                    api_key = session_keys.get(provider) if session_keys else None
                    
                    explanation = explain_results_with_llm(
                        classification_results=classification_results,
                        clustering_results=clustering_results,
                        provider=provider,
                        api_key=api_key
                    )
                    
                    st.markdown("### Explicação Gerada:")
                    st.markdown(explanation)
                    
                    # Salvar no session state
                    st.session_state['llm_explanation'] = explanation
                    
                except Exception as e:
                    st.error(f"Erro ao gerar explicação: {str(e)}")
                    import traceback
                    with st.expander("Detalhes do erro"):
                        st.code(traceback.format_exc())
        else:
            # Mostrar explicação salva se disponível
            if st.session_state.get('llm_explanation'):
                st.markdown("### Explicação Anterior:")
                st.markdown(st.session_state['llm_explanation'])
    else:
        st.warning("Nenhuma API de LLM configurada!")
        st.info("""
        Configure pelo menos uma chave de API no painel acima.
        
        **Onde obter chaves:**
        - **Groq**: https://console.groq.com/ (gratuito e recomendado)
        - **OpenAI**: https://platform.openai.com/api-keys
        - **Gemini**: https://ai.google.dev/
        """)
        
        # Mostrar explicação básica
        try:
            explanation = generate_llm_explanation()
            st.markdown("### Explicação Básica (sem LLM):")
            st.markdown(explanation)
        except Exception as e:
            st.warning(f"Não foi possível gerar explicação básica: {str(e)}")
    
    st.markdown("---")
    
    # Exportação de resultados
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("save", "Exportar Resultados", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    # CSV Export
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f'<h3 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("bar-chart", "Dados CSV", size=20)}</h3>',
            unsafe_allow_html=True
        )
        
        if has_classification:
            if st.button("Exportar Métricas de Classificação (CSV)"):
                export_classification_csv()
        
        if has_clustering:
            if st.button("Exportar Métricas de Clustering (CSV)"):
                export_clustering_csv()
    
    with col2:
        st.markdown(
            f'<h3 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("image", "Gráficos PNG", size=20)}</h3>',
            unsafe_allow_html=True
        )
        
        if has_visualizations:
            visualizations = st.session_state.get('visualizations', {})
            
            # Mostrar aviso sobre visualizações interativas
            st.info("**Dica:** Gráficos Plotly são interativos! Use a ferramenta de screenshot do navegador (F12) para capturar.")
            
            # Criar botões sob demanda para evitar carregar tudo de uma vez
            for idx, viz_name in enumerate(visualizations.keys()):
                with st.expander(f"Exportar {viz_name.replace('_', ' ').title()}", expanded=False):
                    export_visualization(viz_name, key=f"download_{idx}")
    
    # Resumo dos resultados
    st.markdown("---")
    st.markdown(
        f'<h2 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("clipboard", "Resumo dos Resultados", size=24)}</h2>',
        unsafe_allow_html=True
    )
    
    if has_classification:
        show_classification_summary()
    
    if has_clustering:
        show_clustering_summary()


def generate_llm_explanation():
    """Gera explicação dos resultados usando LLM."""
    # Por enquanto, retorna uma explicação básica
    # Em produção, você pode integrar com OpenAI, Gemini, etc.
    
    explanation = "### Resumo dos Resultados\n\n"
    
    if st.session_state.get('classification_results'):
        explanation += "**Classificação:**\n"
        results = st.session_state['classification_results']['results']
        best_model = max(results.items(), key=lambda x: x[1]['f1_macro'])
        explanation += f"- Melhor modelo: {best_model[0]} (F1: {best_model[1]['f1_macro']:.4f})\n"
        explanation += f"- Total de modelos testados: {len(results)}\n\n"
    
    if st.session_state.get('clustering_results'):
        explanation += "**Clustering:**\n"
        results = st.session_state['clustering_results']
        for name, result in results.items():
            metrics = result['metrics']
            if 'silhouette' in metrics:
                explanation += f"- {name}: Silhouette Score = {metrics['silhouette']:.4f}\n"
        explanation += "\n"
    
    explanation += "\n**Nota:** Para explicações mais detalhadas, integre com uma API de LLM (OpenAI, Gemini, etc.)"
    
    return explanation


def export_classification_csv():
    """Exporta métricas de classificação para CSV."""
    results = st.session_state.get('classification_results')
    if results is None:
        st.error("Nenhum resultado de classificação disponível.")
        return
    
    # Criar DataFrame com métricas
    results_dict = results['results']
    cv_results = results['cv_results']
    
    df = pd.DataFrame({
        'Modelo': list(results_dict.keys()),
        'Accuracy': [r['accuracy'] for r in results_dict.values()],
        'F1_Macro': [r['f1_macro'] for r in results_dict.values()],
        'F1_CV_Mean': [cv['mean'] for cv in cv_results.values()],
        'F1_CV_Std': [cv['std'] for cv in cv_results.values()]
    })
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="classification_results.csv",
        mime="text/csv"
    )


def export_clustering_csv():
    """Exporta métricas de clustering para CSV."""
    results = st.session_state.get('clustering_results')
    if results is None:
        st.error("Nenhum resultado de clustering disponível.")
        return
    
    # Criar DataFrame com métricas
    metrics_data = []
    for name, result in results.items():
        metrics = result['metrics']
        if 'error' not in metrics:
            metrics_data.append({
                'Algoritmo': name,
                'Silhouette_Score': metrics.get('silhouette', -1),
                'Davies_Bouldin': metrics.get('davies_bouldin', float('inf')),
                'Num_Clusters': metrics.get('n_clusters', 0),
                'Num_Noise': metrics.get('n_noise', 0)
            })
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="clustering_results.csv",
            mime="text/csv"
        )


def export_visualization(viz_name, key=None):
    """Exporta uma visualização para PNG."""
    import plotly.graph_objects as go
    
    visualizations = st.session_state.get('visualizations', {})
    
    if viz_name not in visualizations:
        st.error(f"Visualização '{viz_name}' não encontrada.")
        return
    
    fig = visualizations[viz_name]
    
    # Verificar se é Plotly Figure
    if isinstance(fig, go.Figure):
        # Para Plotly: tentar usar to_image
        try:
            # Reduzir escala e dimensões para acelerar exportação
            img_bytes = fig.to_image(format='png', width=1200, height=800, scale=1)
            st.download_button(
                label=f"Download {viz_name.replace('_', ' ').title()} (PNG)",
                data=img_bytes,
                file_name=f"{viz_name}.png",
                mime="image/png",
                key=key
            )
        except Exception as e:
            # Se falhar, mostrar mensagem amigável
            st.error(f"Exportação PNG indisponível: {str(e)[:100]}")
            st.info("Gráficos Plotly são interativos! Use a ferramenta de screenshot do navegador para capturar.")
            st.info("Para habilitar exportação PNG, certifique-se que `kaleido` está instalado corretamente.")
    else:
        # Para matplotlib: usar savefig
        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label=f"Download {viz_name.replace('_', ' ').title()} (PNG)",
                data=buf,
                file_name=f"{viz_name}.png",
                mime="image/png",
                key=key
            )
        except Exception as e:
            st.error(f"Erro ao exportar visualização: {str(e)}")


def show_classification_summary():
    """Mostra resumo dos resultados de classificação."""
    from utils.icons import icon_text
    st.markdown(
        f'<h3 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("search", "Classificação", size=20)}</h3>',
        unsafe_allow_html=True
    )
    
    results = st.session_state.get('classification_results')
    if results:
        results_dict = results['results']
        best_model = max(results_dict.items(), key=lambda x: x[1]['f1_macro'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Melhor Modelo", best_model[0])
        with col2:
            st.metric("Melhor F1 Macro", f"{best_model[1]['f1_macro']:.4f}")
        with col3:
            st.metric("Melhor Accuracy", f"{best_model[1]['accuracy']:.4f}")


def show_clustering_summary():
    """Mostra resumo dos resultados de clustering."""
    from utils.icons import icon_text
    st.markdown(
        f'<h3 style="display: inline-flex; align-items: center; gap: 8px;">{icon_text("target", "Clustering", size=20)}</h3>',
        unsafe_allow_html=True
    )
    
    results = st.session_state.get('clustering_results')
    if results:
        best_silhouette = -1
        best_algorithm = None
        
        for name, result in results.items():
            metrics = result['metrics']
            if 'silhouette' in metrics and metrics['silhouette'] > best_silhouette:
                best_silhouette = metrics['silhouette']
                best_algorithm = name
        
        if best_algorithm:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Melhor Algoritmo", best_algorithm)
            with col2:
                st.metric("Melhor Silhouette", f"{best_silhouette:.4f}")
            with col3:
                total_clusters = sum(r['metrics'].get('n_clusters', 0) for r in results.values())
                st.metric("Total de Clusters", total_clusters)

