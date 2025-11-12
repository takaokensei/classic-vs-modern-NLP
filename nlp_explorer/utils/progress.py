"""
Utilitários para barras de progresso no Streamlit.
Integra tqdm com Streamlit para mostrar progresso de operações longas.
"""

import streamlit as st
from typing import Optional, Callable
import time


class StreamlitProgress:
    """Wrapper para tqdm que atualiza o Streamlit."""
    
    def __init__(self, total: int, desc: str = "", unit: str = "it", 
                 position: Optional[int] = None, leave: bool = False):
        """
        Inicializa barra de progresso do Streamlit.
        
        Args:
            total: Número total de iterações
            desc: Descrição da operação
            unit: Unidade (it, epoch, etc.)
            position: Posição da barra (não usado no Streamlit)
            leave: Se deve deixar a barra após completar
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.leave = leave
        self.progress_bar = None
        self.status_text = None
        self.start_time = None
        
    def __enter__(self):
        """Inicia a barra de progresso."""
        self.start_time = time.time()
        if self.total > 0:
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
            self.update_status(0)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza a barra de progresso."""
        if self.progress_bar is not None:
            if self.leave:
                self.update_status(self.total)
            else:
                self.progress_bar.empty()
                if self.status_text:
                    self.status_text.empty()
    
    def update_status(self, n: int):
        """Atualiza o status da barra de progresso."""
        self.current = n
        if self.progress_bar is not None and self.total > 0:
            progress = min(n / self.total, 1.0)
            self.progress_bar.progress(progress)
            
            # Calcular tempo decorrido e estimativa
            elapsed = time.time() - self.start_time if self.start_time else 0
            if n > 0 and elapsed > 0:
                rate = n / elapsed
                remaining = (self.total - n) / rate if rate > 0 else 0
                elapsed_str = f"{elapsed:.1f}s"
                remaining_str = f"{remaining:.1f}s" if remaining > 0 else "calculando..."
                self.status_text.text(
                    f"{self.desc}: {n}/{self.total} {self.unit} "
                    f"({elapsed_str} decorridos, ~{remaining_str} restantes)"
                )
            else:
                self.status_text.text(f"{self.desc}: {n}/{self.total} {self.unit}")
    
    def update(self, n: int = 1):
        """Atualiza a barra de progresso."""
        self.update_status(self.current + n)
    
    def set_description(self, desc: str):
        """Atualiza a descrição."""
        self.desc = desc
        self.update_status(self.current)


def progress_bar(total: int, desc: str = "", unit: str = "it", leave: bool = False):
    """
    Cria uma barra de progresso compatível com tqdm para Streamlit.
    
    Args:
        total: Número total de iterações
        desc: Descrição da operação
        unit: Unidade (it, epoch, etc.)
        leave: Se deve deixar a barra após completar
    
    Returns:
        Context manager para usar com 'with'
    
    Example:
        with progress_bar(10, desc="Treinando modelos"):
            for i in range(10):
                # fazer algo
                progress_bar.update(1)
    """
    return StreamlitProgress(total, desc, unit, leave=leave)

