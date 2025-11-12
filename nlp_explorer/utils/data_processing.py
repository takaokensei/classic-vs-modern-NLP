"""
Utilitários para processamento de dados.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle


def load_preprocessed_20news():
    """Carrega o dataset pré-processado do 20 Newsgroups."""
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "20news_preprocessed.pkl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def validate_data(texts, labels=None):
    """Valida os dados carregados."""
    if not texts or len(texts) == 0:
        raise ValueError("Nenhum texto encontrado nos dados.")
    
    if labels is not None:
        if len(labels) != len(texts):
            raise ValueError(f"Quantidade de rótulos ({len(labels)}) não corresponde à quantidade de textos ({len(texts)}).")
        
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError("É necessário pelo menos 2 classes para classificação.")

