"""
Módulo de pré-processamento de textos.

Funções utilitárias para pré-processamento de textos, incluindo limpeza,
normalização e remoção de stopwords.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def preprocess_text(text, remove_stopwords=True, min_word_length=2):
    """
    Aplica pré-processamento leve em um texto.
    
    Args:
        text: Texto a ser processado
        remove_stopwords: Se True, remove stopwords em inglês
        min_word_length: Tamanho mínimo das palavras a manter
    
    Returns:
        Texto processado
    """
    # Converter para minúsculas
    text = text.lower()
    
    # Remover pontuação e dígitos (manter apenas letras e espaços)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remover múltiplos espaços
    text = re.sub(r'\s+', ' ', text)
    
    # Dividir em palavras e remover stopwords
    words = text.split()
    
    if remove_stopwords:
        words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > min_word_length]
    else:
        words = [w for w in words if len(w) > min_word_length]
    
    # Juntar novamente
    text = ' '.join(words)
    
    return text.strip()


def preprocess_batch(texts, remove_stopwords=True, min_word_length=2):
    """
    Aplica pré-processamento em uma lista de textos.
    
    Args:
        texts: Lista de textos a serem processados
        remove_stopwords: Se True, remove stopwords em inglês
        min_word_length: Tamanho mínimo das palavras a manter
    
    Returns:
        Lista de textos processados
    """
    return [preprocess_text(text, remove_stopwords, min_word_length) for text in texts]

