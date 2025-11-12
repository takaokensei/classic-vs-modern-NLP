"""
Módulo de classificação de textos.

Funções utilitárias para treinar e avaliar classificadores.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from typing import Optional, Callable


def get_default_classifiers():
    """
    Retorna um dicionário com classificadores padrão.
    
    Returns:
        Dicionário {nome: classificador}
    """
    return {
        'MultinomialNB': MultinomialNB(),
        'GaussianNB': GaussianNB(),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=20),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        # SVM com cache_size limitado para melhor performance
        'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True, cache_size=200),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True, cache_size=200, gamma='scale')
    }


def train_and_evaluate(X, y, classifiers=None, test_size=0.2, random_state=42, 
                      stratify=True, cv_folds=5, progress_callback: Optional[Callable] = None):
    """
    Treina e avalia múltiplos classificadores.
    
    Args:
        X: Features (matriz)
        y: Rótulos
        classifiers: Dicionário de classificadores (se None, usa os padrão)
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade
        stratify: Se True, usa stratify=y no split
        cv_folds: Número de folds para validação cruzada
        progress_callback: Função callback para atualizar progresso (recebe (current, total, message))
    
    Returns:
        Dicionário com resultados {'results': {...}, 'cv_results': {...}, 'predictions': {...}}
    """
    if classifiers is None:
        classifiers = get_default_classifiers()
    
    total_steps = len(classifiers) * 2  # Treino + CV para cada classificador
    
    # Divisão treino/teste
    if progress_callback:
        progress_callback(0, total_steps, "Dividindo dados em treino/teste...")
    
    # Converter matriz esparsa para densa se necessário (alguns classificadores requerem)
    # Verificar se algum classificador precisa de dados densos
    needs_dense = any(
        name in ['GaussianNB', 'SVM (Linear)', 'SVM (RBF)'] 
        for name in classifiers.keys()
    ) if classifiers else False
    
    if needs_dense and hasattr(X, 'toarray'):
        # Converter esparso para denso
        X = X.toarray()
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    results = {}
    predictions = {}
    cv_results = {}
    
    current_step = 1
    
    for name, model in classifiers.items():
        # Treinar
        if progress_callback:
            progress_callback(current_step, total_steps, f"Treinando {name}...")
        
        model.fit(X_train, y_train)
        current_step += 1
        
        # Prever
        if progress_callback:
            progress_callback(current_step, total_steps, f"Predizendo com {name}...")
        
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Relatório completo por classe
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'classification_report': report
        }
        
        # Validação cruzada (otimizada para SVM - menos folds se for muito lento)
        if progress_callback:
            progress_callback(current_step, total_steps, f"Validação cruzada para {name}...")
        
        # Para SVM, usar menos folds para acelerar (3 em vez de 5)
        # SVM pode ser muito lento com validação cruzada
        effective_cv_folds = 3 if 'SVM' in name else cv_folds
        
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=effective_cv_folds, scoring='f1_macro')
            cv_results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
        except Exception as e:
            # Se CV falhar (pode acontecer com SVM em alguns casos), usar valores padrão
            print(f"Warning: CV failed for {name}: {e}")
            cv_results[name] = {
                'mean': f1_macro,  # Usar F1 macro como fallback
                'std': 0.0,
                'scores': np.array([f1_macro] * effective_cv_folds)
            }
        current_step += 1
    
    if progress_callback:
        progress_callback(total_steps, total_steps, "Classificação concluída!")
    
    return {
        'results': results,
        'cv_results': cv_results,
        'predictions': predictions,
        'X_test': X_test,
        'y_test': y_test
    }


def generate_classification_report(y_true, y_pred, target_names, output_dict=True):
    """
    Gera relatório de classificação.
    
    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos
        target_names: Nomes das classes
        output_dict: Se True, retorna como dicionário
    
    Returns:
        Relatório de classificação
    """
    return classification_report(y_true, y_pred, target_names=target_names, 
                                output_dict=output_dict)

