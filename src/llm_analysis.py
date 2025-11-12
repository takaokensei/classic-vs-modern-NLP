"""
Módulo de análise por LLM.

Funções para usar LLMs (Groq, OpenAI, Gemini) para:
- Naming de clusters (gerar rótulos e descrições)
- Sumarização orientada a tarefa (resumo por cluster)
- Explicação de resultados
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import time


def get_api_keys(session_keys: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """
    Obtém chaves de API de variáveis de ambiente ou session state.
    
    Args:
        session_keys: Dicionário com chaves do session state (prioridade sobre env vars)
    
    Returns:
        Dicionário com chaves disponíveis
    """
    keys_from_env = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'groq': os.getenv('GROQ_API_KEY'),
        'gemini': os.getenv('GEMINI_API_KEY')
    }
    
    # Se session_keys fornecido, usar essas chaves (prioridade)
    if session_keys:
        return {
            'openai': session_keys.get('openai') or keys_from_env['openai'],
            'groq': session_keys.get('groq') or keys_from_env['groq'],
            'gemini': session_keys.get('gemini') or keys_from_env['gemini']
        }
    
    return keys_from_env


def check_llm_availability(session_keys: Optional[Dict[str, str]] = None) -> Dict[str, bool]:
    """
    Verifica quais APIs de LLM estão disponíveis.
    
    Args:
        session_keys: Dicionário com chaves do session state
    
    Returns:
        Dicionário indicando quais APIs estão disponíveis
    """
    keys = get_api_keys(session_keys)
    availability = {}
    
    # Verificar OpenAI
    try:
        import openai
        availability['openai'] = keys['openai'] is not None and len(keys['openai'].strip()) > 0 if keys['openai'] else False
    except ImportError:
        availability['openai'] = False
    
    # Verificar Groq
    try:
        from groq import Groq
        availability['groq'] = keys['groq'] is not None and len(keys['groq'].strip()) > 0 if keys['groq'] else False
    except ImportError:
        availability['groq'] = False
    
    # Verificar Gemini
    try:
        import google.generativeai as genai
        availability['gemini'] = keys['gemini'] is not None and len(keys['gemini'].strip()) > 0 if keys['gemini'] else False
    except ImportError:
        availability['gemini'] = False
    
    return availability


def name_cluster_with_llm(
    texts: List[str],
    cluster_id: int,
    top_terms: List[str],
    provider: str = 'groq',
    model: str = 'llama-3.1-70b-versatile',
    api_key: Optional[str] = None,
    stream_callback: Optional[callable] = None,
    include_summary: bool = False
) -> Tuple[str, str, Optional[str]]:
    """
    Gera nome, descrição e (opcionalmente) sumário para um cluster usando LLM em uma única chamada.
    Otimizado para reduzir número de chamadas à API.
    
    Args:
        texts: Lista de textos no cluster
        cluster_id: ID do cluster
        top_terms: Top termos mais frequentes no cluster (TF-IDF)
        provider: Provedor LLM ('groq', 'openai', 'gemini')
        model: Nome do modelo a usar
        include_summary: Se True, também gera sumário (usa mais tokens)
    
    Returns:
        Tupla (nome_cluster, descricao_cluster, sumario_cluster) - sumario é None se include_summary=False
    """
    # Amostrar textos para o prompt (máximo 5 para reduzir tokens)
    sample_texts = texts[:5] if len(texts) > 5 else texts
    # Usar apenas 1-2 textos completos e truncar os demais
    sample_text = '\n'.join([
        f"Exemplo {i+1}: {text[:300]}..." if len(text) > 300 else f"Exemplo {i+1}: {text}"
        for i, text in enumerate(sample_texts[:3])  # Máximo 3 exemplos
    ])
    
    # Prompt otimizado e mais curto
    prompt = f"""Analise este cluster de textos usando os principais termos TF-IDF.

PRINCIPAIS TERMOS: {', '.join(top_terms[:15]) if top_terms else 'N/A'}

EXEMPLOS DE TEXTOS:
{sample_text}

Instruções (responda APENAS no formato abaixo):
1. NOME: um nome curto (1-4 palavras, português)
2. DESCRIÇÃO: uma descrição breve (1-2 frases, português)"""
    
    if include_summary:
        prompt += "\n3. SUMÁRIO: um resumo de 2-4 frases do tópico principal (português)"
    
    prompt += "\n\nFormato de resposta:\nNOME: <nome>\nDESCRIÇÃO: <descrição>"
    if include_summary:
        prompt += "\nSUMÁRIO: <sumário>"

    try:
        response_text = call_llm_api(
            prompt, 
            provider=provider, 
            model=model, 
            api_key=api_key,
            max_tokens=300 if not include_summary else 500,  # Reduzir tokens
            stream=True if stream_callback else False,
            stream_callback=stream_callback
        )
        
        # Parsear resposta
        lines = response_text.strip().split('\n')
        name = f"Cluster {cluster_id}"
        description = "Sem descrição disponível"
        summary = None
        
        for line in lines:
            line_upper = line.upper()
            if 'NOME:' in line_upper or 'NAME:' in line_upper:
                name = line.split(':', 1)[1].strip() if ':' in line else line.strip()
            elif 'DESCRIÇÃO:' in line_upper or 'DESCRIPTION:' in line_upper or 'DESC:' in line_upper:
                description = line.split(':', 1)[1].strip() if ':' in line else line.strip()
            elif include_summary and ('SUMÁRIO:' in line_upper or 'SUMARIO:' in line_upper or 'SUMMARY:' in line_upper):
                summary = line.split(':', 1)[1].strip() if ':' in line else line.strip()
        
        # Se não encontrou formato esperado, tentar pegar primeiras linhas
        if name == f"Cluster {cluster_id}" and len(lines) >= 2:
            name = lines[0].strip()
            description = ' '.join(lines[1:3]).strip()
            if include_summary and len(lines) >= 4:
                summary = ' '.join(lines[3:5]).strip()
        
        if include_summary:
            return name, description, summary
        else:
            return name, description, None
        
    except Exception as e:
        error_msg = f"Erro ao gerar descrição: {str(e)}"
        if include_summary:
            return f"Cluster {cluster_id}", error_msg, None
        else:
            return f"Cluster {cluster_id}", error_msg, None


def summarize_cluster_with_llm(
    texts: List[str],
    cluster_name: str,
    provider: str = 'groq',
    model: str = 'llama-3.1-70b-versatile',
    api_key: Optional[str] = None,
    stream_callback: Optional[callable] = None
) -> str:
    """
    Gera sumário de um cluster usando LLM.
    
    Args:
        texts: Lista de textos no cluster
        cluster_name: Nome do cluster
        provider: Provedor LLM
        model: Nome do modelo
    
    Returns:
        Sumário do cluster
    """
    # Amostrar textos (máximo 10 para o prompt)
    sample_texts = texts[:10]
    sample_text = '\n'.join([f"Texto {i+1}: {text[:300]}..." if len(text) > 300 else f"Texto {i+1}: {text}"
                             for i, text in enumerate(sample_texts)])
    
    prompt = f"""Analise os seguintes textos do cluster "{cluster_name}" e gere um resumo conciso do tópico/assunto principal.

TEXTOS DO CLUSTER:
{sample_text}

Gere um resumo de 2-4 frases (em português) descrevendo o tópico principal discutido nestes textos.
"""

    try:
        summary = call_llm_api(
            prompt, 
            provider=provider, 
            model=model, 
            api_key=api_key,
            stream=True if stream_callback else False,
            stream_callback=stream_callback
        )
        return summary.strip()
    except Exception as e:
        return f"Erro ao gerar sumário: {str(e)}"


def explain_results_with_llm(
    classification_results: Optional[Dict] = None,
    clustering_results: Optional[Dict] = None,
    provider: str = 'groq',
    model: str = 'llama-3.1-70b-versatile',
    api_key: Optional[str] = None
) -> str:
    """
    Gera explicação geral dos resultados usando LLM.
    
    Args:
        classification_results: Resultados de classificação (opcional)
        clustering_results: Resultados de clustering (opcional)
        provider: Provedor LLM
        model: Nome do modelo
    
    Returns:
        Explicação dos resultados
    """
    explanation_parts = []
    
    if classification_results:
        results_dict = classification_results.get('results', {})
        if results_dict:
            best_model = max(results_dict.items(), key=lambda x: x[1].get('f1_macro', 0))
            explanation_parts.append(
                f"CLASSIFICAÇÃO: Melhor modelo foi {best_model[0]} com F1-macro de {best_model[1].get('f1_macro', 0):.4f}. "
            )
    
    if clustering_results:
        for name, result in clustering_results.items():
            metrics = result.get('metrics', {})
            silhouette = metrics.get('silhouette', -1)
            n_clusters = metrics.get('n_clusters', 0)
            explanation_parts.append(
                f"CLUSTERING ({name}): {n_clusters} clusters encontrados com Silhouette Score de {silhouette:.4f}. "
            )
    
    context = ' '.join(explanation_parts) if explanation_parts else "Nenhum resultado disponível."
    
    prompt = f"""Analise os seguintes resultados de análise de NLP e gere uma explicação detalhada e didática em português.

RESULTADOS:
{context}

Gere uma explicação de 3-5 parágrafos (em português) que:
1. Interprete os resultados obtidos
2. Explique o que significam as métricas
3. Compare os métodos usados (TF-IDF vs Embeddings, diferentes algoritmos)
4. Identifique pontos fortes e limitações
5. Sugira próximos passos ou melhorias
"""

    try:
        explanation = call_llm_api(prompt, provider=provider, model=model, api_key=api_key)
        return explanation.strip()
    except Exception as e:
        return f"Erro ao gerar explicação: {str(e)}"


def call_llm_api(
    prompt: str,
    provider: str = 'groq',
    model: str = 'llama-3.1-70b-versatile',
    max_tokens: int = 500,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    stream: bool = False,
    stream_callback: Optional[callable] = None
) -> str:
    """
    Chama API de LLM.
    
    Args:
        prompt: Prompt para enviar ao LLM
        provider: Provedor ('groq', 'openai', 'gemini')
        model: Nome do modelo
        max_tokens: Máximo de tokens na resposta
        temperature: Temperatura para geração
        api_key: Chave de API (opcional, se não fornecido usa variáveis de ambiente)
    
    Returns:
        Resposta do LLM
    """
    # Usar api_key fornecida ou buscar de variáveis de ambiente
    keys = get_api_keys()
    
    # Se api_key fornecida, usar ela
    if api_key:
        keys[provider] = api_key
    
    if provider == 'groq':
        api_key_to_use = api_key or keys['groq']
        if not api_key_to_use or len(api_key_to_use.strip()) == 0:
            raise ValueError("Chave GROQ_API_KEY não encontrada. Configure no frontend ou nas variáveis de ambiente.")
        
        try:
            from groq import Groq
            client = Groq(api_key=api_key_to_use)
            
            # Modelos disponíveis no Groq
            if model not in ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768']:
                model = 'llama-3.1-70b-versatile'
            
            if stream and stream_callback:
                # Streaming para Groq
                stream_obj = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Você é um assistente especializado em análise de NLP. Responda sempre em português."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream_obj:
                    if chunk.choices and len(chunk.choices) > 0:
                        if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta:
                            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                stream_callback(content)
                return full_response if full_response else ""
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Você é um assistente especializado em análise de NLP. Responda sempre em português."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("Biblioteca 'groq' não instalada. Instale com: pip install groq")
    
    elif provider == 'openai':
        api_key_to_use = api_key or keys['openai']
        if not api_key_to_use or len(api_key_to_use.strip()) == 0:
            raise ValueError("Chave OPENAI_API_KEY não encontrada. Configure no frontend ou nas variáveis de ambiente.")
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key_to_use)
            
            response = client.chat.completions.create(
                model=model if model.startswith('gpt') else 'gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em análise de NLP. Responda sempre em português."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("Biblioteca 'openai' não instalada. Instale com: pip install openai")
    
    elif provider == 'gemini':
        api_key_to_use = api_key or keys['gemini']
        if not api_key_to_use or len(api_key_to_use.strip()) == 0:
            raise ValueError("Chave GEMINI_API_KEY não encontrada. Configure no frontend ou nas variáveis de ambiente.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key_to_use)
            
            # Detectar modelo disponível automaticamente
            try:
                # Listar modelos disponíveis da API
                all_models = list(genai.list_models())
                
                # Filtrar modelos que suportam generateContent e extrair nomes
                available_models = []
                for m in all_models:
                    if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods:
                        model_name = m.name if isinstance(m.name, str) else str(m.name)
                        # Remover prefixo 'models/' se existir
                        clean_name = model_name.replace('models/', '')
                        available_models.append(clean_name)
                
                # Ordem de preferência para tentar (modelos mais recentes primeiro)
                preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro', 'gemini-pro']
                
                # Se o modelo especificado está disponível, usar ele
                model_to_use = None
                if model.startswith('gemini'):
                    # Verificar se algum modelo disponível contém o nome especificado
                    for available in available_models:
                        if model in available or available.endswith(model):
                            model_to_use = available
                            break
                
                # Se não encontrou, tentar na ordem de preferência
                if not model_to_use:
                    for preferred in preferred_models:
                        for available in available_models:
                            if preferred in available or available.endswith(preferred):
                                model_to_use = available
                                break
                        if model_to_use:
                            break
                
                # Se ainda não encontrou, usar o primeiro disponível
                if not model_to_use and available_models:
                    model_to_use = available_models[0]
                
                # Se nenhum modelo disponível, lançar erro informativo
                if not model_to_use:
                    model_names_str = ', '.join([str(m.name) if hasattr(m, 'name') else str(m) for m in all_models[:5]])
                    raise ValueError(
                        f"Nenhum modelo Gemini disponível para generateContent. "
                        f"Modelos listados pela API: {model_names_str}"
                    )
                
                gemini_model = genai.GenerativeModel(model_to_use)
                
                # Implementar retry com backoff exponencial para rate limiting
                max_retries = 3
                base_delay = 1.0
                
                for attempt in range(max_retries):
                    try:
                        # Streaming para Gemini
                        if stream and stream_callback:
                            full_response = ""
                            for chunk in gemini_model.generate_content(
                                f"Você é um assistente especializado em análise de NLP. Responda sempre em português.\n\n{prompt}",
                                generation_config={
                                    'max_output_tokens': max_tokens,
                                    'temperature': temperature
                                },
                                stream=True
                            ):
                                try:
                                    chunk_text = chunk.text
                                    full_response += chunk_text
                                    stream_callback(chunk_text)
                                except:
                                    # Alguns chunks podem não ter texto
                                    pass
                            
                            if not full_response:
                                raise ValueError("Resposta vazia do streaming Gemini")
                            return full_response
                        else:
                            response = gemini_model.generate_content(
                                f"Você é um assistente especializado em análise de NLP. Responda sempre em português.\n\n{prompt}",
                                generation_config={
                                    'max_output_tokens': max_tokens,
                                    'temperature': temperature
                                }
                            )
                            
                            # Verificar se a resposta foi bloqueada (finish_reason 2 = SAFETY)
                            if response.candidates:
                                candidate = response.candidates[0]
                                # finish_reason 2 = SAFETY (bloqueado por filtros de segurança)
                                # finish_reason 1 = STOP (resposta completa)
                                # finish_reason 3 = MAX_TOKENS (limite de tokens atingido)
                                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                                    # Tentar com prompt menos restritivo
                                    if attempt < max_retries - 1:
                                        # Modificar prompt para evitar bloqueio
                                        prompt_modified = prompt.replace(
                                            "Você é um assistente especializado em análise de NLP.",
                                            "Você é um assistente de análise de dados textuais."
                                        )
                                        response = gemini_model.generate_content(
                                            prompt_modified,
                                            generation_config={
                                                'max_output_tokens': max_tokens,
                                                'temperature': temperature
                                            }
                                        )
                                
                                # Tentar obter texto
                                try:
                                    return response.text
                                except ValueError as e:
                                    # Se não conseguiu obter texto, verificar o motivo
                                    if 'finish_reason is 2' in str(e) or 'SAFETY' in str(e):
                                        raise ValueError(
                                            "A resposta foi bloqueada pelos filtros de segurança do Gemini. "
                                            "Isso pode acontecer com conteúdo sensível. Tente usar Groq ou OpenAI como alternativa."
                                        )
                                    raise
                        
                        # Se não há candidatos, lançar erro
                        raise ValueError("Resposta vazia da API Gemini")
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        # Verificar se é erro de quota/rate limit (429)
                        if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                            if attempt < max_retries - 1:
                                # Extrair delay sugerido se disponível
                                delay = base_delay * (2 ** attempt)  # Backoff exponencial
                                
                                # Tentar extrair delay do erro
                                if 'retry_delay' in error_str:
                                    try:
                                        import re
                                        delay_match = re.search(r'retry_delay.*?seconds.*?(\d+)', error_str)
                                        if delay_match:
                                            delay = float(delay_match.group(1)) + 1  # +1 segundo de margem
                                    except:
                                        pass
                                
                                # Delay antes de tentar novamente
                                time.sleep(delay)
                                continue
                            else:
                                raise ValueError(
                                    f"Quota/rate limit excedido no Gemini. "
                                    f"O tier gratuito permite apenas 2 requisições por minuto. "
                                    f"Recomendamos usar Groq (gratuito e sem limites rígidos) ou adicionar delays maiores entre requisições."
                                )
                        else:
                            # Outro tipo de erro, lançar diretamente
                            raise
                
            except Exception as e:
                # Se falhar, tentar listar modelos disponíveis para debug
                try:
                    models_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    available_models_str = ', '.join(models_list[:5])  # Mostrar até 5
                    raise ValueError(f"Erro ao usar modelo Gemini. Modelos disponíveis: {available_models_str}. Erro original: {str(e)}")
                except:
                    raise ValueError(f"Erro ao usar API Gemini: {str(e)}")
            
        except ImportError:
            raise ImportError("Biblioteca 'google-generativeai' não instalada. Instale com: pip install google-generativeai")
    
    else:
        raise ValueError(f"Provedor '{provider}' não suportado. Use 'groq', 'openai' ou 'gemini'")


def get_top_terms_for_cluster(
    texts: List[str],
    vectorizer,
    n_terms: int = 20
) -> List[str]:
    """
    Obtém os termos mais relevantes de um cluster usando TF-IDF.
    
    Args:
        texts: Lista de textos do cluster
        vectorizer: Vetorizador TF-IDF já treinado
        n_terms: Número de termos a retornar
    
    Returns:
        Lista de termos mais relevantes
    """
    try:
        # Vetorizar textos do cluster
        X_cluster = vectorizer.transform(texts)
        
        # Calcular média dos valores TF-IDF
        mean_tfidf = np.array(X_cluster.mean(axis=0)).flatten()
        
        # Obter índices dos top termos
        top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
        
        # Obter termos do vectorizer
        feature_names = vectorizer.get_feature_names_out()
        top_terms = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
        
        return top_terms[:n_terms]
    except Exception as e:
        return []

