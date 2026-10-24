"""
extraction.py
=============
Módulo responsável pela EXTRAÇÃO de dados brutos.

Responsabilidade: baixar arquivos Excel da SSP-SP e consolidar
todas as abas em um único DataFrame.
"""

import os
import pandas as pd
import requests
import time
import urllib3

# Suprime avisos de InsecureRequestWarning (devido ao verify=False para sites do governo)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extrair_e_consolidar_dados(
    lista_de_links: list, 
    pasta_downloads: str = 'downloads',
    tentativas: int = 5
) -> pd.DataFrame | None:
    """
    Recebe uma lista de URLs de arquivos Excel, baixa todos (com retentativas),
    lê todas as abas e consolida em um único DataFrame.

    Args:
        lista_de_links: Lista de URLs diretas para os arquivos .xlsx da SSP-SP.
        pasta_downloads: Caminho da pasta onde os arquivos serão salvos.
        tentativas: Número de vezes que tentará baixar cada arquivo em caso de erro.

    Returns:
        DataFrame consolidado ou None se nenhuma planilha for lida.
    """
    print(f"Iniciando download de {len(lista_de_links)} arquivos.")

    # 1. Cria a pasta de downloads se não existir
    if not os.path.exists(pasta_downloads):
        os.makedirs(pasta_downloads)

    # 2. Baixa cada arquivo da lista
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url_arquivo in lista_de_links:
        nome_arquivo = url_arquivo.split('/')[-1]
        caminho_arquivo = os.path.join(pasta_downloads, nome_arquivo)

        # Lógica de Retentativa
        sucesso = False
        for i in range(tentativas):
            print(f"Baixando '{nome_arquivo}' (Tentativa {i+1}/{tentativas})...")
            try:
                with requests.get(url_arquivo, stream=True, headers=headers, timeout=60, verify=False) as r:
                    r.raise_for_status()
                    with open(caminho_arquivo, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                print(f" -> '{nome_arquivo}' baixado com sucesso.")
                sucesso = True
                break # Sai do loop de tentativas se deu certo

            except Exception as e:
                print(f" !! Erro na tentativa {i+1}: {e}")
                if os.path.exists(caminho_arquivo):
                    os.remove(caminho_arquivo)
                
                if i < tentativas - 1:
                    wait = (i + 1) * 3 # Espera aumenta a cada tentativa (3s, 6s, 9s...)
                    print(f"    Aguardando {wait}s para tentar novamente...")
                    time.sleep(wait)
        
        if not sucesso:
            print(f" ERROR: Falha definitiva ao baixar '{nome_arquivo}' após {tentativas} tentativas.")

    print("\nTodos os arquivos foram baixados com sucesso.")

    # 3. Lê e consolida todos os arquivos .xlsx baixados
    lista_dfs = []

    # Filtros prévios de memória (mantém apenas DDM Sorocaba e Votorantim)
    # Isso evita que o Python tente guardar gigabytes de dados irrelevantes na RAM.
    DELEGACIAS = ['DDM SOROCABA', 'DDM VOTORANTIM']
    MUNICIPIOS = ['SOROCABA', 'VOTORANTIM']

    arquivos_excel = [f for f in os.listdir(pasta_downloads) if f.endswith('.xlsx')]
    
    for arquivo in arquivos_excel:
        caminho_completo = os.path.join(pasta_downloads, arquivo)
        print(f"\n📖 Processando arquivo: {arquivo}")
        
        try:
            # Usa ExcelFile para carregar apenas a estrutura do arquivo (rápido)
            excel = pd.ExcelFile(caminho_completo)
            abas = excel.sheet_names
            total_abas = len(abas)
            print(f" -> Encontradas {total_abas} abas para leitura.")

            for i, nome_aba in enumerate(abas):
                # Feedback de progresso para o usuário
                print(f"    [{i+1}/{total_abas}] Lendo aba: '{nome_aba}'...", end="\r", flush=True)
                
                # Lê apenas a aba atual
                df_aba = pd.read_excel(excel, sheet_name=nome_aba)
                
                if df_aba is not None and not df_aba.empty:
                    # Verifica se as colunas necessárias existem para filtrar
                    if 'NOME_DELEGACIA' in df_aba.columns and 'NOME_MUNICIPIO' in df_aba.columns:
                        mask = (
                            df_aba['NOME_DELEGACIA'].astype(str).str.upper().isin(DELEGACIAS) &
                            df_aba['NOME_MUNICIPIO'].astype(str).str.upper().isin(MUNICIPIOS)
                        )
                        df_aba_filtrada = df_aba[mask].copy()
                        
                        if not df_aba_filtrada.empty:
                            lista_dfs.append(df_aba_filtrada)
                            # Imprime apenas quando encontra dados relevantes
                            print(f"    [{i+1}/{total_abas}] Encontrados {len(df_aba_filtrada)} registros em '{nome_aba}'.")
            
            print(f" -> Concluído processamento de '{arquivo}'.")

        except Exception as e:
            print(f" !! Erro ao ler abas de '{arquivo}': {e}")

    if not lista_dfs:
        print("\nNenhuma aba válida com os filtros (Sorocaba/Votorantim) foi encontrada nas planilhas.")
        return None

    # Consolida apenas os dados filtrados (muito mais leve)
    df_consolidado = pd.concat(lista_dfs, ignore_index=True)
    print(f"\n{'-'*30}")
    print(f"DADOS CONSOLIDADOS: {len(df_consolidado)} registros filtrados.")
    print(f"{'-'*30}")
    
    return df_consolidado
