"""
transformation.py
=================
Módulo responsável pela TRANSFORMAÇÃO dos dados brutos.

Responsabilidade: limpar, normalizar e enriquecer o DataFrame
consolidado, preparando-o para a carga no BigQuery.
"""

import os
import re
import pandas as pd
from rapidfuzz import process, fuzz


def normalizar_bairros(series_bairros: pd.Series) -> pd.Series:
    """
    Recebe uma Series (coluna) do pandas e aplica uma normalização
    completa para nomes de bairros, em duas etapas.

    Args:
        series_bairros: Coluna com os nomes de bairros a serem normalizados.

    Returns:
        Series com os nomes de bairros normalizados e padronizados.
    """
    # 1. Garante que é string e converte para minúsculo
    col_norm = series_bairros.astype(str).str.lower()

    # 2. Remove acentos (ex: "vila antônia" -> "vila antonia")
    col_norm = col_norm.str.normalize('NFKD') \
                       .str.encode('ascii', errors='ignore') \
                       .str.decode('utf-8')

    # 3. Remove pontuações (substitui por espaço)
    col_norm = col_norm.str.replace(r'[^a-z0-9\s]', ' ', regex=True)

    # 4. Limpa espaços duplos (importante fazer antes dos mapeamentos)
    col_norm = col_norm.str.replace(r'\s+', ' ', regex=True).str.strip()

    # 5. Mapeamento - ETAPA 1: ABREVIAÇÕES (troca pedaços de palavras)
    map_abreviacoes = {
        r'\bjd\b':       'jardim',
        r'\bjard\b':     'jardim',
        r'\bvl\b':       'vila',
        r'\bpq\b':       'parque',
        r'\bprq\b':      'parque',
        r'\bcent\b':     'centro',
        r'\bcaguassu\b': 'caguacu'
    }
    col_norm = col_norm.replace(map_abreviacoes, regex=True)

    # 6. Mapeamento - ETAPA 2: CORREÇÕES TOTAIS (troca a string inteira)
    # Usamos ^ (início) e $ (fim) para garantir correspondência exata.
    map_correcoes_totais = {
        r'^\s*b funda\s*$':                                     'barra funda',
        r'^\s*bairro barra funda\s*$':                          'barra funda',
        r'^\s*morros\s*$':                                      'bairro dos morros',
        r'^\s*bairro jacutinga\s*$':                            'jacutinga',
        r'^\s*bairro rio acima\s*$':                            'rio acima',
        r'^\s*cajuru\s*$':                                      'cajuru do sul',
        r'^\s*campolim\s*$':                                    'parque campolim',
        r'^\s*central parque sorocaba\s*$':                     'central parque',
        r'^\s*conjunto habitacional julio de mesquita\s*$':     'julio de mesquita filho',
        r'^\s*julio de mesquita\s*$':                           'julio de mesquita filho',
        r'^\s*guaiba\s*$':                                      'jardim guaiba',
        r'^\s*jardim archila\s*$':                              'jardim archilla',
        r'^\s*jardim magnolias\s*$':                            'jardim magnolia',
        r'^\s*jardim nikey\s*$':                                'jardim nikkey',
        r'^\s*jardim nikkei\s*$':                               'jardim nikkey',
        r'^\s*jatai\s*$':                                       'parque jatai',
        r'^\s*nilton torres\s*$':                               'jardim nilton torres',
        r'^\s*wanel ville\s*$':                                 'jardim wanel ville',
        r'^\s*zacarias\s*$':                                    'vila zacarias',
        r'^\s*votocel\s*$':                                     'vila votocel',
        r'^\s*barra funda\s*$':                                 'barra funda',
        r'^\s*nao informado\s*$':                               'nao informado',
    }
    col_norm = col_norm.replace(map_correcoes_totais, regex=True)

    # 7. Limpeza Final
    col_norm = col_norm.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Se a string ficar vazia após a limpeza, volta "nao informado"
    col_norm = col_norm.replace(r'^\s*$', 'nao informado', regex=True)

    return col_norm


def aplicar_fuzzy_matching_bairros(df: pd.DataFrame, min_score: int = 85) -> pd.Series:
    """
    Aplica Fuzzy Matching para alinhar os bairros (já normalizados por regras básicas)
    com as listas oficiais de Sorocaba e Votorantim.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_sorocaba = os.path.join(base_dir, 'data', 'raw', 'bairros_sorocaba.xlsx')
    caminho_votorantim = os.path.join(base_dir, 'data', 'raw', 'bairros_votorantim.xlsx')
    
    # Se os arquivos não existirem, retorna a coluna original
    if not os.path.exists(caminho_sorocaba) or not os.path.exists(caminho_votorantim):
        print("Listas oficiais de bairros não encontradas. Pulando fuzzy matching.")
        return df['BAIRRO']

    df_oficial_sorocaba = pd.read_excel(caminho_sorocaba, engine='calamine')
    df_oficial_votorantim = pd.read_excel(caminho_votorantim, engine='calamine')
    
    coluna_sorocaba = df_oficial_sorocaba.columns[0]
    coluna_votorantim = df_oficial_votorantim.columns[0]
    
    oficial_sorocaba_norm = normalizar_bairros(df_oficial_sorocaba[coluna_sorocaba]).unique()
    oficial_votorantim_norm = normalizar_bairros(df_oficial_votorantim[coluna_votorantim]).unique()
    
    bairros_finais = df['BAIRRO'].copy()
    
    for municipio, lista_oficial in [('SOROCABA', oficial_sorocaba_norm), ('VOTORANTIM', oficial_votorantim_norm)]:
        mascara_mun = df['NOME_MUNICIPIO'].str.upper() == municipio
        bairros_unicos_mun = df.loc[mascara_mun, 'BAIRRO'].unique()
        
        mapa_fuzzy = {}
        for bairro_sujo in bairros_unicos_mun:
            if pd.isna(bairro_sujo) or bairro_sujo == 'nao informado':
                mapa_fuzzy[bairro_sujo] = bairro_sujo
                continue
                
            resultado = process.extractOne(bairro_sujo, lista_oficial, scorer=fuzz.token_sort_ratio)
            if resultado and resultado[1] >= min_score:
                mapa_fuzzy[bairro_sujo] = resultado[0]
            else:
                mapa_fuzzy[bairro_sujo] = bairro_sujo
                
        bairros_finais.loc[mascara_mun] = df.loc[mascara_mun, 'BAIRRO'].map(mapa_fuzzy)
        
    return bairros_finais


def transformar_dados(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Filtra, limpa e transforma o DataFrame consolidado, preparando-o
    para a carga no Data Warehouse.

    Etapas: filtragem por delegacia/município, tratamento de datas,
    preenchimento de nulos, normalização de bairros, criação de colunas
    derivadas (mês, dia da semana), renomeação de colunas e garantia
    de tipos de dados corretos.

    Args:
        df: DataFrame consolidado gerado pela etapa de extração.

    Returns:
        DataFrame limpo e transformado, pronto para a carga, ou None
        se a entrada for inválida.
    """
    if df is None:
        return None

    # --- FILTRAGEM ---
    delegacias_desejadas = ['DDM SOROCABA', 'DDM VOTORANTIM']
    municipios_desejados = ['SOROCABA', 'VOTORANTIM']
    df_filtrado = df[
        (df['NOME_DELEGACIA'].str.upper().isin(delegacias_desejadas)) &
        (df['NOME_MUNICIPIO'].str.upper().isin(municipios_desejados))
    ].copy()
    print(f"\nDados filtrados. {len(df_filtrado)} registros encontrados.")

    # --- LIMPEZA E TRANSFORMAÇÃO ---
    df_filtrado['DATA_OCORRENCIA_BO'] = pd.to_datetime(
        df_filtrado['DATA_OCORRENCIA_BO'], format='%d/%m/%Y', errors='coerce'
    )
    df_filtrado.dropna(subset=['DATA_OCORRENCIA_BO'], inplace=True)

    colunas_para_preencher = ['BAIRRO', 'LOGRADOURO']
    for coluna in colunas_para_preencher:
        df_filtrado[coluna] = df_filtrado[coluna].fillna('Não Informado')

    # Preencher DESC_PERIODO vazio usando HORA_OCORRENCIA_BO
    def inferir_periodo(hora_str):
        if pd.isna(hora_str):
            return 'Não Informado'
        try:
            # Extrai a hora assumindo formato HH:MM ou HH
            hora = int(str(hora_str).split(':')[0])
            if 0 <= hora < 6:
                return 'De Madrugada'
            elif 6 <= hora < 12:
                return 'Pela Manhã'
            elif 12 <= hora < 18:
                return 'A Tarde'
            elif 18 <= hora <= 23:
                return 'A Noite'
        except:
            pass
        return 'Não Informado'

    mascara_periodo_vazio = df_filtrado['DESC_PERIODO'].isna() | (df_filtrado['DESC_PERIODO'].str.upper() == 'NÃO INFORMADO') | (df_filtrado['DESC_PERIODO'] == '')
    df_filtrado.loc[mascara_periodo_vazio, 'DESC_PERIODO'] = df_filtrado.loc[mascara_periodo_vazio, 'HORA_OCORRENCIA_BO'].apply(inferir_periodo)
    df_filtrado['DESC_PERIODO'] = df_filtrado['DESC_PERIODO'].fillna('Não Informado')

    print("\nIniciando normalização de bairros...")
    df_filtrado['BAIRRO'] = normalizar_bairros(df_filtrado['BAIRRO'])
    print("Normalização de bairros concluída.")

    print("\nAplicando Fuzzy Matching com base nas listas oficiais...")
    df_filtrado['BAIRRO'] = aplicar_fuzzy_matching_bairros(df_filtrado)
    print("Fuzzy Matching concluído.")

    # --- COLUNAS DERIVADAS ---
    df_filtrado['MES_OCORRENCIA'] = df_filtrado['DATA_OCORRENCIA_BO'].dt.month

    df_filtrado['DIA_SEMANA_EN'] = df_filtrado['DATA_OCORRENCIA_BO'].dt.day_name()
    mapa_dias = {
        'Monday':    'Segunda-feira',
        'Tuesday':   'Terça-feira',
        'Wednesday': 'Quarta-feira',
        'Thursday':  'Quinta-feira',
        'Friday':    'Sexta-feira',
        'Saturday':  'Sábado',
        'Sunday':    'Domingo'
    }
    df_filtrado['DIA_SEMANA'] = df_filtrado['DIA_SEMANA_EN'].map(mapa_dias)
    df_filtrado.drop(columns=['DIA_SEMANA_EN'], inplace=True)

    # --- RENOMEAÇÃO DE COLUNAS ---
    mapa_renomear = {
        'NUM_BO':             'codigo_bo',
        'NOME_MUNICIPIO':     'nome_municipio',
        'NOME_DELEGACIA':     'nome_delegacia',
        'ANO_ESTATISTICA':    'ano_ocorrencia',
        'MES_OCORRENCIA':     'mes_ocorrencia',
        'DATA_OCORRENCIA_BO': 'data_ocorrencia_bo',
        'HORA_OCORRENCIA_BO': 'hora_ocorrencia_bo',
        'DESC_PERIODO':       'periodo_ocorrencia',
        'DIA_SEMANA':         'dia_semana',
        'DESCR_SUBTIPOLOCAL': 'local_ocorrencia',
        'BAIRRO':             'bairro',
        'LOGRADOURO':         'logradouro',
        'LATITUDE':           'latitude',
        'LONGITUDE':          'longitude',
        'RUBRICA':            'artigo_ocorrencia',
        'NATUREZA_APURADA':   'tipo_ocorrencia'
    }
    df_renomeado = df_filtrado.rename(columns=mapa_renomear)

    # --- PADRONIZAÇÃO DE TEXTO ---
    colunas_texto = df_renomeado.select_dtypes(include=['object']).columns
    for coluna in colunas_texto:
        if coluna == 'hora_ocorrencia_bo':
            continue
        df_renomeado[coluna] = df_renomeado[coluna].astype(str).str.title()
        df_renomeado[coluna] = df_renomeado[coluna].str.replace('Nao Informado', 'Não Informado')
        df_renomeado[coluna] = df_renomeado[coluna].str.replace('Nan', 'Não Informado')

    # --- GARANTIA DE TIPOS ---
    print("\nGarantindo os tipos de dados corretos antes da carga...")

    for coluna in ['ano_ocorrencia', 'mes_ocorrencia']:
        if coluna in df_renomeado.columns:
            df_renomeado[coluna] = pd.to_numeric(df_renomeado[coluna], errors='coerce')
            df_renomeado.dropna(subset=[coluna], inplace=True)
            df_renomeado[coluna] = df_renomeado[coluna].astype(int)

    for coluna in ['latitude', 'longitude']:
        if coluna in df_renomeado.columns:
            df_renomeado[coluna] = df_renomeado[coluna].astype(str).str.replace(',', '.', regex=False)
            df_renomeado[coluna] = pd.to_numeric(df_renomeado[coluna], errors='coerce')

    # --- SELEÇÃO E ORDENAÇÃO FINAL DE COLUNAS ---
    ordem_final_colunas = [
        'codigo_bo', 'nome_municipio', 'nome_delegacia',
        'ano_ocorrencia', 'mes_ocorrencia', 'data_ocorrencia_bo',
        'hora_ocorrencia_bo', 'periodo_ocorrencia', 'dia_semana',
        'local_ocorrencia', 'bairro', 'logradouro',
        'latitude', 'longitude', 'artigo_ocorrencia', 'tipo_ocorrencia',
    ]
    colunas_existentes = [col for col in ordem_final_colunas if col in df_renomeado.columns]
    df_transformado = df_renomeado[colunas_existentes]

    print("Dados transformados com sucesso!")
    return df_transformado


def encontrar_valores_nao_numericos(df: pd.DataFrame, colunas: list) -> None:
    """
    Verifica uma lista de colunas em um DataFrame e imprime os valores
    que não podem ser convertidos para números. Útil para debug antes da carga.

    Args:
        df: DataFrame a ser inspecionado.
        colunas: Lista de nomes de colunas que deveriam ser numéricas.
    """
    print("\n--- INICIANDO VERIFICAÇÃO DE VALORES NÃO NUMÉRICOS ---")
    problema_encontrado = False

    for coluna in colunas:
        if coluna in df.columns:
            numerico = pd.to_numeric(df[coluna], errors='coerce')
            linhas_problematicas = df[numerico.isna() & df[coluna].notna()]

            if not linhas_problematicas.empty:
                problema_encontrado = True
                print(f"\n!! Problema na coluna '{coluna}'. Valores que não são números:")
                print(linhas_problematicas[coluna].unique())

    if not problema_encontrado:
        print("--- NENHUM VALOR NÃO NUMÉRICO ENCONTRADO NAS COLUNAS VERIFICADAS ---")
