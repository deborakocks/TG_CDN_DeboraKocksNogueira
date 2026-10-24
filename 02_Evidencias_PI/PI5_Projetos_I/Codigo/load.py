"""
load.py
=======
Módulo responsável pela CARGA dos dados transformados.

Responsabilidade: definir o schema do BigQuery e carregar
o DataFrame final na tabela de destino.
"""

import pandas as pd
from google.cloud import bigquery


# =============================================
# SCHEMA DA TABELA NO BIGQUERY
# =============================================
# Define a estrutura e os tipos de dados de cada coluna.

schema_definido = [
    bigquery.SchemaField("codigo_bo",          "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("nome_municipio",      "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("nome_delegacia",      "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("ano_ocorrencia",      "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("mes_ocorrencia",      "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("data_ocorrencia_bo",  "DATE",    mode="NULLABLE"),
    bigquery.SchemaField("hora_ocorrencia_bo",  "TIME",    mode="NULLABLE"),
    bigquery.SchemaField("periodo_ocorrencia",  "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("dia_semana",          "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("local_ocorrencia",    "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("bairro",              "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("logradouro",          "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("latitude",            "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("longitude",           "FLOAT",   mode="NULLABLE"),
    bigquery.SchemaField("artigo_ocorrencia",   "STRING",  mode="NULLABLE"),
    bigquery.SchemaField("tipo_ocorrencia",     "STRING",  mode="NULLABLE"),
]


def carregar_dados_bigquery(
    df,
    project_id: str,
    table_id: str,
    schema: list
) -> None:
    """
    Carrega um DataFrame do Pandas em uma tabela do BigQuery
    usando um schema pré-definido e modo WRITE_TRUNCATE
    (substitui os dados existentes a cada carga).

    Args:
        df: DataFrame transformado e pronto para carga.
        project_id: ID do projeto no Google Cloud (ex: 'projetointegrador4-473718').
        table_id: ID completo da tabela no formato 'dataset.tabela'
                  (ex: 'dados_ssp.dados_ddm').
        schema: Lista de SchemaField do BigQuery definindo as colunas.
    """
    if df is None or df.empty:
        print("DataFrame está vazio. Nenhum dado para carregar.")
        return

    # --- PRÉ-PROCESSAMENTO: alinha o DataFrame com o schema ---
    colunas_schema = [field.name for field in schema]

    # 1. Mantém apenas as colunas que existem tanto no df quanto no schema
    colunas_presentes = [c for c in colunas_schema if c in df.columns]
    df = df[colunas_presentes].copy()

    # 2. Converte hora_ocorrencia_bo para datetime.time (BigQuery tipo TIME)
    if 'hora_ocorrencia_bo' in df.columns:
        def _para_time(val):
            try:
                parsed = pd.to_datetime(str(val), format='%H:%M:%S', errors='coerce')
                return parsed.time() if pd.notna(parsed) else None
            except Exception:
                return None
        df['hora_ocorrencia_bo'] = df['hora_ocorrencia_bo'].apply(_para_time)

    # 3. Garante que data_ocorrencia_bo seja do tipo date (BigQuery tipo DATE)
    if 'data_ocorrencia_bo' in df.columns:
        df['data_ocorrencia_bo'] = pd.to_datetime(
            df['data_ocorrencia_bo'], errors='coerce'
        ).dt.date

    print(f"Colunas a carregar: {list(df.columns)}")

    full_table_id = f"{project_id}.{table_id}"
    client = bigquery.Client(project=project_id)
    print(f"\nConectado ao projeto '{project_id}'.")

    # --- GARANTE QUE O DATASET EXISTE ---
    dataset_id = table_id.split('.')[0]
    full_dataset_id = f"{project_id}.{dataset_id}"
    
    try:
        client.get_dataset(full_dataset_id)
        print(f"✅ Dataset '{dataset_id}' localizado.")
    except Exception:
        print(f"⚠️ Dataset '{dataset_id}' não encontrado. Tentando criar...")
        try:
            client.create_dataset(bigquery.Dataset(full_dataset_id), timeout=30)
            print(f"✨ Dataset '{dataset_id}' criado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao criar dataset: {e}")
            if "403" in str(e):
                print("\n" + "!"*60)
                print("ERRO DE PERMISSÃO IDENTIFICADO!")
                print("Sua conta precisa da role 'BigQuery Admin' no PROJETO.")
                print("!"*60 + "\n")
            raise e

    # --- TESTE DE CONECTIVIDADE INICIAL ---
    try:
        # Tenta apenas listar os datasets para ver se o ID do projeto e a credencial básica funcionam
        datasets = list(client.list_datasets())
        print(f"✅ Verificação de Conexão: OK! Encontrados {len(datasets)} datasets no projeto.")
    except Exception as e:
        print("\n" + "!"*60)
        print("❌ ERRO CRÍTICO DE ACESSO AO PROJETO")
        print(f"Não foi possível ler as informações do projeto '{project_id}'.")
        print("\nMOTIVOS PROVÁVEIS:")
        print("1. O ID DO PROJETO está errado (verifique se não é 'projetointegrador5-XXXXXX').")
        print("2. A Conta de Serviço não tem permissão nem de 'Viewer' no projeto.")
        print("3. O Faturamento (Billing) não está ativado ou a API do BigQuery está desativada.")
        print("!"*60 + "\n")
        raise e

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
    )

    print(f"Iniciando o carregamento de {len(df)} linhas para '{full_table_id}'...")
    
    try:
        job = client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
        job.result() # Aguarda a conclusão
    except Exception as e:
        if "403" in str(e) or "Access Denied" in str(e):
            print("\n" + "!"*50)
            print("❌ ERRO DE PERMISSÃO (403 Forbidden)")
            print("Sua Conta de Serviço não tem permissão para criar Tarefas (Jobs) no BigQuery.")
            print(f"Verifique se o e-mail '{client.get_service_account_email() if hasattr(client, 'get_service_account_email') else 'da sua credencial'}' possui a role 'BigQuery Admin' ou 'BigQuery Job User' no projeto '{project_id}'.")
            print("!"*50 + "\n")
        raise e

    print("Carga de dados concluída com sucesso!")
    table = client.get_table(full_table_id)
    print(f"A tabela agora contém {table.num_rows} linhas.")
