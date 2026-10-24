# 🔍 Predição e Análise de Hotspots: Violência Contra a Mulher

### Projeto Integrador 5 | Ciência de Dados para Negócios — Fatec Votorantim

![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge)

🌐 **Acesse a Aplicação Online:** [pifatec.useconectai.com.br](https://pifatec.useconectai.com.br/)

---

## 📊 Sobre o Projeto

Este projeto representa a evolução da análise iniciada no PI4, agora consolidada em uma solução **end-to-end** de Ciência de Dados. O objetivo central é a transição de uma análise meramente descritiva para uma abordagem **preditiva**, focada na identificação de **hotspots** (pontos críticos) de violência contra a mulher nas cidades de **Sorocaba** e **Votorantim**.

A solução utiliza algoritmos de aprendizado de máquina para processar dados históricos da Secretaria de Segurança Pública de São Paulo (SSP-SP) obtidos através das Delegacias de Defesa da Mulher (DDMs) e prever a probabilidade de um determinado cenário (Município + Bairro + Mês + Ano) se tornar um ponto de alta concentração de ocorrências.

---

## 🛠️ Stack Tecnológica

O projeto foi construído utilizando as ferramentas mais modernas do ecossistema de Data Science:

- **Linguagem:** Python 3.9+
- **Machine Learning:**
  - **XGBoost Classifier** (Modelo preditivo final vencedor por alta acurácia e robustez)
  - **Target Encoding** (Representação estatística contínua de risco para bairros de alta cardinalidade)
  - **Negative Sampling** (Geração artificial de cenários negativos para balancear dados e mitigar o viés de seleção)
- **Data Engineering:**
  - **Google BigQuery** (Data Warehouse na nuvem para persistência e consultas estruturadas)
  - **Pipeline ETL modular** em Python (`src/`)
- **Interface & Experiência de Usuário (UI/UX):**
  - **Streamlit** com customização avançada via **CSS Injection**.
  - Tipografia premium utilizando **Google Fonts** (`Outfit` e `Inter`).
  - Cards de resultado dinâmicos com efeitos de **glassmorphism**, cores adaptativas e bordas coloridas de acordo com o nível de risco.
  - Gradiente radial dinâmico para background em **Dark Mode** (`#1a1025` a `#0d0d18`).
  - Barra de progresso horizontal personalizada e responsiva mostrando a probabilidade exata calculada pelo modelo.
- **Gestão:** ClickUp e Metodologia Ágil.

---

## 🚀 Arquitetura da Solução

O pipeline de inteligência do projeto segue o fluxo estruturado:

1. **ETL (`src/`):** Extração dos dados brutos de ocorrências da SSP-SP, transformação e carga estruturada no Google BigQuery.
2. **Negative Sampling:** Criação de um dataset balanceado simulando locais/tempos onde _não_ houve ocorrências registradas, combatendo o viés de seleção característico de dados criminais.
3. **Feature Engineering:**
   - **Lags Temporais:** Criação de variáveis de atraso mensal (`lag_mensal`) e anual (`lag_anual`) para capturar a persistência do crime.
   - **Média Móvel:** Cálculo da média móvel de 3 meses (`media_movel_3m`) para suavizar variações de curto prazo e identificar tendências locais.
   - **Target Encoding:** Conversão da variável categórica bairro (com alta cardinalidade) em um score contínuo de risco histórico, evitando sobreajuste.
4. **Modelagem:** Treinamento, validação temporal e tuning do classificador **XGBoost**, com persistência dos pesos (`models/modelo_hotspot_xgboost_v1.pkl`) e mapeamentos.
5. **Deployment:** Interface de consulta interativa em tempo real com design otimizado para tomada de decisão e interpretação facilitada para o usuário.

---

## 📈 Escala de Risco Preditivo

O modelo XGBoost estima a probabilidade de um cenário configurado se tornar um hotspot de ocorrências. Essa estimativa é categorizada na interface em 4 níveis visuais:

| Nível                | Probabilidade | Cor Indicativa | Significado Visual e Contextual                                               |
| :------------------- | :------------ | :------------: | :---------------------------------------------------------------------------- |
| 🟢 **Baixo Risco**   | < 25%         |     Verde      | Sem histórico significativo ou indicativo de hotspot para o cenário.          |
| 🟡 **Atenção**       | 25% - 50%     |    Amarelo     | Risco moderado; o local merece monitoramento preventivo leve.                 |
| 🟠 **Alto Risco**    | 50% - 75%     |    Laranja     | **Hotspot Identificado**; concentração de risco acima da média.               |
| 🔴 **Risco Crítico** | > 75%         |    Vermelho    | **Hotspot de Alta Intensidade**; atenção máxima e ação imediata recomendadas. |

---

## 📅 Cronograma de Desenvolvimento

Todas as etapas do cronograma do Projeto Integrador 5 foram concluídas com sucesso:

| Sprint       | Foco Principal                                                                                         |    Status    |
| :----------- | :----------------------------------------------------------------------------------------------------- | :----------: |
| **Sprint 1** | Alinhamento de escopo, especificação dos requisitos e coleta/carga de dados (ETL).                     | ✅ Concluído |
| **Sprint 2** | Análise Exploratória de Dados (EDA), Feature Engineering inicial e protótipo de modelagem.             | ✅ Concluído |
| **Sprint 3** | Engenharia avançada (Negative Sampling, Lags), tuning de hiperparâmetros e criação do App de Predição. | ✅ Concluído |
| **Sprint 4** | Overhaul de UI/UX da interface, consolidação de modelos, documentação final e fechamento.              | ✅ Concluído |

---

## 📂 Estrutura do Repositório

```text
├── app.py                # Dashboard Interativo e Preditor (Streamlit com CSS Overrides)
├── credenciais.json      # Arquivo de credenciais da conta de serviço para o BigQuery (ignorado no git)
├── requirements.txt      # Dependências do ecossistema do projeto
├── data/                 # Bases de dados locais de suporte
│   └── .gitkeep
├── models/               # Modelos treinados (.pkl) e mapeamentos estatísticos (.json)
│   ├── modelo_hotspot_xgboost_v1.pkl       # Modelo XGBoost Classifier final
│   ├── modelo_hotspot_random_forest_v1.pkl  # Modelo Random Forest alternativo para comparação
│   ├── target_encoding_bairros.json        # Mapeamentos de scores por bairro
│   ├── stats_bairros.json                  # Estatísticas de lags e média móvel por bairro
│   ├── bairro_municipio_map.json           # Dicionário de bairro -> município correspondente
│   ├── encoder_municipio.pkl               # LabelEncoder para os municípios
│   ├── encoder_bairro.pkl                  # LabelEncoder para bairros
│   └── encoder_local.pkl                   # LabelEncoder para locais físicos
├── notebooks/            # Jupyter Notebooks de desenvolvimento passo a passo
│   ├── 01_pipeline_etl.ipynb               # Extração de dados da SSP e ingestão no BigQuery
│   ├── 02_analise_exploratoria.ipynb       # Análise estatística e geoespacial de hotspots
│   └── 03_modelagem_preditiva.ipynb        # Modelagem final, tuning e exportação de artefatos
├── src/                  # Módulos Python reutilizáveis (ETL, Transformações)
│   ├── extraction.py                       # Funções para extração estruturada de dados
│   ├── transformation.py                   # Geração de cenários, lags e negative sampling
│   └── load.py                             # Cargas locais e interações com BigQuery
└── docs/                 # Documentos de entrega acadêmica e guias do usuário
    ├── Guia de Uso - PI5.pdf                # Manual detalhado de utilização do painel Streamlit
    ├── Deploy Guide - PI5.pdf              # Instruções técnicas para hospedagem e implantação
    ├── Apresentação - PI5.pdf              # Slides em PDF da defesa final do projeto
    ├── Relatório - PI5.pdf                 # Monografia/relatório científico completo
    ├── app_preview.png                     # Captura de tela da nova interface premium
    └── ... (outros relatórios e gráficos gerados no pipeline)
```

---

## 🌐 Demonstração Online (Deploy)

A aplicação está implantada e totalmente funcional em produção, podendo ser acessada de qualquer dispositivo sem a necessidade de instalação local:

👉 **Link de Acesso Direto:** [https://pifatec.useconectai.com.br/](https://pifatec.useconectai.com.br/)

> [!NOTE]
> O servidor de produção executa a interface interativa integrada aos modelos preditivos XGBoost treinados e persistidos, fornecendo análises e predições de hotspots em tempo real diretamente pelo navegador.

---

## 🚀 Como Executar o Projeto Localmente

### 1. Clonagem do Repositório e Configuração do Ambiente Virtual

```bash
# Clone o repositório
git clone https://github.com/[seu-usuario]/ProjetoIntegrador5.git
cd ProjetoIntegrador5

# Crie o ambiente virtual
python3 -m venv .venv

# Ativação no Mac/Linux:
source .venv/bin/activate

# Ativação no Windows:
# .venv\Scripts\activate
```

### 2. Instalação das Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Execução da Interface Streamlit

O aplicativo web pode ser executado localmente. Ele carrega automaticamente os modelos e dados estatísticos consolidados a partir da pasta `models/`, não necessitando de conexão ativa ao BigQuery para predições em tempo real:

```bash
streamlit run app.py
```

---

## 👥 Autores

- **Débora Kocks Nogueira**
- **Giovana Perugini Guenka**
- **Stefanie Mayumi Inacio Kobayashi**
- **Vitória Lisauskaz Ferraz da Silva**

---

## 🎓 Instituição

**Fatec Votorantim** – Curso Superior de Tecnologia em Ciência de Dados para Negócios.

---

<p align="center">
  Desenvolvido com ❤️ para promover a inteligência de dados na segurança pública e combater a violência contra a mulher.
</p>
