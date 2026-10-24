"""
Aplicativo de Predição de Hotspots — PI5
=========================================
Prediz o risco de violência contra a mulher em um bairro de Sorocaba ou Votorantim
com base em dados de ocorrências históricas das DDMs (2022–2026).

Como rodar:
    .venv/bin/streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Hotspots — VCM",
    page_icon="🔍",
    layout="centered",
)

# ── CSS customizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;600&display=swap');

    /* ── Tipografia global ── */
    html, body, [class*="css"], p, label {
        font-family: 'Outfit', 'Inter', sans-serif !important;
    }

    /* ── Layout ── */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 800px; }

    /* ── Background do app — segue o tema do Streamlit ── */
    [data-testid="stAppViewContainer"] {
        background: var(--background-color) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* Dark mode: gradiente sobre o fundo padrão */
    @media (prefers-color-scheme: dark) {
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #1a1025 0%, #0d0d18 100%) !important;
        }
    }

    /* ── Header Block ── */
    .header-block {
        background: var(--secondary-background-color) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 16px;
        padding: 1.2rem 1.8rem 1rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
        position: relative;
        overflow: hidden;
    }
    .header-block::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(138,43,226,0.08) 0%, transparent 60%);
        z-index: 0; pointer-events: none;
    }
    .main-title {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -1px !important;
        line-height: 1.1 !important;
        position: relative; z-index: 1;
        background: linear-gradient(to right, #7c3aed, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    @media (prefers-color-scheme: dark) {
        .main-title {
            background: linear-gradient(to right, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
        }
    }
    .subtitle {
        font-size: 1.05rem;
        margin-top: 0.6rem;
        margin-bottom: 0;
        font-weight: 300;
        position: relative; z-index: 1;
        color: var(--text-color) !important;
        opacity: 0.75;
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-left: 4px solid #818cf8;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.92rem;
        color: var(--text-color) !important;
        margin-bottom: 1.4rem;
    }
    .info-box strong { color: var(--text-color) !important; font-weight: 700; }

    /* ── Labels dos widgets ── */
    .stSelectbox label,
    .stSlider label,
    div[data-testid="stWidgetLabel"] > p,
    div[data-testid="stWidgetLabel"] label {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
    }

    /* ── Selectbox / Dropdown ── */
    div[data-baseweb="select"] > div {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.30) !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
    }
    @media (prefers-color-scheme: dark) {
        div[data-baseweb="select"] > div:hover {
            border-color: #a78bfa !important;
            box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.20) !important;
        }
    }
    /* Texto dentro dos selectboxes */
    div[data-baseweb="select"] [data-testid="stWidgetLabel"],
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] [aria-selected="true"],
    div[data-baseweb="select"] [role="option"] span,
    div[data-baseweb="select"] > div > div > div {
        color: var(--text-color) !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.95rem !important;
    }

    /* ── Botao Principal ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
        letter-spacing: 0.02em !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.50) !important;
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    }
    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span {
        color: #ffffff !important;
    }

    /* ── Result Cards ── */
    .result-card {
        border-radius: 16px;
        padding: 1.6rem 2rem;
        text-align: left;
        animation: fadeIn 0.5s ease-out;
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }
    .result-label {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .result-sub {
        font-size: 0.95rem;
        font-weight: 400;
        opacity: 0.90;
        letter-spacing: 0.01em;
    }
    .result-baixo {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.22));
        border-left: 5px solid #10b981;
        color: #059669;
        box-shadow: 0 6px 24px rgba(16,185,129,0.12);
    }
    @media (prefers-color-scheme: dark) {
        .result-baixo { color: #6ee7b7; }
    }
    .result-atencao {
        background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(245,158,11,0.22));
        border-left: 5px solid #f59e0b;
        color: #b45309;
        box-shadow: 0 6px 24px rgba(245,158,11,0.12);
    }
    @media (prefers-color-scheme: dark) {
        .result-atencao { color: #fcd34d; }
    }
    .result-alto {
        background: linear-gradient(135deg, rgba(249,115,22,0.12), rgba(249,115,22,0.22));
        border-left: 5px solid #f97316;
        color: #c2410c;
        box-shadow: 0 6px 24px rgba(249,115,22,0.12);
    }
    @media (prefers-color-scheme: dark) {
        .result-alto { color: #fdba74; }
    }
    .result-critico {
        background: linear-gradient(135deg, rgba(225,29,72,0.12), rgba(225,29,72,0.22));
        border-left: 5px solid #e11d48;
        color: #be123c;
        box-shadow: 0 6px 24px rgba(225,29,72,0.15);
    }
    @media (prefers-color-scheme: dark) {
        .result-critico { color: #fda4af; }
    }

    /* ── Legend Bar ── */
    .legend-bar { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
    .legend-item {
        flex: 1; text-align: center;
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.3rem 0.4rem; border-radius: 6px;
    }
    .legend-baixo   { background: rgba(16,185,129,0.15); color: #059669; border: 1px solid rgba(16,185,129,0.35); }
    .legend-atencao { background: rgba(245,158,11,0.15); color: #b45309; border: 1px solid rgba(245,158,11,0.35); }
    .legend-alto    { background: rgba(249,115,22,0.15); color: #c2410c; border: 1px solid rgba(249,115,22,0.35); }
    .legend-critico { background: rgba(225,29,72,0.15);  color: #be123c; border: 1px solid rgba(225,29,72,0.35); }
    @media (prefers-color-scheme: dark) {
        .legend-baixo   { color: #6ee7b7; }
        .legend-atencao { color: #fcd34d; }
        .legend-alto    { color: #fdba74; }
        .legend-critico { color: #fda4af; }
    }

    /* ── Progress Bar customizada ── */
    .risk-track {
        width: 100%;
        height: 10px;
        background: rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .risk-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #f59e0b, #f97316, #e11d48);
        background-repeat: no-repeat;
        border-radius: 10px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        min-width: 6px;
    }

    /* ── Expander ── */
    .stExpander {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.18) !important;
        border-radius: 12px !important;
    }

    /* ── Caption / Rodape ── */
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: var(--text-color) !important;
        opacity: 0.65;
    }

    /* ── Divisor ── */
    hr {
        border-color: rgba(128, 128, 128, 0.25) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* ── Animacao ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)




# ── Carregamento dos modelos e mapeamentos ──────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource(show_spinner="Carregando modelo…")
def carregar_modelo():
    # Carregar explicitamente o modelo XGBoost vencedor
    modelo_path = os.path.join(MODELS_DIR, "modelo_hotspot_xgboost_v1.pkl")
    modelo    = joblib.load(modelo_path)
    le_munic  = joblib.load(os.path.join(MODELS_DIR, "encoder_municipio.pkl"))
    
    # Mapeamento bairro → município
    mapa_path = os.path.join(MODELS_DIR, "bairro_municipio_map.json")
    bairro_municipio = {}
    if os.path.exists(mapa_path):
        with open(mapa_path, "r", encoding="utf-8") as f:
            bairro_municipio = json.load(f)

    # Carregar o mapeamento de Target Encoding (Score de Risco por bairro)
    target_path = os.path.join(MODELS_DIR, "target_encoding_bairros.json")
    target_encoding = {}
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            target_encoding = json.load(f)

    # Estatísticas de Bairros (Lags e Média Móvel)
    stats_path = os.path.join(MODELS_DIR, "stats_bairros.json")
    stats_bairros = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_bairros = json.load(f)

    return modelo, le_munic, bairro_municipio, target_encoding, stats_bairros

modelo, le_municipio, BAIRRO_MUNICIPIO, TARGET_ENCODING, STATS_BAIRROS = carregar_modelo()

# ── Mapeamentos fixos (iguais ao notebook 03) ──────────────────────────────
MESES = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4,
    "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8,
    "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12,
}
FEATURES = [
    "mes_ocorrencia", "bairro_target_enc", "lag_mensal", 
    "lag_anual", "media_movel_3m", "municipio_enc", "ano_ocorrencia"
]


# ── Interface ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="header-block">'
    '<h1 class="main-title">Predição de Hotspots</h1>'
    '<p class="subtitle">Violência Contra a Mulher — DDMs de Sorocaba e Votorantim</p>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-box">Preencha as informações abaixo e clique em <strong>Analisar risco</strong> '
    'para saber se o cenário representa um <strong>local de alto risco (hotspot)</strong>, '
    'com base no modelo espaço-temporal.</div>',
    unsafe_allow_html=True,
)

# ── Primeira linha (Município e Bairro) ──
col1, col2 = st.columns(2)

with col1:
    municipio_sel = st.selectbox("Município", options=sorted(le_municipio.classes_.tolist()))

with col2:
    if BAIRRO_MUNICIPIO:
        bairros_disp = sorted([
            b for b, m in BAIRRO_MUNICIPIO.items() 
            if m == municipio_sel and b in TARGET_ENCODING
        ])
    else:
        bairros_disp = sorted(list(TARGET_ENCODING.keys()))
    bairro_sel = st.selectbox("Bairro", options=bairros_disp)

# ── Segunda linha (Mês, Ano e Botão Inline) ──
col3, col4, col5 = st.columns([1.5, 1.5, 1.2])

with col3:
    mes_sel = st.selectbox("Mês", options=list(MESES.keys()))

with col4:
    ano_sel = st.selectbox("Ano de referência", options=list(range(2022, 2031)), index=4)

with col5:
    st.markdown("<div style='margin-top: 1.81rem;'></div>", unsafe_allow_html=True)
    analisar = st.button("Analisar Risco", use_container_width=True, type="primary")


# ── Predição ────────────────────────────────────────────────────────────────
if analisar:
    try:
        municipio_enc = le_municipio.transform([municipio_sel])[0]
        # O bairro_enc agora é o próprio score de risco
        bairro_score = float(TARGET_ENCODING.get(bairro_sel, 0))
    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
        st.stop()

    mes_val = MESES[mes_sel]
    
    # Busca o Score de Risco (Target Encoding) e os Stats do bairro selecionado
    b_score = float(TARGET_ENCODING.get(bairro_sel, 0))
    b_stats = STATS_BAIRROS.get(bairro_sel, {'lag_mensal': 0, 'lag_anual': 0, 'media_movel_3m': 0})

    cenario = pd.DataFrame([{
        "mes_ocorrencia":  mes_val,
        "bairro_target_enc": bairro_score,
        "lag_mensal":      b_stats['lag_mensal'],
        "lag_anual":       b_stats['lag_anual'],
        "media_movel_3m":  b_stats['media_movel_3m'],
        "municipio_enc":   municipio_enc,
        "ano_ocorrencia":  ano_sel,
    }])[FEATURES]
    
    prob = modelo.predict_proba(cenario)[0]   # [P(normal), P(hotspot)]
    p_risco = prob[1]  # probabilidade de alto risco (0–1)

    # ── Classificação em 4 níveis ──────────────────────────────────────────
    if p_risco < 0.25:
        nivel      = "baixo"
        css_class  = "result-card result-baixo"
        badge_label = "BAIXO RISCO"
        badge_sub   = "Sem indicativo de hotspot"
        descricao_nivel = "não apresenta histórico significativo de ocorrências e é classificado como **baixo risco**."
    elif p_risco < 0.50:
        nivel      = "atencao"
        css_class  = "result-card result-atencao"
        badge_label = "ATENÇÃO"
        badge_sub   = "Risco moderado"
        descricao_nivel = "apresenta um volume **moderado** de ocorrências. Merece atenção, mas não é classificado como hotspot pelo modelo."
    elif p_risco < 0.75:
        nivel      = "alto"
        css_class  = "result-card result-alto"
        badge_label = "ALTO RISCO"
        badge_sub   = "Hotspot Identificado"
        descricao_nivel = "concentra **acima da média** de ocorrências (percentil 75+) e é classificado como hotspot de risco."
    else:
        nivel      = "critico"
        css_class  = "result-card result-critico"
        badge_label = "RISCO CRÍTICO"
        badge_sub   = "Hotspot de Alta Intensidade"
        descricao_nivel = "apresenta **concentração muito elevada** de ocorrências e é classificado como hotspot crítico. Atenção máxima recomendada."

    st.markdown("---")
    st.markdown("**Resultado da Análise**")

    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="result-label">{badge_label}</div>'
        f'<div class="result-sub">{badge_sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)

    # Barra de risco com legenda visual
    st.markdown(
        '<div class="legend-bar">'
        '<span class="legend-item legend-baixo">Baixo &lt;25%</span>'
        '<span class="legend-item legend-atencao">Atenção 25–50%</span>'
        '<span class="legend-item legend-alto">Alto 50–75%</span>'
        '<span class="legend-item legend-critico">Crítico &gt;75%</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    bg_size = f"{100 / p_risco:.0f}% 100%" if p_risco > 0.01 else "100% 100%"
    st.markdown(
        f'<div class="risk-track">'
        f'<div class="risk-fill" style="width: {p_risco:.1%}; background-size: {bg_size};"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Probabilidade de alto risco calculada: **{p_risco:.1%}**")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Como interpretar este resultado?"):
        st.markdown(
            f"""
            O modelo analisou o seguinte cenário:
            - **Município:** {municipio_sel}
            - **Bairro:** {bairro_sel}
            - **Mês:** {mes_sel}
            - **Ano:** {ano_sel}

            Com base no histórico de ocorrências nas DDMs, este bairro {descricao_nivel}

            **Sobre a probabilidade calculada ({p_risco:.1%}):**  
            O modelo calcula a chance de esse cenário ser classificado como hotspot com base nos padrões
            históricos. Quanto maior o valor, mais o bairro se assemelha a locais com alta concentração de ocorrências.

            | Faixa de Probabilidade | Classificação |
            |---|---|
            | Menor que 25% | Baixo risco |
            | Entre 25% e 50% | Atenção |
            | Entre 50% e 75% | Alto risco |
            | Acima de 75% | Risco crítico |

            > Aviso: Este é um modelo preditivo baseado em dados históricos, não em tempo real.
            > Os resultados devem ser interpretados como apoio à análise, não como diagnóstico definitivo.
            """
        )

# ── Rodapé ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Projeto Integrador 5 — Fatec Votorantim · Ciência de Dados para Negócios · 2026  \n"
    "Modelo: XGBoost treinado com dados da SSP-SP (DDMs de Sorocaba e Votorantim)"
)