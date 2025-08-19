# streamlit_app.py — Scoring de Crédito con Google Sheets + Tabs + UI
# ---------------------------------------------------------------
# App de Streamlit con:
#  - Conexión a Google Sheets (vía credenciales de servicio)
#  - Limpieza/ingeniería de variables (compatible con tu notebook)
#  - Modelo RandomForest (clasificación) + Regresor de límite de crédito
#  - Pestañas: Gráficas | Reporte | Formulario
#  - UI estilizada con tarjetas y plots Plotly/Matplotlib
# ---------------------------------------------------------------
# Cómo usar (local):
#   1) pip install -r requirements.txt
#   2) Define variables de entorno:
#        GOOGLE_SERVICE_ACCOUNT_JSON = '{...json de la cuenta de servicio...}'
#        GSHEET_SPREADSHEET_ID       = 'tu_spreadsheet_id'
#        GSHEET_WORKSHEET            = 'nombre_de_hoja' (p.ej. 'Hoja 1')
#   3) streamlit run streamlit_app.py
#
# Cómo usar (Render):
#   - Configura un servicio Web (Build Command: "pip install -r requirements.txt")
#   - Start Command: "streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"
#   - Añade las variables de entorno anteriores en el Dashboard de Render.

import os, re, json, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, auc, accuracy_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

# ============== Estilos globales ==============
st.set_page_config(page_title="Scoring de Crédito", page_icon="💳", layout="wide")

CUSTOM_CSS = """
<style>
html, body, [class*="css"]  {font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji','Segoe UI Emoji';}
.card {background: linear-gradient(180deg, #0f172a 0%, #111827 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 16px 18px; color: #e5e7eb; box-shadow: 0 10px 25px rgba(0,0,0,0.25);}
.card h3 { margin: 0 0 8px 0; font-weight: 700; color: #f9fafb; }
.card .muted { color: #9ca3af; font-size: 0.9rem; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; }
.badge.green { background:#064e3b; color:#d1fae5; }
.badge.yellow{ background:#78350f; color:#fde68a; }
.badge.red   { background:#7f1d1d; color:#fecaca; }
button[kind="primary"] { border-radius: 10px !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============== Utilidades de datos ==============

def get_service_account_from_env():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        try: raw = st.secrets.get("google_service_account_json", None)
        except Exception: raw = None
    if not raw: return None
    try: return json.loads(raw)
    except Exception: return None

@st.cache_resource(show_spinner=True)
def read_google_sheet(spreadsheet_id: str, worksheet: str) -> pd.DataFrame:
    creds_json = get_service_account_from_env()
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        if creds_json:
            scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
            creds = Credentials.from_service_account_info(creds_json, scopes=scopes)
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(spreadsheet_id)
            ws = sh.worksheet(worksheet)
            data = ws.get_all_records()
            return pd.DataFrame(data)
    except Exception: pass  # fallback

    # CSV público fallback
    csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={worksheet}"
    try:
        return pd.read_csv(csv_url)
    except Exception as e:
        st.error("No se pudo leer Google Sheets. Configura credenciales o haz la hoja pública.")
        raise e

# ============== Limpieza / Ingeniería ==============

def to_number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace('%','', regex=False)
    s = s.str.replace(r'[\$Qq, ]','', regex=True)
    s = s.str.replace(r'\.(?=.*\.)','', regex=True)
    return pd.to_numeric(s, errors='coerce')

def clean_percent(series: pd.Series) -> pd.Series:
    vals = to_number(series)
    if (vals > 1).mean() > 0.5: vals = vals.clip(0,100)/100
    return vals.clip(0,1)

def map_clasificacion(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {'muy bueno':3,'muybueno':3,'mb':3,'bueno':2,'b':2,'regular':1,'r':1,'malo':0,'m':0}
    out = s.map(mapping)
    out_num = pd.to_numeric(series, errors='coerce')
    return out.fillna(out_num).astype(float)

def map_no_empleados(series: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x): return np.nan
        t = str(x).strip().lower().replace('más','mas').replace('empleados','').strip()
        if re.search(r'^\s*0\s*-\s*25|0\s*a\s*25|0\s*–\s*25', t): return 0
        if re.search(r'^\s*26\s*-\s*50|26\s*a\s*50|26\s*–\s*50', t): return 1
        if re.search(r'^\s*51\s*-\s*100|51\s*a\s*100|51\s*–\s*100', t): return 2
        if 'mas de 100' in t or '100+' in t or re.search(r'>\s*100', t): return 3
        m = re.search(r'\d+', t)
        if m:
            v = int(m.group())
            if v <= 25: return 0
            if v <= 50: return 1
            if v <= 100: return 2
            return 3
        return np.nan
    return series.apply(f).astype(float)

# ============== Carga de datos ==============

SPREADSHEET_ID = os.environ.get('GSHEET_SPREADSHEET_ID', st.secrets.get('GSHEET_SPREADSHEET_ID', ''))
WORKSHEET      = os.environ.get('GSHEET_WORKSHEET', st.secrets.get('GSHEET_WORKSHEET', 'Hoja 1'))

st.sidebar.header("⚙️ Configuración")
st.sidebar.text_input("Spreadsheet ID", value=SPREADSHEET_ID, key="sheet_id")
st.sidebar.text_input("Worksheet", value=WORKSHEET, key="ws_name")
reload_btn = st.sidebar.button("🔄 Recargar datos")
if reload_btn and st.session_state.sheet_id:
    SPREADSHEET_ID = st.session_state.sheet_id
    WORKSHEET = st.session_state.ws_name

with st.spinner("Conectando a Google Sheets..."):
    df_raw = read_google_sheet(SPREADSHEET_ID, WORKSHEET)
st.success(f"Datos cargados desde Google Sheets: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")

# ============== Transformaciones, modelo, tabs, formulario ==============
# El resto de tu código se mantiene exactamente igual, incluyendo:
# - Limpieza de columnas
# - Generación de features
# - Entrenamiento RandomForest
# - Cálculo de métricas
# - Gráficas Plotly/Matplotlib
# - Formulario interactivo con riesgo y límite de crédito
# - Footer
# No es necesario modificar nada más, solo se reemplaza la función read_google_sheet
