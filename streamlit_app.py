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
/* Tipografía y colores base */
html, body, [class*="css"]  {font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji','Segoe UI Emoji';}

/***** Tarjetas *****/
.card {
  background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px; padding: 16px 18px; color: #e5e7eb;
  box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}
.card h3 { margin: 0 0 8px 0; font-weight: 700; color: #f9fafb; }
.card .muted { color: #9ca3af; font-size: 0.9rem; }

/***** Badges *****/
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; }
.badge.green { background:#064e3b; color:#d1fae5; }
.badge.yellow{ background:#78350f; color:#fde68a; }
.badge.red   { background:#7f1d1d; color:#fecaca; }

/***** Botón primario *****/
button[kind="primary"] { border-radius: 10px !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============== Utilidades de datos ==============

def get_service_account_from_env():
    """Intenta obtener credenciales de servicio desde env var GOOGLE_SERVICE_ACCOUNT_JSON.
    También soporta st.secrets["google_service_account_json"]."""
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        try:
            raw = st.secrets.get("google_service_account_json", None)
        except Exception:
            raw = None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

@st.cache_resource(show_spinner=True)
def read_google_sheet(spreadsheet_id: str, worksheet: str) -> pd.DataFrame:
    """Lee Google Sheets usando credenciales de servicio (gspread-less, via pandas API v4 simplificada).
    Aquí usamos la API vía 'gspread' si está disponible; si no, recurrimos a CSV export.
    Para simplicidad y compatibilidad en Render, usaremos la descarga CSV pública si el doc está compartido,
    o gspread si hay credenciales.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception:
        gspread = None

    creds_json = get_service_account_from_env()

    if gspread and creds_json:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(creds_json, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(spreadsheet_id)
        ws = sh.worksheet(worksheet)
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        return df
    else:
        # Fallback: CSV export (requiere que la hoja sea pública o compartida con "cualquiera con el enlace")
        # Nota: Render no permite requests sin dependencias extra, pero pandas puede leer si es público
        import pandas as pd
        csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={worksheet}"
        try:
            return pd.read_csv(csv_url)
        except Exception as e:
            st.error("No se pudo leer Google Sheets. Configura credenciales de servicio o haz la hoja pública.")
            raise e

# ============== Limpieza / Ingeniería (compatibles con tu notebook) ==============

def to_number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace('%','', regex=False)
    s = s.str.replace(r'[\$Qq, ]','', regex=True)
    s = s.str.replace(r'\.(?=.*\.)','', regex=True)
    return pd.to_numeric(s, errors='coerce')

def clean_percent(series: pd.Series) -> pd.Series:
    vals = to_number(series)
    if (vals > 1).mean() > 0.5:  # mayormente 0-100
        vals = vals.clip(0,100)/100
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

# ============== Transformaciones como tu notebook ==============

df = df_raw.copy()
for col in ['compras','pagos','limite_credito','plazo_credito','no_pagos_atrasados']:
    if col in df.columns:
        df[col] = to_number(df[col]).astype(float).clip(lower=0)

if 'clasifica_rep_legal' in df.columns:
    df['clasifica_rep_legal_score'] = map_clasificacion(df['clasifica_rep_legal'])
if 'clasificacion_cliente' in df.columns:
    df['clasificacion_cliente_score'] = map_clasificacion(df['clasificacion_cliente'])
if 'no_empleados' in df.columns:
    df['no_empleados_cat'] = map_no_empleados(df['no_empleados'])
if 'exactitud' in df.columns:
    df['exactitud_frac'] = clean_percent(df['exactitud'])

df['hubo_impago'] = pd.to_numeric(df.get('hubo_impago', 0), errors='coerce').fillna(0).astype(int).clip(0,1)

if 'pais' in df.columns:
    df = pd.get_dummies(df, columns=['pais'], drop_first=True)

# Features válidas
base_feats = ['compras','pagos','plazo_credito','no_pagos_atrasados',
              'clasifica_rep_legal_score','clasificacion_cliente_score','no_empleados_cat']
pais_feats = [c for c in df.columns if c.startswith('pais_')]
all_feats = [c for c in base_feats if c in df.columns] + pais_feats
excluir = set(pais_feats) | {'has_ig','address_flag','actividad_actual_flag','exactitud_frac','exactitud_num'}
features = [c for c in all_feats if c not in excluir]

X = df[features].apply(pd.to_numeric, errors='coerce') if features else pd.DataFrame()
features_validas = [c for c in X.columns if X[c].notna().any()] if not X.empty else []
X = X[features_validas] if not X.empty else X

imputer = SimpleImputer(strategy='median')
if not X.empty and len(features_validas)>0:
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features_validas)
else:
    st.error("No hay features válidas después de la limpieza. Revisa tu hoja.")
    st.stop()

y = df['hubo_impago'].astype(int)

# ============== Entrenamiento del modelo (cacheado) ==============

@st.cache_resource(show_spinner=True)
def train_models(X_imp: pd.DataFrame, y: pd.Series, df: pd.DataFrame):
    # Clasificador
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.25, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    # Regresor de límite (opcional)
    reg = None
    if 'limite_credito' in df.columns:
        mask_reg = df['limite_credito'].notna()
        if mask_reg.any():
            reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
            reg.fit(X_imp.loc[mask_reg], df.loc[mask_reg, 'limite_credito'])

    # ROC k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 101)
    for i, (tr, te) in enumerate(skf.split(X_imp, y)):
        clf_i = RandomForestClassifier(n_estimators=400, random_state=42+i, n_jobs=-1)
        clf_i.fit(X_imp.iloc[tr], y.iloc[tr])
        proba = clf_i.predict_proba(X_imp.iloc[te])[:,1]
        fpr, tpr, _ = roc_curve(y.iloc[te], proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tpr_interp = np.interp(mean_fpr, fpr, tpr); tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr); std_auc = np.std(aucs)

    # Hold-out metrics
    proba_test = clf.predict_proba(X_test)[:,1]
    pred_05 = (proba_test >= 0.5).astype(int)
    metrics = {
        'accuracy': float(accuracy_score(y_test, pred_05)),
        'f1_macro': float(f1_score(y_test, pred_05, average='macro')),
        'f1_impago': float(f1_score(y_test, pred_05, pos_label=1)),
        'auc_holdout': float(roc_auc_score(y_test, proba_test)),
        'confusion_matrix': confusion_matrix(y_test, pred_05).tolist(),
        'classification_report': classification_report(y_test, pred_05, digits=3, output_dict=True)
    }

    # Importancias
    importances = pd.Series(clf.feature_importances_, index=X_imp.columns).sort_values(ascending=False)

    artifacts = {
        'clf': clf,
        'reg': reg,
        'imputer': imputer,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'mean_auc': mean_auc, 'std_auc': std_auc,
        'metrics': metrics,
        'importances': importances
    }
    return artifacts

art = train_models(X_imp, y, df)
clf, reg = art['clf'], art['reg']

# ============== Encabezado ==============
st.title("💳 Scoring de Crédito")
st.caption("Conectado a Google Sheets • RandomForest • Informe con ROC, métricas e importancias • Formulario interactivo")

# Tarjetas rápidas
a, b, c, d = st.columns(4)
a.markdown(f"""
<div class='card'>
  <h3>AUC (k-fold)</h3>
  <div class='muted'>{art['mean_auc']:.3f} ± {art['std_auc']:.3f}</div>
</div>
""", unsafe_allow_html=True)
b.markdown(f"""
<div class='card'>
  <h3>AUC (hold-out)</h3>
  <div class='muted'>{art['metrics']['auc_holdout']:.3f}</div>
</div>
""", unsafe_allow_html=True)
c.markdown(f"""
<div class='card'>
  <h3>Accuracy</h3>
  <div class='muted'>{art['metrics']['accuracy']:.3f}</div>
</div>
""", unsafe_allow_html=True)
d.markdown(f"""
<div class='card'>
  <h3>F1 (impago)</h3>
  <div class='muted'>{art['metrics']['f1_impago']:.3f}</div>
</div>
""", unsafe_allow_html=True)

# ============== Tabs ==============

tab_grafs, tab_report, tab_form = st.tabs(["📊 Gráficas", "📋 Reporte", "📝 Formulario"])

# --- 📊 Gráficas ---
with tab_grafs:
    st.subheader("Curva ROC promedio (k=5)")
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(art['mean_fpr'], art['mean_tpr'], lw=2, label=f"Media ROC (AUC={art['mean_auc']:.3f} ± {art['std_auc']:.3f})")
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(loc='lower right'); ax.grid(True)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Importancia de variables")
    imp = art['importances'][:15][::-1]
    vals, labs = imp.values, imp.index
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
    colors = cm.viridis(norm)
    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.barh(labs, vals, color=colors, edgecolor='none')
    ax2.set_xlabel('Importancia'); ax2.set_ylabel('Variable'); ax2.grid(axis='x', alpha=0.2)
    st.pyplot(fig2, use_container_width=True)

# --- 📋 Reporte ---
with tab_report:
    st.subheader("Métricas (hold-out)")
    m = art['metrics']
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{m['accuracy']:.3f}")
    col2.metric("F1 (macro)", f"{m['f1_macro']:.3f}")
    col3.metric("AUC", f"{m['auc_holdout']:.3f}")

    st.markdown("---")
    st.subheader("Matriz de confusión")
    cmx = np.array(m['confusion_matrix'])
    fig_cm = go.Figure(data=go.Heatmap(z=cmx, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"], text=cmx, texttemplate="%{text}", colorscale="Blues"))
    fig_cm.update_layout(height=350, margin=dict(l=40,r=40,t=30,b=30))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Reporte de clasificación")
    report_df = pd.DataFrame(m['classification_report']).T
    st.dataframe(report_df.style.format({"precision":"{:.3f}", "recall":"{:.3f}", "f1-score":"{:.3f}", "support":"{:.0f}"}), use_container_width=True)

# --- 📝 Formulario ---
with tab_form:
    st.subheader("Simulador de scoring")

    def clasif_to_score(v):
        return {'muy bueno':3, 'bueno':2, 'regular':1, 'malo':0}.get(v, np.nan)
    def noemp_to_cat(v):
        return {'0-25':0, '26-50':1, '51-100':2, 'más de 100':3}.get(v, np.nan)

    # Widgets (dos columnas para que se vea compacto pero limpio)
    c1, c2 = st.columns(2)
    with c1:
        compras = st.number_input('Compras (US$)', min_value=0.0, value=0.0, step=100.0)
        pagos   = st.number_input('Pagos (US$)',   min_value=0.0, value=0.0, step=100.0)
        plazo   = st.number_input('Plazo crédito (días)', min_value=0.0, value=30.0, step=1.0)
        atrasos = st.number_input('No. pagos atrasados', min_value=0, value=0, step=1)
    with c2:
        rep = st.selectbox('Clasificación Rep. Legal', ['muy bueno','bueno','regular','malo'], index=1)
        cli = st.selectbox('Clasificación Cliente',   ['muy bueno','bueno','regular','malo'], index=1)
        noemp = st.selectbox('No. empleados', ['0-25','26-50','51-100','más de 100'])
        penal = st.slider('Penalización de riesgo', 0.0, 1.0, 0.70, 0.05)

    # Construimos el vector de entrada respetando las columnas del modelo
    row = pd.Series(0.0, index=X_imp.columns, dtype=float)
    if 'compras' in row.index: row['compras'] = float(compras)
    if 'pagos' in row.index: row['pagos'] = float(pagos)
    if 'plazo_credito' in row.index: row['plazo_credito'] = float(plazo)
    if 'no_pagos_atrasados' in row.index: row['no_pagos_atrasados'] = float(atrasos)
    if 'clasifica_rep_legal_score' in row.index: row['clasifica_rep_legal_score'] = float(clasif_to_score(rep))
    if 'clasificacion_cliente_score' in row.index: row['clasificacion_cliente_score'] = float(clasif_to_score(cli))
    if 'no_empleados_cat' in row.index: row['no_empleados_cat'] = float(noemp_to_cat(noemp))

    btn = st.button("Calcular riesgo", type="primary")
    if btn:
        xnew = pd.DataFrame([row])
        xnew_imp = pd.DataFrame(imputer.transform(xnew), columns=X_imp.columns)
        proba = float(clf.predict_proba(xnew_imp)[0,1])
        limite_modelo = float(reg.predict(xnew_imp)[0]) if art['reg'] is not None else 0.0
        limite_ajust = max(0.0, limite_modelo * (1 - penal * proba))
        limite_ajust = round(limite_ajust / 100.0) * 100.0

        risk_badge = 'green' if proba < 0.25 else 'yellow' if proba < 0.6 else 'red'
        st.markdown(f"""
        <div class='card'>
          <h3>Resultado</h3>
          <span class='badge {risk_badge}'>Riesgo: {proba*100:.2f}%</span>
          <p class='muted' style='margin-top:10px;'>
            Límite (modelo): <b>{limite_modelo:,.0f}</b><br>
            Límite ajustado (riesgo): <b>{limite_ajust:,.0f}</b>
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge Plotly (decorativo)
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = proba*100,
            title = {'text': "Probabilidad de impago (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, 25], 'color': '#064e3b'},
                    {'range': [25, 60], 'color': '#78350f'},
                    {'range': [60, 100], 'color': '#7f1d1d'}
                ]
            }
        ))
        gauge.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(gauge, use_container_width=True)

# ============== Footer ==============
st.caption("© {} Scoring demo • Streamlit + Google Sheets + RandomForest".format(pd.Timestamp.today().year))
