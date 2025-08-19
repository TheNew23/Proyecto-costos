import os, json
import pandas as pd
import numpy as np
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix, classification_report
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# ---------------------------
# 1) Conectar con Google Sheets
# ---------------------------
service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(service_account_info, scopes=scope)
gc = gspread.authorize(credentials)

SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID")
WORKSHEET = os.getenv("GSHEET_WORKSHEET", "Hoja 1")

sh = gc.open_by_key(SPREADSHEET_ID)
ws = sh.worksheet(WORKSHEET)
df = pd.DataFrame(ws.get_all_records())

st.title("Dashboard de Riesgo Crediticio")

# ---------------------------
# 2) Limpieza y transformación simple
# ---------------------------
def to_number(series):
    s = series.astype(str).str.replace('%','', regex=False).str.replace(r'[\$Qq, ]','', regex=True)
    return pd.to_numeric(s, errors='coerce')

def map_score(series):
    mapping = {'muy bueno':3, 'bueno':2, 'regular':1, 'malo':0}
    s = series.astype(str).str.strip().str.lower()
    return s.map(mapping)

for col in ['compras','pagos','plazo_credito','no_pagos_atrasados']:
    if col in df.columns:
        df[col] = to_number(df[col]).clip(lower=0)

if 'clasifica_rep_legal' in df.columns:
    df['clasifica_rep_legal_score'] = map_score(df['clasifica_rep_legal'])
if 'clasificacion_cliente' in df.columns:
    df['clasificacion_cliente_score'] = map_score(df['clasificacion_cliente'])

df['hubo_impago'] = pd.to_numeric(df.get('hubo_impago', 0)).fillna(0).astype(int)

# ---------------------------
# 3) Features y imputación
# ---------------------------
features = ['compras','pagos','plazo_credito','no_pagos_atrasados','clasifica_rep_legal_score','clasificacion_cliente_score']
X = df[features].apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features)
y = df['hubo_impago'].astype(int)

# ---------------------------
# 4) Entrenamiento
# ---------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_imp, y)

reg = None
if 'limite_credito' in df.columns:
    mask = df['limite_credito'].notna()
    y_reg = df.loc[mask,'limite_credito']
    X_reg = X_imp.loc[mask]
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_reg, y_reg)

# ---------------------------
# 5) Pestañas con Streamlit
# ---------------------------
tabs = st.tabs(["Gráficas", "Reporte", "Formulario"])

# --- Gráficas ---
with tabs[0]:
    st.subheader("Importancia de variables")
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=importances.values,
        y=importances.index,
        orientation='h',
        marker_color=importances.values
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC Curve (Hold-out)")
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.25, stratify=y, random_state=42)
    proba_test = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    auc_val = roc_auc_score(y_test, proba_test)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc_val:.3f}'))
    fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='gray')))
    st.plotly_chart(fig2, use_container_width=True)

# --- Reporte ---
with tabs[1]:
    st.subheader("Métricas Hold-out")
    pred_05 = (proba_test >= 0.5).astype(int)
    st.write(f"Accuracy: {accuracy_score(y_test, pred_05):.3f}")
    st.write(f"F1 (macro): {f1_score(y_test, pred_05, average='macro'):.3f}")
    st.write(f"F1 (impago=1): {f1_score(y_test, pred_05, pos_label=1):.3f}")
    st.write("Matriz de Confusión:")
    st.write(confusion_matrix(y_test, pred_05))
    st.write("Reporte de Clasificación:")
    st.text(classification_report(y_test, pred_05, digits=3))

# --- Formulario ---
with tabs[2]:
    st.subheader("Simulador de riesgo")
    w_compras = st.number_input("Valor de las compras realizadas en US$", value=0.0)
    w_pagos = st.number_input("Valor total de los pagos en US$", value=0.0)
    w_atrasos = st.number_input("Número de pagos atrasados", value=0, min_value=0)
    w_plazo = st.number_input("Plazo en días", value=30.0, min_value=0)
    w_rep = st.selectbox("Clasificación del representante legal", ['muy bueno','bueno','regular','malo'])
    w_cli = st.selectbox("Clasificación del cliente", ['muy bueno','bueno','regular','malo'])

    if st.button("Calcular"):
        xnew = pd.DataFrame({
            'compras':[w_compras],
            'pagos':[w_pagos],
            'plazo_credito':[w_plazo],
            'no_pagos_atrasados':[w_atrasos],
            'clasifica_rep_legal_score':[{'muy bueno':3,'bueno':2,'regular':1,'malo':0}[w_rep]],
            'clasificacion_cliente_score':[{'muy bueno':3,'bueno':2,'regular':1,'malo':0}[w_cli]]
        })
        xnew_imp = pd.DataFrame(imputer.transform(xnew), columns=features)
        proba = clf.predict_proba(xnew_imp)[0,1]
        st.write(f"Probabilidad de impago: {proba*100:.2f}%")
        if reg is not None:
            limite = float(reg.predict(xnew_imp)[0])
            st.write(f"Límite de crédito sugerido: {limite:,.0f} US$")
