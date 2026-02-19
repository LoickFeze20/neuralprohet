import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from neuralprophet import NeuralProphet
from datetime import datetime, timedelta

st.set_page_config(page_title="Apple Predictor", layout="wide")
st.title("🍎 Apple Stock Predictor")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    jours = st.slider("Jours à prédire", 1, 90, 30)
    annees = st.slider("Années d'historique", 1, 3, 2)
    if st.button("🚀 LANCER LA PRÉDICTION", type="primary"):
        st.session_state.run = True

# Charger données
@st.cache_data
def load_data(annees):
    apple = yf.Ticker("AAPL")
    hist = apple.history(period=f"{annees}y")
    hist.reset_index(inplace=True)
    return hist

data = load_data(annees)
st.metric("Prix actuel", f"${data['Close'].iloc[-1]:.2f}")

# Graphique
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='AAPL'))
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

if 'run' in st.session_state and st.session_state.run:
    with st.spinner("Calcul des prédictions..."):
        # Préparer données
        df = pd.DataFrame({'ds': data['Date'], 'y': data['Close']})
        
        # Modèle
        model = NeuralProphet(
            n_forecasts=jours,
            n_lags=60,
            yearly_seasonality=True,
            weekly_seasonality=True
        )
        
        # Entraînement
        model.fit(df, freq='D')
        
        # Prédictions
        future = model.make_future_dataframe(df, periods=jours)
        forecast = model.predict(future)
        
        # Graphique résultat
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data['Date'].tail(180), y=data['Close'].tail(180), 
                                  mode='lines', name='Historique'))
        fig2.add_trace(go.Scatter(x=forecast['ds'].tail(jours), y=forecast['yhat1'].tail(jours),
                                  mode='lines+markers', name='Prédictions'))
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Tableau
        results = pd.DataFrame({
            'Date': forecast['ds'].tail(jours).dt.strftime('%Y-%m-%d'),
            'Prix': forecast['yhat1'].tail(jours).round(2)
        })
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        st.session_state.run = False
