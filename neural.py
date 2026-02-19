import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from neuralprophet import NeuralProphet
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="Apple Predictor", layout="wide")
st.title("🍎 Prédictions Apple - NeuralProphet")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    jours = st.slider("Jours à prédire", 1, 90, 30)
    annees = st.slider("Années d'historique", 1, 5, 2)
    lags = st.slider("Lags", 10, 90, 60)
    epochs = st.slider("Époques", 10, 100, 30)
    
    if st.button("🚀 LANCER", type="primary"):
        st.session_state.run = True

# Chargement données
@st.cache_data
def load_data(annees):
    apple = yf.Ticker("AAPL")
    hist = apple.history(period=f"{annees}y")
    hist.reset_index(inplace=True)
    return hist

hist = load_data(annees)

# Métriques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prix actuel", f"${hist['Close'].iloc[-1]:.2f}")
col2.metric("Volume", f"{hist['Volume'].iloc[-1]/1e6:.1f}M")
col3.metric("Date début", hist['Date'].iloc[0].strftime('%Y-%m-%d'))
col4.metric("Jours", len(hist))

# Graphique historique
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='AAPL'))
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Prédictions
if 'run' in st.session_state and st.session_state.run:
    with st.spinner("Entraînement en cours..."):
        
        # Préparer données
        df = pd.DataFrame({'ds': hist['Date'], 'y': hist['Close']})
        
        # Créer et entraîner modèle
        model = NeuralProphet(
            n_forecasts=jours,
            n_lags=lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            epochs=epochs
        )
        
        # Barre progression
        progress = st.progress(0)
        for i in range(epochs):
            model.fit(df, freq='D', epochs=1, progress=None)
            progress.progress((i+1)/epochs)
        
        # Prédictions
        future = model.make_future_dataframe(df, periods=jours)
        forecast = model.predict(future)
        
        # Graphique prédictions
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist['Date'].tail(180), y=hist['Close'].tail(180), 
                                  mode='lines', name='Historique'))
        fig2.add_trace(go.Scatter(x=forecast['ds'].tail(jours), y=forecast['yhat1'].tail(jours),
                                  mode='lines+markers', name='Prédictions', 
                                  line=dict(dash='dash', color='red')))
        fig2.add_vline(x=hist['Date'].iloc[-1], line_dash="dash")
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Résultats
        st.subheader("📋 Résultats")
        results = pd.DataFrame({
            'Date': forecast['ds'].tail(jours).dt.strftime('%Y-%m-%d'),
            'Prix': forecast['yhat1'].tail(jours).round(2)
        })
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        # Téléchargement
        csv = results.to_csv(index=False)
        st.download_button("📥 Télécharger CSV", csv, "predictions.csv")
        
        st.session_state.run = False
