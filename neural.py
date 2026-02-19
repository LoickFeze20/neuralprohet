import subprocess
import sys

# Forcer l'installation de setuptools
subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools==69.0.0"])

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from neuralprophet import NeuralProphet
from datetime import datetime, timedelta
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Apple NeuralProphet - Entraînement Direct",
    page_icon="🍎",
    layout="wide"
)

# Style CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
    }
    .stButton > button {
        background-color: #1E3C72;
        color: white;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='main-header'>
    <h1>🍎 Apple Stock Predictor</h1>
    <p>Entraînement NeuralProphet en direct</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Période d'entraînement
    st.subheader("📅 Période d'entraînement")
    annees = st.slider("Années d'historique", 1, 5, 2)
    
    # Paramètres du modèle
    st.subheader("🧠 Paramètres du modèle")
    n_forecasts = st.slider("Jours à prédire (n_forecasts)", 1, 90, 30)
    n_lags = st.slider("Jours d'historique (n_lags)", 10, 90, 60)
    epochs = st.slider("Nombre d'époques", 10, 200, 50)
    
    # Bouton d'entraînement
    train_button = st.button("🚀 LANCER L'ENTRAÎNEMENT", type="primary")

# Chargement des données
@st.cache_data
def load_data(annees):
    with st.spinner("Chargement des données Apple..."):
        apple = yf.Ticker("AAPL")
        hist = apple.history(period=f"{annees}y")
        hist.reset_index(inplace=True)
        
        # Préparer pour NeuralProphet
        df = pd.DataFrame()
        df['ds'] = pd.to_datetime(hist['Date'])
        df['y'] = hist['Close'].values
        
        return df, hist

df, hist_raw = load_data(annees)

# Affichage des données
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Prix actuel", f"${df['y'].iloc[-1]:.2f}")
with col2:
    st.metric("Date début", df['ds'].iloc[0].strftime('%Y-%m-%d'))
with col3:
    st.metric("Date fin", df['ds'].iloc[-1].strftime('%Y-%m-%d'))
with col4:
    st.metric("Jours de données", len(df))

# Graphique des données
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['ds'],
    y=df['y'],
    mode='lines',
    name='Prix Apple',
    line=dict(color='#1E3C72', width=2)
))
fig.update_layout(
    title="📈 Données historiques Apple",
    height=400,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

# Entraînement et prédictions
if train_button:
    with st.spinner(f"Entraînement du modèle sur {len(df)} jours..."):
        
        # Création du modèle
        model = NeuralProphet(
            n_forecasts=n_forecasts,
            n_lags=n_lags,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=epochs,
            learning_rate=0.1
        )
        
        # Ajouter une barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fonction de callback pour la progression
        class ProgressCallback:
            def __init__(self):
                self.epoch = 0
            def on_epoch_end(self, epoch, logs=None):
                self.epoch = epoch + 1
                progress_bar.progress(self.epoch / epochs)
                status_text.text(f"Époque {self.epoch}/{epochs}")
        
        # Entraînement
        metrics = model.fit(df, freq='D', progress_callback=ProgressCallback())
        
        status_text.text("✅ Entraînement terminé!")
        
        # Prédictions futures
        st.subheader(f"🔮 Prédictions sur {n_forecasts} jours")
        
        # Créer le dataframe futur
        future = model.make_future_dataframe(df, periods=n_forecasts)
        forecast = model.predict(future)
        
        # Extraire les prédictions futures
        future_forecast = forecast.tail(n_forecasts).copy()
        
        # Graphique des prédictions
        fig2 = go.Figure()
        
        # Historique (derniers 180 jours)
        hist_dates = df['ds'].tail(180)
        hist_values = df['y'].tail(180)
        
        fig2.add_trace(go.Scatter(
            x=hist_dates,
            y=hist_values,
            mode='lines',
            name='Historique',
            line=dict(color='#1E3C72', width=2)
        ))
        
        # Prédictions
        fig2.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat1'],
            mode='lines+markers',
            name='Prédictions',
            line=dict(color='#DC143C', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Ligne de séparation
        fig2.add_vline(
            x=df['ds'].iloc[-1],
            line_dash="dash",
            line_color="gray"
        )
        
        fig2.update_layout(
            height=500,
            hovermode='x unified',
            template='plotly_white',
            title=f"Prédictions Apple sur {n_forecasts} jours",
            xaxis_title="Date",
            yaxis_title="Prix ($)"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Tableau des prédictions
        st.subheader("📋 Détail des prédictions")
        
        resultat = pd.DataFrame({
            'Date': future_forecast['ds'].dt.strftime('%Y-%m-%d'),
            'Prix ($)': future_forecast['yhat1'].round(2),
            'Variation ($)': future_forecast['yhat1'].diff().round(2),
            'Variation (%)': (future_forecast['yhat1'].pct_change() * 100).round(2)
        }).fillna(0)
        
        st.dataframe(resultat, use_container_width=True, hide_index=True)
        
        # Statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prix minimum", f"${future_forecast['yhat1'].min():.2f}")
        with col2:
            st.metric("Prix maximum", f"${future_forecast['yhat1'].max():.2f}")
        with col3:
            trend = future_forecast['yhat1'].iloc[-1] - future_forecast['yhat1'].iloc[0]
            st.metric("Tendance", f"${trend:+.2f}")
        
        # Sauvegarde du modèle (optionnel)
        csv = resultat.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les prédictions (CSV)",
            data=csv,
            file_name=f"apple_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Afficher les métriques d'entraînement
        with st.expander("📊 Métriques d'entraînement"):
            if isinstance(metrics, pd.DataFrame):
                st.line_chart(metrics[['SmoothL1Loss']])
            else:
                st.write("Métriques disponibles:", metrics)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🍎 Entraînement NeuralProphet en direct - Pas de modèle pré-entraîné nécessaire</p>
</div>
""", unsafe_allow_html=True)

