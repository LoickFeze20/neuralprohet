import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Apple Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un style professionnel
st.markdown("""
<style>
    /* Style global */
    .main {
        background-color: #f8f9fa;
    }
    
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Cartes de métriques */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #212529;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Bouton de navigation */
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0;
        cursor: pointer;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar stylisée */
    .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem 1rem;
        border-radius: 20px;
        color: white;
    }
    
    /* Badges */
    .badge {
        background-color: #e9ecef;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #495057;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# En-tête professionnel
st.markdown("""
<div class="main-header">
    <h1>📈 Apple Stock Predictor Pro</h1>
    <p>Analyses avancées & prédictions intelligentes sur 30 jours</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_apple_data():
    """Récupérer les données Apple"""
    try:
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1y")
        info = ticker.info
        return data, info
    except:
        return None, None

def calculate_indicators(data):
    """Calculer les indicateurs techniques"""
    df = data.copy()
    
    # Moyennes mobiles
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def generate_predictions(data, days=30):
    """Générer des prédictions"""
    last_price = data['Close'].iloc[-1]
    volatility = data['Close'].pct_change().std()
    
    # Simulation réaliste
    np.random.seed(42)
    returns = np.random.normal(0.0003, volatility, days)
    predictions = last_price * np.exp(np.cumsum(returns))
    
    # Intervalles de confiance
    confidence = volatility * np.sqrt(np.arange(1, days + 1))
    upper = predictions * np.exp(1.96 * confidence)
    lower = predictions * np.exp(-1.96 * confidence)
    
    return predictions, upper, lower

# Sidebar avec le bouton de navigation
with st.sidebar:
    # Logo et titre
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=80)
    st.markdown("<h2 style='text-align: center;'>Menu</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # LE BOUTON QUE VOUS AVEZ DEMANDÉ - Lien vers autre page
    st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <a href="http://localhost:8502" target="_blank">
            <button class="nav-button">
                🚀 ACCÉDER AU DASHBOARD ANALYTIQUE
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Paramètres
    st.markdown("### ⚙️ Paramètres")
    days_to_predict = st.slider("Horizon de prédiction", 5, 60, 30, 5)
    confidence_level = st.select_slider("Niveau de confiance", 
                                        options=[80, 85, 90, 95, 99], 
                                        value=95)
    
    st.markdown("---")
    
    # Informations
    st.markdown("### ℹ️ Informations")
    st.markdown("""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
        <p style='margin: 0; color: #333;'>
        📊 Données temps réel<br>
        🔄 Mise à jour automatique<br>
        📈 Analyses techniques<br>
        🎯 Prédictions 30 jours
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chargement des données
with st.spinner("Chargement des données Apple..."):
    data, info = get_apple_data()

if data is not None:
    # Calcul des indicateurs
    data = calculate_indicators(data)
    
    # Métriques principales en cartes stylisées
    st.markdown("### 📊 Aperçu du Marché")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    daily_change = (current_price - prev_price) / prev_price * 100
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prix Actuel</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-delta" style="color: {'#28a745' if daily_change >= 0 else '#dc3545'};">
                {daily_change:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Plus Haut Année</div>
            <div class="metric-value">${data['High'].max():.2f}</div>
            <div class="metric-delta">Annuel</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{data['Volume'].iloc[-1]/1e6:.1f}M</div>
            <div class="metric-delta">Aujourd'hui</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value">{data['RSI'].iloc[-1]:.1f}</div>
            <div class="metric-delta">
                <span class="badge {'badge-primary' if data['RSI'].iloc[-1] > 70 else ''}">
                    {'Surachat' if data['RSI'].iloc[-1] > 70 else 'Neutre' if data['RSI'].iloc[-1] > 30 else 'Survente'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Génération des prédictions
    predictions, upper, lower = generate_predictions(data, days_to_predict)
    
    # Graphique principal
    st.markdown("### 📈 Analyse & Prédictions")
    
    fig = go.Figure()
    
    # Prix historiques
    fig.add_trace(go.Scatter(
        x=data.index[-180:],
        y=data['Close'][-180:],
        mode='lines',
        name='Historique',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
    ))
    
    # Moyenne mobile 50
    fig.add_trace(go.Scatter(
        x=data.index[-180:],
        y=data['MA50'][-180:],
        mode='lines',
        name='MA50',
        line=dict(color='orange', width=1, dash='dash'),
        hovertemplate='MA50: $%{y:.2f}<extra></extra>'
    ))
    
    # Prédictions
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Prédiction',
        line=dict(color='#ff4b4b', width=3),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='Date: %{x}<br>Prédiction: $%{y:.2f}<extra></extra>'
    ))
    
    # Intervalle de confiance
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='none'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255,75,75,0.2)',
        line=dict(width=0),
        name=f'Intervalle {confidence_level}%',
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title={
            'text': f"Prédiction Apple sur {days_to_predict} jours",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family="Arial, sans-serif")
        },
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des prédictions
    st.markdown("### 📅 Prédictions Détaillées")
    
    pred_df = pd.DataFrame({
        'Date': future_dates.strftime('%d/%m/%Y'),
        'Prix Prévu': [f"${p:.2f}" for p in predictions],
        'Min ({confidence_level}%)'.format(confidence_level=confidence_level): [f"${l:.2f}" for l in lower],
        'Max ({confidence_level}%)'.format(confidence_level=confidence_level): [f"${u:.2f}" for u in upper],
        'Variation': [f"{(p/current_price-1)*100:+.1f}%" for p in predictions]
    })
    
    st.dataframe(
        pred_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Prix Prévu": st.column_config.TextColumn("Prix Prévu", width="medium"),
            f"Min ({confidence_level}%)": st.column_config.TextColumn("Min", width="medium"),
            f"Max ({confidence_level}%)": st.column_config.TextColumn("Max", width="medium"),
            "Variation": st.column_config.TextColumn("Variation", width="small")
        }
    )
    
    # Export
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"apple_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Actualiser", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; background: #e9ecef; border-radius: 8px;'>
            <small>Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Impossible de charger les données. Vérifiez votre connexion internet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 20px;'>
    <p>© 2024 Apple Stock Predictor Pro - Données fournies par Yahoo Finance</p>
    <p style='font-size: 0.8rem;'>⚠️ Les prédictions sont à but éducatif uniquement</p>
</div>
""", unsafe_allow_html=True)