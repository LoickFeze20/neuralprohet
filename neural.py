import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Apple Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease;
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
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 20px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
<div class="main-header">
    <h1>🍎 Apple Stock Predictor </h1>
    <p>Simulation intelligente • Mode démonstration</p>
</div>
""", unsafe_allow_html=True)

# Fonction pour générer des données simulées réalistes
@st.cache_data(ttl=300)  # Cache 5 minutes
def generate_simulated_data():
    """Génère des données Apple simulées basées sur des prix réalistes"""
    np.random.seed(42)
    
    # Prix de départ réaliste pour Apple (environ $175-180)
    start_price = 175 + np.random.randn() * 5
    
    # Générer 365 jours de données
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    
    # Simuler les prix avec une tendance haussière et de la volatilité
    returns = np.random.normal(0.0005, 0.015, 365)  # 0.05% de tendance par jour
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Ajouter quelques variations pour rendre les données réalistes
    noise = np.random.randn(365) * 0.5
    prices = prices + noise
    
    # Simuler OHLC
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    
    # High/Low basés sur la volatilité
    daily_volatility = prices * 0.01  # 1% de volatilité quotidienne
    df['High'] = prices + np.abs(np.random.randn(365) * daily_volatility)
    df['Low'] = prices - np.abs(np.random.randn(365) * daily_volatility)
    df['Open'] = df['Close'].shift(1) + np.random.randn(365) * 0.3
    df['Open'].fillna(df['Close'].iloc[0] * 0.99, inplace=True)
    
    # Volume
    df['Volume'] = np.random.randint(50000000, 150000000, 365)
    
    return df

def calculate_indicators(data):
    """Calculer les indicateurs techniques"""
    df = data.copy()
    
    # Moyennes mobiles
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Remplacer les infinis
    df['RSI'] = df['RSI'].fillna(50)
    
    return df

def generate_predictions(data, days=30):
    """Générer des prédictions"""
    last_price = data['Close'].iloc[-1]
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std()
    avg_return = returns.mean()
    
    # Simulation Monte Carlo
    np.random.seed(42)
    n_simulations = 1000
    simulations = []
    
    for i in range(n_simulations):
        prices = [last_price]
        for j in range(days):
            ret = np.random.normal(avg_return, volatility)
            prices.append(prices[-1] * (1 + ret))
        simulations.append(prices[1:])
    
    simulations = np.array(simulations)
    
    # Moyenne et intervalles
    predictions = np.mean(simulations, axis=0)
    lower = np.percentile(simulations, 2.5, axis=0)
    upper = np.percentile(simulations, 97.5, axis=0)
    
    return predictions, lower, upper

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=100)
    st.markdown("## Menu de navigation")
    
    st.markdown("---")
    
    # LE BOUTON DEMANDÉ
    st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <a href="https://lstm-1.streamlit.app" target="_blank">
            <button class="nav-button">
                🚀 ANALYSE AVANCÉE
            </button>
        </a>
        <p style='font-size: 0.8rem; color: #6c757d; margin-top: 5px;'>
            Accéder au dashboard secondaire
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Paramètres
    st.markdown("### ⚙️ Configuration")
    days_to_predict = st.slider(
        "Horizon de prédiction (jours)",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )
    
    show_indicators = st.checkbox("Afficher les indicateurs", value=True)
    
    st.markdown("---")
    
    # Mode de données
    st.markdown("### ℹ️ Mode")
    st.info("📊 Mode simulation • Données générées")

# Génération des données
with st.spinner("Génération des données de simulation..."):
    data = generate_simulated_data()
    time.sleep(0.5)  # Petit délai pour l'effet de chargement

if data is not None:
    # Calcul des indicateurs
    data = calculate_indicators(data)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    daily_change = (current_price - prev_price) / prev_price * 100
    
    with col1:
        delta_color = "normal" if daily_change >= 0 else "inverse"
        st.metric(
            "Prix Actuel",
            f"${current_price:.2f}",
            f"{daily_change:+.2f}%",
            delta_color=delta_color
        )
    
    with col2:
        st.metric(
            "Plus Haut 52 sem",
            f"${data['High'].tail(252).max():.2f}"
        )
    
    with col3:
        st.metric(
            "Volume",
            f"{data['Volume'].iloc[-1]/1e6:.1f}M"
        )
    
    with col4:
        st.metric(
            "RSI (14)",
            f"{data['RSI'].iloc[-1]:.1f}"
        )
    
    # Génération des prédictions
    predictions, lower, upper = generate_predictions(data, days_to_predict)
    
    # Graphique
    st.markdown("## 📈 Analyse et Prédictions")
    
    fig = go.Figure()
    
    # Prix historiques (6 mois)
    fig.add_trace(go.Scatter(
        x=data.index[-180:],
        y=data['Close'][-180:],
        mode='lines',
        name='Historique',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
    ))
    
    # Moyennes mobiles
    if show_indicators:
        fig.add_trace(go.Scatter(
            x=data.index[-180:],
            y=data['MA50'][-180:],
            mode='lines',
            name='MA50',
            line=dict(color='orange', width=1, dash='dash'),
            hovertemplate='MA50: $%{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index[-180:],
            y=data['MA200'][-180:],
            mode='lines',
            name='MA200',
            line=dict(color='red', width=1, dash='dash'),
            hovertemplate='MA200: $%{y:.2f}<extra></extra>'
        ))
    
    # Prédictions
    future_dates = pd.date_range(
        start=data.index[-1] + timedelta(days=1), 
        periods=days_to_predict
    )
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Prédiction',
        line=dict(color='#ff4b4b', width=3),
        marker=dict(size=6),
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
        name='Intervalle 95%',
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title={
            'text': f"Prédiction sur {days_to_predict} jours avec intervalle de confiance 95%",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
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
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des prédictions
    st.markdown("## 📅 Détail des prédictions")
    
    pred_df = pd.DataFrame({
        'Date': future_dates.strftime('%d/%m/%Y'),
        'Prix Prévu': predictions,
        'Min (95%)': lower,
        'Max (95%)': upper,
        'Variation': ((predictions / current_price - 1) * 100)
    })
    
    # Formatage
    pred_df['Prix Prévu'] = pred_df['Prix Prévu'].apply(lambda x: f"${x:.2f}")
    pred_df['Min (95%)'] = pred_df['Min (95%)'].apply(lambda x: f"${x:.2f}")
    pred_df['Max (95%)'] = pred_df['Max (95%)'].apply(lambda x: f"${x:.2f}")
    pred_df['Variation'] = pred_df['Variation'].apply(lambda x: f"{x:+.1f}%")
    
    st.dataframe(
        pred_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": "Date",
            "Prix Prévu": "Prix Prévu",
            "Min (95%)": "Min",
            "Max (95%)": "Max",
            "Variation": "Variation"
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
        if st.button("🔄 Nouvelle simulation", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        st.info(f"Simulation: {datetime.now().strftime('%H:%M:%S')}")
    
    # Message d'info sur le mode simulation
    st.info("""
    ℹ️ **Mode simulation activé** - Les données sont générées localement pour éviter les limitations de l'API Yahoo Finance.
    Les prix sont réalistes (autour de $175) mais ne reflètent pas les données en temps réel.
    """)

else:
    st.error("Erreur de génération des données")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2024 Apple Stock Predictor • Mode simulation</p>
    <p style='font-size: 0.8rem;'>⚠️ Les prédictions sont à but éducatif uniquement</p>
</div>
""", unsafe_allow_html=True)
