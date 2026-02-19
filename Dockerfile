FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers
COPY requirements.txt .
COPY neural.py .
COPY apple_neural.pt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Commande pour lancer l'app
CMD ["streamlit", "run", "neural.py", "--server.port=8501", "--server.address=0.0.0.0"]
