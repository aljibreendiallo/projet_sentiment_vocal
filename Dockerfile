# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Dépendances système nécessaires à librosa/soundfile (décodage audio) et build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python d'abord (meilleur cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY api/ ./api/
COPY app_gradio.py .
COPY tests/audio_samples/ ./tests/audio_samples/

# Répertoire de logs
RUN mkdir -p logs

# Utilisateur non-root pour la sécurité
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 7860

# Par défaut, lance l'API. Le docker-compose.yml surcharge la commande
# pour le service Gradio.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
