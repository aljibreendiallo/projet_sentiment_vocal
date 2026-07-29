"""
call_api.py
===========
Exemple d'appel programmatique de l'API POST /predict en Python.

Usage :
    python examples/call_api.py chemin/vers/fichier.wav
"""

import sys
import json
from pathlib import Path

import requests

API_URL = "http://localhost:8000/predict"


def predict_sentiment(audio_path: str, api_url: str = API_URL) -> dict:
    """Envoie un fichier audio à l'API et retourne la réponse JSON."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {audio_path}")

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        response = requests.post(api_url, files=files, timeout=120)

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    default_file = "tests/audio_samples/exemple_positif.wav"
    audio_file = sys.argv[1] if len(sys.argv) > 1 else default_file

    print(f"Analyse de : {audio_file}")
    try:
        result = predict_sentiment(audio_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except requests.exceptions.HTTPError as exc:
        print(f"Erreur API ({exc.response.status_code}) : {exc.response.text}")
    except Exception as exc:
        print(f"Erreur : {exc}")
