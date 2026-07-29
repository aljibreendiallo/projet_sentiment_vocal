#!/usr/bin/env bash
#
# curl_example.sh
# ================
# Exemple d'appel de l'API POST /predict avec curl.
#
# Prérequis : l'API doit être lancée (voir README.md), par défaut sur
# http://localhost:8000
#
# Usage :
#   ./curl_example.sh chemin/vers/fichier.wav

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
AUDIO_FILE="${1:-tests/audio_samples/exemple_positif.wav}"

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Erreur : fichier introuvable -> $AUDIO_FILE"
    exit 1
fi

echo "Envoi de '$AUDIO_FILE' vers ${API_URL}/predict ..."

curl -X POST "${API_URL}/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@${AUDIO_FILE}" \
     | python3 -m json.tool
