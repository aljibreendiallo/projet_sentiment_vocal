# 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux

**Module :** Deep Learning 2 — Dakar Institute of Technology (DIT)
**Pipeline :** Audio → ASR (Wav2Vec 2.0) → Analyse de sentiment (BERT) → JSON

---

## 1. Présentation

Ce projet implémente un pipeline automatisé qui :

1. Transcrit un appel vocal client (`.wav` / `.mp3`) en texte via un modèle **Wav2Vec 2.0** ;
2. Analyse le sentiment de la transcription (**positif / neutre / négatif**) via un modèle **BERT** ;
3. Expose ce pipeline via une **API REST (FastAPI)** et une **interface Gradio**.

```
Audio (.wav/.mp3) → Prétraitement (16kHz, mono) → ASR (Wav2Vec2) → Sentiment (BERT) → {transcription, sentiment, confidence}
```

---

## 2. Architecture du projet

```
projet_sentiment_vocal/
├── src/
│   ├── config.py          # Paramètres centralisés (modèles, contraintes audio)
│   ├── preprocessing.py   # Chargement / nettoyage audio + validations
│   ├── asr.py              # Module Wav2Vec2 (audio -> texte)
│   ├── sentiment.py        # Module BERT (texte -> sentiment)
│   ├── pipeline.py         # Orchestrateur du pipeline complet
│   └── utils.py            # Exceptions métier + logging
├── api/
│   └── main.py              # API FastAPI (POST /predict)
├── app_gradio.py            # Interface Gradio
├── tests/
│   ├── test_pipeline.py     # Tests unitaires (validations, règles métier)
│   └── audio_samples/       # 3 fichiers de démo (positif/négatif/neutre)
├── evaluation/
│   ├── evaluate.py          # Script WER + Accuracy/F1 (bonus)
│   └── dataset_example.csv  # Exemple de jeu de données annoté
├── examples/
│   ├── curl_example.sh      # Exemple d'appel API en curl
│   └── call_api.py          # Exemple d'appel API en Python
├── Dockerfile                # Conteneurisation (bonus)
├── docker-compose.yml         # Lance API + Gradio ensemble
├── .github/workflows/ci.yml   # CI : exécution automatique des tests
├── requirements.txt
├── pytests.ini
└── README.md
```

---

## 3. Modèles utilisés et justification

| Tâche | Modèle | Lien Hugging Face | Justification |
|---|---|---|---|
| ASR | `jonatasgrosman/wav2vec2-large-xlsr-53-french` | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french | Wav2Vec2 XLSR-53 **fine-tuné spécifiquement sur le français**, baseline ASR FR la plus utilisée sur Hugging Face, bon compromis qualité/disponibilité pour un projet académique. |
| Sentiment | `nlptown/bert-base-multilingual-uncased-sentiment` | https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment | BERT multilingue fine-tuné sur des avis clients (notation 1 à 5 étoiles), couvre nativement le français. La sortie 5 classes est **remappée** vers les 3 classes métier : `1-2★ → négatif`, `3★ → neutre`, `4-5★ → positif`. |

**Alternative envisagée et écartée :** `tblard/tf-allocine` (CamemBERT fine-tuné, binaire positif/négatif) — écarté car il ne fournit pas nativement de classe neutre, requise par le cahier des charges.

---

## 4. Installation

### Prérequis
- Python ≥ 3.9
- `ffmpeg` et `libsndfile1` (décodage audio)

### Étapes

```bash
git clone <url_du_depot>
cd projet_sentiment_vocal

python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

Sur Ubuntu/Debian, si `ffmpeg`/`libsndfile1` sont absents :
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
```

---

## 5. Utilisation

### 5.1 Lancer l'API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive Swagger : http://localhost:8000/docs

Appel de test :
```bash
./examples/curl_example.sh tests/audio_samples/exemple_positif.wav
# ou
python examples/call_api.py tests/audio_samples/exemple_positif.wav
```

Réponse type :
```json
{
  "transcription": "je suis très satisfait de votre service",
  "sentiment": "positif",
  "confidence": 0.87,
  "raw_label": "5 stars",
  "processing_time_seconds": 2.14
}
```

### 5.2 Lancer l'interface Gradio

```bash
python app_gradio.py
```

Interface disponible sur http://localhost:7860 — upload ou enregistrement micro, transcription affichée en temps réel, sentiment + score de confiance.

### 5.3 Avec Docker

```bash
docker compose up --build
```
- API disponible sur `http://localhost:8000`
- Gradio disponible sur `http://localhost:7860`

### 5.4 Exécuter les tests

```bash
pytest tests/ -v --cov=src
```

### 5.5 Évaluation quantitative (bonus)

```bash
python evaluation/evaluate.py --dataset evaluation/dataset_example.csv
```
Calcule le **WER** (transcription) et l'**accuracy / F1 macro** (sentiment) sur le jeu de données annoté.

---

## 6. Gestion des erreurs

Le pipeline lève des exceptions métier dédiées (`src/utils.py`), traduites par l'API en réponses HTTP explicites :

| Cas | Exception | Code HTTP |
|---|---|---|
| Format non supporté (ni `.wav` ni `.mp3`) | `InvalidAudioFormatError` | 422 |
| Fichier vide / illisible | `EmptyAudioError` | 422 |
| Audio trop court | `AudioTooShortError` | 422 |
| Audio > 5 minutes | `AudioTooLongError` | 422 |
| Audio silencieux | `SilentAudioError` | 422 |
| Fichier trop volumineux | `FileSizeLimitExceededError` | 422 |
| Échec transcription / sentiment | `TranscriptionError` / `SentimentAnalysisError` | 422 |
| Erreur inattendue | — | 500 |

---

## 7. Limites connues du système

- Le modèle ASR n'a pas été évalué sur des données d'appels téléphoniques réels (bruit de fond, qualité GSM) : les performances peuvent être inférieures à celles annoncées sur des benchmarks propres (Common Voice).
- Le mapping 5 étoiles → 3 classes de sentiment est une **heuristique** ; il n'a pas été calibré/validé sur un jeu de données d'appels clients réel.
- Durée maximale de traitement fixée à 5 minutes par fichier, conformément au cahier des charges — au-delà, le fichier est rejeté plutôt que découpé.
- Les 3 fichiers de démonstration fournis par défaut dans `tests/audio_samples/` sont des **tonalités synthétiques** de test (voir `tests/audio_samples/README.md`) et doivent être remplacés par de vrais enregistrements vocaux pour une démonstration métier probante.
- Pas de gestion de la diarisation (plusieurs locuteurs) : le pipeline traite l'audio comme un flux mono-locuteur.

---

## 8. Reproductibilité

- Versions figées des dépendances majeures dans `requirements.txt`.
- CI GitHub Actions (`.github/workflows/ci.yml`) exécutant les tests à chaque push/PR.
- Modèles chargés directement depuis Hugging Face Hub (pas de poids ré-entraînés localement), garantissant une reproductibilité complète à partir des identifiants de modèles indiqués en section 3.

---

## 9. Auteur

Alpha Oumar DIALLO  Master 2 AI/Deep Learning, Dakar Institute of Technology (DIT)
Projet d'examen — Module_Deep_Learning_2 2026
=======
