"""
api/main.py
===========
API FastAPI exposant le pipeline de détection de sentiment vocal.

Endpoint principal :
    POST /predict
        - Reçoit un fichier audio (.wav ou .mp3)
        - Retourne un JSON {transcription, sentiment, confidence, ...}

Lancement local :
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.config import API_CONFIG
from src.pipeline import get_pipeline
from src.utils import get_logger, PipelineError

logger = get_logger(__name__)

app = FastAPI(
    title=API_CONFIG.title,
    description=API_CONFIG.description,
    version=API_CONFIG.version,
)

# --------------------------------------------------------------------------
# Chargement paresseux du pipeline au démarrage du serveur (pas à l'import),
# afin que `uvicorn api.main:app` ne bloque pas si les modèles sont lourds.
# --------------------------------------------------------------------------
@app.on_event("startup")
def load_models_on_startup() -> None:
    logger.info("Démarrage de l'API : préchargement des modèles...")
    get_pipeline()
    logger.info("Modèles prêts. API opérationnelle.")


@app.get("/", tags=["Santé"])
def root():
    """Point de contrôle simple pour vérifier que l'API répond."""
    return {"status": "ok", "service": API_CONFIG.title, "version": API_CONFIG.version}


@app.get("/health", tags=["Santé"])
def health_check():
    """Endpoint de health-check (utile pour Docker/monitoring)."""
    return {"status": "healthy"}


@app.post("/predict", tags=["Prédiction"])
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Analyse un fichier audio et retourne sa transcription + son sentiment.

    - **file**: fichier audio `.wav` ou `.mp3`, 5 minutes maximum

    Retourne un JSON contenant :
    - `transcription`: texte transcrit par le modèle ASR
    - `sentiment`: "positif" | "neutre" | "négatif"
    - `confidence`: score de confiance (0-1)
    - `raw_label`: label brut du modèle de sentiment
    - `processing_time_seconds`: temps de traitement total
    """
    suffix = Path(file.filename).suffix.lower()

    # Écriture temporaire du fichier reçu sur disque (les modèles attendent
    # un chemin de fichier, pas un flux en mémoire).
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        pipeline = get_pipeline()
        result = pipeline.run(tmp_path)
        return JSONResponse(content=result, status_code=200)

    except PipelineError as exc:
        # Erreurs métier connues -> 422 Unprocessable Entity avec message clair
        logger.warning(f"Requête rejetée : {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        # Erreurs inattendues -> 500, sans exposer les détails internes au client
        logger.exception("Erreur interne lors du traitement de la requête")
        raise HTTPException(
            status_code=500, detail="Erreur interne du serveur."
        ) from exc

    finally:
        # Nettoyage du fichier temporaire dans tous les cas
        Path(tmp_path).unlink(missing_ok=True)
