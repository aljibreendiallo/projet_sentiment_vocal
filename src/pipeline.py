"""
pipeline.py
===========
Orchestrateur du pipeline complet :

    Audio -> Prétraitement -> ASR (Wav2Vec2) -> Sentiment (BERT) -> Résultat JSON

Ce module est le point d'entrée unique utilisé à la fois par l'API FastAPI
et par l'interface Gradio, afin de garantir un comportement identique
entre les deux surfaces d'exposition.
"""

import time

from src.preprocessing import load_and_preprocess_audio
from src.asr import Wav2Vec2ASR
from src.sentiment import SentimentAnalyzer
from src.utils import get_logger, PipelineError

logger = get_logger(__name__)


class SentimentVocalPipeline:
    """
    Pipeline de bout en bout : fichier audio -> transcription + sentiment.

    Les modèles sont chargés une seule fois à l'instanciation du pipeline
    (singleton applicatif recommandé côté API/Gradio) pour éviter de payer
    le coût de chargement à chaque requête.
    """

    def __init__(self):
        logger.info("Initialisation du pipeline de sentiment vocal...")
        self.asr = Wav2Vec2ASR()
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("Pipeline prêt.")

    def run(self, file_path: str) -> dict:
        """
        Exécute le pipeline complet sur un fichier audio.

        Args:
            file_path: chemin vers un fichier .wav ou .mp3

        Returns:
            dict: {
                "transcription": str,
                "sentiment": "positif" | "neutre" | "négatif",
                "confidence": float,
                "raw_label": str,
                "processing_time_seconds": float
            }

        Raises:
            PipelineError (ou une sous-classe) en cas d'échec à n'importe
            quelle étape ; l'appelant (API/Gradio) est responsable de
            transformer cette exception en réponse adaptée à l'utilisateur.
        """
        start = time.time()

        try:
            # 1. Prétraitement + validations métier
            signal = load_and_preprocess_audio(file_path)

            # 2. Transcription
            transcription = self.asr.transcribe(signal)

            # 3. Analyse de sentiment
            sentiment_result = self.sentiment_analyzer.analyze(transcription)

        except PipelineError:
            # Erreurs métier connues : on les laisse remonter telles quelles
            raise
        except Exception as exc:
            # Filet de sécurité pour toute erreur inattendue
            logger.exception("Erreur inattendue dans le pipeline")
            raise PipelineError(f"Erreur inattendue du pipeline : {exc}") from exc

        elapsed = round(time.time() - start, 3)
        logger.info(f"Pipeline exécuté en {elapsed}s")

        return {
            "transcription": transcription,
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"],
            "raw_label": sentiment_result["raw_label"],
            "processing_time_seconds": elapsed,
        }


# --------------------------------------------------------------------------
# Singleton applicatif : évite de recharger les modèles à chaque requête
# --------------------------------------------------------------------------
_pipeline_instance: "SentimentVocalPipeline | None" = None


def get_pipeline() -> SentimentVocalPipeline:
    """Retourne une instance unique (lazy-loaded) du pipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SentimentVocalPipeline()
    return _pipeline_instance
