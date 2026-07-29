"""
utils.py
========
Exceptions métier du pipeline + utilitaires transverses (logging).

Centraliser les exceptions permet à l'API de les attraper précisément
et de renvoyer des codes HTTP / messages adaptés, plutôt que de laisser
remonter des stack traces génériques.
"""

import logging
import sys

from src.config import LOG_DIR


# --------------------------------------------------------------------------
# Exceptions métier
# --------------------------------------------------------------------------
class PipelineError(Exception):
    """Classe de base pour toutes les erreurs métier du pipeline."""


class InvalidAudioFormatError(PipelineError):
    """Levée quand l'extension du fichier n'est pas supportée (.wav / .mp3)."""


class EmptyAudioError(PipelineError):
    """Levée quand le fichier audio est vide ou illisible."""


class AudioTooShortError(PipelineError):
    """Levée quand la durée audio est inférieure au minimum exploitable."""


class AudioTooLongError(PipelineError):
    """Levée quand la durée audio dépasse la limite (5 minutes)."""


class SilentAudioError(PipelineError):
    """Levée quand l'audio est jugé silencieux (RMS sous le seuil)."""


class TranscriptionError(PipelineError):
    """Levée quand le modèle ASR échoue à produire une transcription exploitable."""


class SentimentAnalysisError(PipelineError):
    """Levée quand le modèle de sentiment échoue sur le texte transcrit."""


class FileSizeLimitExceededError(PipelineError):
    """Levée quand le fichier dépasse la taille maximale autorisée par l'API."""


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger configuré (console + fichier), réutilisable dans
    tous les modules via `get_logger(__name__)`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # évite les handlers dupliqués si appelé plusieurs fois
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
