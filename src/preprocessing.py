"""
preprocessing.py
=================
Chargement, validation et nettoyage des fichiers audio avant transcription.

Étapes :
    1. Validation de l'extension (.wav / .mp3) et de la taille du fichier
    2. Chargement (librosa gère nativement wav/mp3 via audioread/soundfile)
    3. Conversion mono
    4. Rééchantillonnage à 16 kHz
    5. Normalisation de l'amplitude
    6. Détection des cas invalides (vide, trop court, trop long, silencieux)
"""

from pathlib import Path

import numpy as np
import librosa

from src.config import AUDIO_CONFIG
from src.utils import (
    get_logger,
    InvalidAudioFormatError,
    EmptyAudioError,
    AudioTooShortError,
    AudioTooLongError,
    SilentAudioError,
    FileSizeLimitExceededError,
)

logger = get_logger(__name__)


def validate_file_format(file_path: str) -> None:
    """Vérifie que l'extension du fichier est supportée."""
    ext = Path(file_path).suffix.lower()
    if ext not in AUDIO_CONFIG.allowed_extensions:
        raise InvalidAudioFormatError(
            f"Format '{ext}' non supporté. Formats acceptés : "
            f"{AUDIO_CONFIG.allowed_extensions}"
        )


def validate_file_size(file_path: str) -> None:
    """Vérifie que le fichier ne dépasse pas la taille maximale autorisée."""
    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    if size_mb > AUDIO_CONFIG.max_file_size_mb:
        raise FileSizeLimitExceededError(
            f"Fichier de {size_mb:.1f} Mo, limite autorisée : "
            f"{AUDIO_CONFIG.max_file_size_mb} Mo."
        )
    if size_mb <= 0:
        raise EmptyAudioError("Le fichier audio est vide (0 octet).")


def normalize_amplitude(signal: np.ndarray) -> np.ndarray:
    """Normalise le signal dans l'intervalle [-1, 1] (peak normalization)."""
    peak = np.max(np.abs(signal)) if signal.size > 0 else 0.0
    if peak == 0:
        return signal
    return signal / peak


def is_silent(signal: np.ndarray) -> bool:
    """Détecte un audio silencieux via l'énergie RMS du signal."""
    if signal.size == 0:
        return True
    rms = np.sqrt(np.mean(np.square(signal)))
    return rms < AUDIO_CONFIG.silence_rms_threshold


def load_and_preprocess_audio(file_path: str) -> np.ndarray:
    """
    Pipeline complet de prétraitement d'un fichier audio.

    Args:
        file_path: chemin vers un fichier .wav ou .mp3

    Returns:
        np.ndarray: signal mono, 16 kHz, normalisé, prêt pour Wav2Vec2

    Raises:
        InvalidAudioFormatError, FileSizeLimitExceededError, EmptyAudioError,
        AudioTooShortError, AudioTooLongError, SilentAudioError
    """
    logger.info(f"Prétraitement du fichier : {file_path}")

    # 1. Validations "à froid" (sans décoder l'audio)
    validate_file_format(file_path)
    validate_file_size(file_path)

    # 2. Chargement : librosa force mono + resample en une seule passe
    try:
        signal, sr = librosa.load(
            file_path,
            sr=AUDIO_CONFIG.target_sample_rate,
            mono=True,
        )
    except Exception as exc:
        raise EmptyAudioError(
            f"Impossible de décoder le fichier audio : {exc}"
        ) from exc

    if signal is None or signal.size == 0:
        raise EmptyAudioError("Le fichier audio ne contient aucune donnée exploitable.")

    # 3. Validation de durée
    duration = len(signal) / sr
    logger.info(f"Durée détectée : {duration:.2f}s @ {sr}Hz")

    if duration < AUDIO_CONFIG.min_duration_seconds:
        raise AudioTooShortError(
            f"Audio trop court ({duration:.2f}s), minimum requis : "
            f"{AUDIO_CONFIG.min_duration_seconds}s."
        )
    if duration > AUDIO_CONFIG.max_duration_seconds:
        raise AudioTooLongError(
            f"Audio trop long ({duration:.1f}s), maximum autorisé : "
            f"{AUDIO_CONFIG.max_duration_seconds}s (5 minutes)."
        )

    # 4. Détection de silence
    if is_silent(signal):
        raise SilentAudioError(
            "Le fichier audio est jugé silencieux (aucune énergie détectée)."
        )

    # 5. Normalisation
    signal = normalize_amplitude(signal)

    logger.info("Prétraitement terminé avec succès.")
    return signal
