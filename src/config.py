"""
config.py
=========
Paramètres centralisés du pipeline de détection de sentiment vocal.

Toute constante "métier" (modèles, contraintes audio, seuils) est définie
ici et NULLE PART ailleurs, afin d'avoir une source unique de vérité.
"""

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Contraintes audio
# --------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """Contraintes et paramètres de prétraitement audio."""

    target_sample_rate: int = 16_000          # Wav2Vec2 attend du 16 kHz
    target_channels: int = 1                  # mono
    max_duration_seconds: int = 300            # 5 minutes (cahier des charges)
    min_duration_seconds: float = 0.3          # rejette les fichiers quasi vides
    allowed_extensions: tuple = (".wav", ".mp3")
    silence_rms_threshold: float = 1e-4        # en dessous -> considéré silencieux
    max_file_size_mb: int = 25                 # garde-fou anti-DOS sur l'API


# --------------------------------------------------------------------------
# Modèles Hugging Face
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """
    Modèles pré-entraînés utilisés par le pipeline.

    ASR :
        jonatasgrosman/wav2vec2-large-xlsr-53-french
        -> Wav2Vec2 XLSR-53 fine-tuné spécifiquement sur le français,
           bon compromis performance / disponibilité publique, largement
           utilisé comme baseline ASR FR sur Hugging Face.

    Sentiment :
        nlptown/bert-base-multilingual-uncased-sentiment
        -> BERT multilingue fine-tuné sur des avis clients (1 à 5 étoiles),
           couvre le français nativement. On remappe la sortie 5 classes
           vers les 3 classes métier demandées (positif / neutre / négatif) :
             1-2 étoiles -> négatif
             3 étoiles   -> neutre
             4-5 étoiles -> positif
           Alternative envisagée : CamemBERT fine-tuné binaire (tblard/tf-allocine),
           écartée car elle ne fournit pas de classe neutre.
    """

    asr_model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"

    # Mapping des labels bruts du modèle de sentiment (5 étoiles) vers les
    # 3 classes métier attendues par le cahier des charges.
    star_to_label: dict = field(
        default_factory=lambda: {
            "1 star": "négatif",
            "2 stars": "négatif",
            "3 stars": "neutre",
            "4 stars": "positif",
            "5 stars": "positif",
        }
    )

    device: str = "cpu"  # "cuda" si un GPU est disponible, détecté dynamiquement ailleurs


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "API de Détection de Sentiment Vocal"
    description: str = (
        "Pipeline Audio -> ASR (Wav2Vec2) -> Sentiment (BERT) pour l'analyse "
        "automatique des appels clients."
    )
    version: str = "1.0.0"


# Instances uniques réutilisées dans tout le projet
AUDIO_CONFIG = AudioConfig()
MODEL_CONFIG = ModelConfig()
API_CONFIG = APIConfig()
