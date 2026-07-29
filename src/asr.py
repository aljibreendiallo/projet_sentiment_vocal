"""
asr.py
======
Module de transcription vocale (Speech-to-Text) basé sur Wav2Vec2.

Modèle : jonatasgrosman/wav2vec2-large-xlsr-53-french
(cf. src/config.py pour la justification du choix)
"""

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.config import MODEL_CONFIG, AUDIO_CONFIG
from src.utils import get_logger, TranscriptionError

logger = get_logger(__name__)


class Wav2Vec2ASR:
    """
    Wrapper autour d'un modèle Wav2Vec2 pour la transcription audio -> texte.

    Le modèle et le processor sont chargés une seule fois à l'instanciation
    (coûteux en I/O et mémoire), puis réutilisés pour chaque appel à `transcribe`.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or MODEL_CONFIG.asr_model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else MODEL_CONFIG.device
        )

        logger.info(f"Chargement du modèle ASR '{self.model_name}' sur {self.device}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise TranscriptionError(
                f"Échec du chargement du modèle ASR '{self.model_name}' : {exc}"
            ) from exc

        logger.info("Modèle ASR chargé avec succès.")

    @torch.inference_mode()
    def transcribe(self, signal: np.ndarray) -> str:
        """
        Transcrit un signal audio (mono, 16kHz, normalisé) en texte.

        Args:
            signal: np.ndarray produit par `preprocessing.load_and_preprocess_audio`

        Returns:
            str: transcription textuelle (peut être une chaîne vide si le
                 modèle ne détecte aucune parole intelligible)

        Raises:
            TranscriptionError si l'inférence échoue.
        """
        try:
            inputs = self.processor(
                signal,
                sampling_rate=AUDIO_CONFIG.target_sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self.device)

            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

        except Exception as exc:
            raise TranscriptionError(f"Échec de la transcription audio : {exc}") from exc

        transcription = transcription.strip()
        logger.info(f"Transcription obtenue : '{transcription}'")
        return transcription
