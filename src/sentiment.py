"""
sentiment.py
============
Module d'analyse de sentiment (texte -> positif / négatif / neutre)
basé sur un modèle BERT multilingue fine-tuné (cf. src/config.py).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODEL_CONFIG
from src.utils import get_logger, SentimentAnalysisError

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Wrapper autour d'un modèle BERT de classification de sentiment.

    Le modèle brut prédit une note de 1 à 5 étoiles ; cette classe remappe
    la sortie vers les 3 classes métier (positif / neutre / négatif) et
    renvoie un score de confiance associé.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or MODEL_CONFIG.sentiment_model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else MODEL_CONFIG.device
        )
        self.label_map = MODEL_CONFIG.star_to_label

        logger.info(f"Chargement du modèle de sentiment '{self.model_name}' sur {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise SentimentAnalysisError(
                f"Échec du chargement du modèle de sentiment '{self.model_name}' : {exc}"
            ) from exc

        logger.info("Modèle de sentiment chargé avec succès.")

    @torch.inference_mode()
    def analyze(self, text: str) -> dict:
        """
        Analyse le sentiment d'un texte.

        Args:
            text: transcription issue du module ASR

        Returns:
            dict: {
                "sentiment": "positif" | "neutre" | "négatif",
                "confidence": float (0-1),
                "raw_label": label brut du modèle (ex: "4 stars")
            }

        Raises:
            SentimentAnalysisError si le texte est vide ou si l'inférence échoue.
        """
        if not text or not text.strip():
            raise SentimentAnalysisError(
                "Impossible d'analyser un texte vide (transcription ASR vide)."
            )

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

            predicted_idx = int(torch.argmax(probs).item())
            confidence = float(probs[predicted_idx].item())
            raw_label = self.model.config.id2label[predicted_idx]

        except Exception as exc:
            raise SentimentAnalysisError(
                f"Échec de l'analyse de sentiment : {exc}"
            ) from exc

        sentiment = self.label_map.get(raw_label, "neutre")

        logger.info(
            f"Sentiment détecté : {sentiment} (label brut='{raw_label}', "
            f"confiance={confidence:.3f})"
        )

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "raw_label": raw_label,
        }
