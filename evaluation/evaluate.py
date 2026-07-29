"""
evaluate.py
===========
Évaluation quantitative du pipeline sur un petit jeu de données annoté (bonus).

Métriques :
    - WER (Word Error Rate) pour l'ASR, via `jiwer`
    - Accuracy / F1 (macro) pour la classification de sentiment, via `sklearn`

Utilisation :
    python evaluation/evaluate.py --dataset evaluation/dataset_example.csv

Le CSV attendu doit contenir les colonnes :
    audio_path, reference_transcript, true_sentiment
"""

import argparse
import csv
import sys
from pathlib import Path

# Permet l'exécution du script directement (python evaluation/evaluate.py)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import jiwer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.pipeline import get_pipeline
from src.utils import get_logger, PipelineError

logger = get_logger(__name__)


def load_dataset(csv_path: str) -> list[dict]:
    """Charge le jeu de données annoté depuis un fichier CSV."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_cols = {"audio_path", "reference_transcript", "true_sentiment"}
    if rows and not required_cols.issubset(rows[0].keys()):
        raise ValueError(
            f"Colonnes manquantes dans le CSV. Attendu : {required_cols}, "
            f"trouvé : {set(rows[0].keys())}"
        )
    return rows


def evaluate(dataset_path: str) -> dict:
    """
    Exécute le pipeline sur chaque échantillon annoté et calcule les métriques.

    Returns:
        dict avec les métriques WER, accuracy et F1 macro.
    """
    dataset = load_dataset(dataset_path)
    if not dataset:
        logger.warning("Le jeu de données est vide.")
        return {}

    pipeline = get_pipeline()

    hypotheses, references = [], []
    y_true, y_pred = [], []

    for row in dataset:
        audio_path = row["audio_path"]
        reference_transcript = row["reference_transcript"]
        true_sentiment = row["true_sentiment"].strip().lower()

        try:
            result = pipeline.run(audio_path)
        except PipelineError as exc:
            logger.warning(f"Échantillon ignoré ({audio_path}) : {exc}")
            continue

        hypotheses.append(result["transcription"])
        references.append(reference_transcript)

        y_true.append(true_sentiment)
        y_pred.append(result["sentiment"])

    if not hypotheses:
        logger.error("Aucun échantillon n'a pu être évalué.")
        return {}

    # --- WER (ASR) ---
    wer_score = jiwer.wer(references, hypotheses)

    # --- Accuracy / F1 (sentiment) ---
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "n_samples_evaluated": len(hypotheses),
        "wer": round(wer_score, 4),
        "sentiment_accuracy": round(acc, 4),
        "sentiment_f1_macro": round(f1_macro, 4),
    }

    logger.info(f"Résultats de l'évaluation : {metrics}")
    print("\n=== Rapport de classification (sentiment) ===")
    print(report)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation quantitative du pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/dataset_example.csv",
        help="Chemin vers le CSV annoté (audio_path, reference_transcript, true_sentiment)",
    )
    args = parser.parse_args()

    metrics = evaluate(args.dataset)
    print("\n=== Métriques finales ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
