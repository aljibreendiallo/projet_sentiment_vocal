"""
app_gradio.py
=============
Interface Gradio permettant de tester le pipeline complet
(audio -> transcription -> sentiment) de façon interactive.

Lancement :
    python app_gradio.py
"""

import gradio as gr

from src.pipeline import get_pipeline
from src.utils import get_logger, PipelineError
from src.config import AUDIO_CONFIG, MODEL_CONFIG

logger = get_logger(__name__)

# Chargé une seule fois au démarrage de l'app (modèles réutilisés entre requêtes)
pipeline = get_pipeline()

SENTIMENT_EMOJI = {
    "positif": "🟢 Positif",
    "neutre": "🟡 Neutre",
    "négatif": "🔴 Négatif",
}


def analyze_audio(audio_file):
    """
    Callback Gradio : reçoit le chemin d'un fichier audio uploadé/enregistré
    et renvoie (transcription, sentiment formaté, score de confiance, message d'erreur).
    """
    if audio_file is None:
        return "", "", 0.0, "⚠️ Veuillez fournir un fichier audio."

    try:
        result = pipeline.run(audio_file)
        sentiment_display = SENTIMENT_EMOJI.get(result["sentiment"], result["sentiment"])
        return (
            result["transcription"],
            sentiment_display,
            result["confidence"],
            "",
        )
    except PipelineError as exc:
        logger.warning(f"Erreur métier Gradio : {exc}")
        return "", "", 0.0, f"❌ {exc}"
    except Exception as exc:
        logger.exception("Erreur inattendue dans l'interface Gradio")
        return "", "", 0.0, f"❌ Erreur inattendue : {exc}"


with gr.Blocks(title="Détection de Sentiment Vocal") as demo:
    gr.Markdown(
        f"""
        # 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux
        Pipeline **Wav2Vec 2.0** (transcription) + **BERT** (analyse de sentiment).

        Formats acceptés : `.wav`, `.mp3` — durée max. {AUDIO_CONFIG.max_duration_seconds // 60} minutes.

        Modèle ASR : `{MODEL_CONFIG.asr_model_name}`
        Modèle sentiment : `{MODEL_CONFIG.sentiment_model_name}`
        """
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Fichier audio à analyser",
        )

    submit_btn = gr.Button("🔍 Analyser", variant="primary")

    with gr.Row():
        transcription_output = gr.Textbox(
            label="📝 Transcription (ASR)", lines=4, interactive=False
        )

    with gr.Row():
        sentiment_output = gr.Textbox(label="😊 Sentiment détecté", interactive=False)
        confidence_output = gr.Number(label="🎯 Score de confiance", interactive=False)

    error_output = gr.Markdown()

    submit_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[transcription_output, sentiment_output, confidence_output, error_output],
    )

    gr.Examples(
        examples=[
            "tests/audio_samples/exemple_audio_positif.mp3",
            "tests/audio_samples/exemple_audio_neutre.mp3",
            "tests/audio_samples/exemple_audio_negatif.mp3",
        ],
        inputs=audio_input,
        label="Exemples de démonstration (un par classe de sentiment)",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
