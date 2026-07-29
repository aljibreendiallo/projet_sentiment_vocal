"""
test_pipeline.py
=================
Tests unitaires ciblant principalement les règles métier et validations
(prétraitement, exceptions), sans dépendre du téléchargement des poids
des modèles Hugging Face (mockés via unittest.mock / monkeypatch).

Lancement :
    pytest tests/ -v
"""

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing import (
    validate_file_format,
    validate_file_size,
    normalize_amplitude,
    is_silent,
    load_and_preprocess_audio,
)
from src.utils import (
    InvalidAudioFormatError,
    EmptyAudioError,
    AudioTooShortError,
    AudioTooLongError,
    SilentAudioError,
)
from src.config import AUDIO_CONFIG


# --------------------------------------------------------------------------
# Fixtures : génération de fichiers audio synthétiques pour les tests
# --------------------------------------------------------------------------
@pytest.fixture
def tmp_wav_factory(tmp_path):
    """Factory créant un fichier .wav synthétique avec une durée/contenu donnés."""

    def _make(filename: str, duration_s: float, sr: int = 16000, silent: bool = False):
        n_samples = int(duration_s * sr)
        if silent or n_samples == 0:
            signal = np.zeros(max(n_samples, 1), dtype=np.float32)
        else:
            t = np.linspace(0, duration_s, n_samples, endpoint=False)
            signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # tonalité 440Hz
        path = tmp_path / filename
        sf.write(str(path), signal, sr)
        return str(path)

    return _make


# --------------------------------------------------------------------------
# Tests : validation de format
# --------------------------------------------------------------------------
def test_validate_file_format_accepts_wav(tmp_wav_factory):
    path = tmp_wav_factory("ok.wav", duration_s=1.0)
    validate_file_format(path)  # ne doit pas lever d'exception


def test_validate_file_format_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "audio.ogg"
    bad_file.write_bytes(b"fake content")
    with pytest.raises(InvalidAudioFormatError):
        validate_file_format(str(bad_file))


# --------------------------------------------------------------------------
# Tests : validation de taille
# --------------------------------------------------------------------------
def test_validate_file_size_rejects_empty_file(tmp_path):
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(EmptyAudioError):
        validate_file_size(str(empty_file))


def test_validate_file_size_rejects_oversized_file(tmp_path, monkeypatch):
    big_file = tmp_path / "big.wav"
    big_file.write_bytes(b"0" * 1024)  # petit fichier réel

    # On force artificiellement la limite à 0 Mo pour déclencher le rejet
    monkeypatch.setattr(AUDIO_CONFIG, "max_file_size_mb", 0)
    with pytest.raises(Exception):
        validate_file_size(str(big_file))


# --------------------------------------------------------------------------
# Tests : normalisation et détection de silence
# --------------------------------------------------------------------------
def test_normalize_amplitude_scales_to_unit_peak():
    signal = np.array([0.0, 2.0, -4.0, 1.0], dtype=np.float32)
    normalized = normalize_amplitude(signal)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)


def test_normalize_amplitude_handles_zero_signal():
    signal = np.zeros(10, dtype=np.float32)
    normalized = normalize_amplitude(signal)
    assert np.all(normalized == 0)


def test_is_silent_detects_zero_signal():
    signal = np.zeros(1000, dtype=np.float32)
    assert bool(is_silent(signal)) is True


def test_is_silent_detects_active_signal():
    t = np.linspace(0, 1, 16000)
    signal = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert bool(is_silent(signal)) is False


# --------------------------------------------------------------------------
# Tests : pipeline de prétraitement complet (règles métier bout-en-bout)
# --------------------------------------------------------------------------
def test_load_and_preprocess_audio_success(tmp_wav_factory):
    path = tmp_wav_factory("valid.wav", duration_s=2.0)
    signal = load_and_preprocess_audio(path)
    assert isinstance(signal, np.ndarray)
    assert signal.size > 0


def test_load_and_preprocess_audio_rejects_too_short(tmp_wav_factory):
    path = tmp_wav_factory("short.wav", duration_s=0.05)
    with pytest.raises(AudioTooShortError):
        load_and_preprocess_audio(path)


def test_load_and_preprocess_audio_rejects_too_long(tmp_wav_factory, monkeypatch):
    # On réduit temporairement la limite max pour ne pas générer un fichier de 5min+ en test
    monkeypatch.setattr(AUDIO_CONFIG, "max_duration_seconds", 1)
    path = tmp_wav_factory("long.wav", duration_s=2.0)
    with pytest.raises(AudioTooLongError):
        load_and_preprocess_audio(path)


def test_load_and_preprocess_audio_rejects_silent_file(tmp_wav_factory):
    path = tmp_wav_factory("silent.wav", duration_s=1.0, silent=True)
    with pytest.raises(SilentAudioError):
        load_and_preprocess_audio(path)


def test_load_and_preprocess_audio_rejects_wrong_extension(tmp_path):
    bad_file = tmp_path / "audio.flac"
    bad_file.write_bytes(b"not a real audio file")
    with pytest.raises(InvalidAudioFormatError):
        load_and_preprocess_audio(str(bad_file))


# --------------------------------------------------------------------------
# Tests : SentimentAnalyzer avec mock (pas de téléchargement de modèle réel)
# --------------------------------------------------------------------------
class DummyLogits:
    """Simule la sortie `.logits` d'un modèle HuggingFace pour les tests mockés."""

    def __init__(self, values):
        import torch

        self._tensor = torch.tensor([values])

    def __getattr__(self, item):
        return getattr(self._tensor, item)


def test_sentiment_star_mapping_positive(monkeypatch):
    """
    Vérifie le remappage 5 étoiles -> 3 classes métier, sans charger
    de vrais poids de modèle (mock des couches transformers).
    """
    from src import sentiment as sentiment_module

    class FakeModel:
        class config:
            id2label = {
                0: "1 star",
                1: "2 stars",
                2: "3 stars",
                3: "4 stars",
                4: "5 stars",
            }

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            import torch

            class Output:
                logits = torch.tensor([[0.0, 0.0, 0.0, 0.1, 5.0]])  # "5 stars" gagnant

            return Output()

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            import torch

            class Encoded(dict):
                def to(self, device):
                    return self

            return Encoded(input_ids=torch.tensor([[1, 2, 3]]))

    monkeypatch.setattr(
        sentiment_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: FakeModel(),
    )
    monkeypatch.setattr(
        sentiment_module.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )

    analyzer = sentiment_module.SentimentAnalyzer()
    result = analyzer.analyze("Ce service est excellent, je suis ravi !")

    assert result["sentiment"] == "positif"
    assert result["raw_label"] == "5 stars"
    assert 0.0 <= result["confidence"] <= 1.0


def test_sentiment_analyzer_rejects_empty_text(monkeypatch):
    from src import sentiment as sentiment_module

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        sentiment_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **k: FakeModel(),
    )
    monkeypatch.setattr(
        sentiment_module.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )

    analyzer = sentiment_module.SentimentAnalyzer()
    with pytest.raises(Exception):
        analyzer.analyze("   ")
