"""Загрузка joblib-модели и предсказание. Модель сохранена в artifacts/ (ноутбук экспериментов)."""

import json
from pathlib import Path
from typing import Dict

import joblib


class ModelService:
    """load() при старте приложения, predict() в запросах."""

    def __init__(self, model_path, metadata_path):
        self._model_path = Path(model_path)
        self._metadata_path = Path(metadata_path)
        self._model = None
        self._metadata: Dict = {}

    def load(self) -> None:
        self._model = joblib.load(self._model_path)
        if self._metadata_path.exists():
            self._metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> Dict:
        return self._metadata

    @property
    def classes(self):
        return list(self._model.classes_) if self._model else []

    def predict(self, text: str) -> Dict:
        proba = self._model.predict_proba([text])[0]
        probabilities = {label: float(p) for label, p in zip(self._model.classes_, proba)}
        label = max(probabilities, key=probabilities.get)
        return {"label": label, "probabilities": probabilities}
