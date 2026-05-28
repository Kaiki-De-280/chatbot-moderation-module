import json
from pathlib import Path
from typing import Any

import joblib

from src.preprocessing import preprocess_text_for_ml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MLModerator:
    """ML-слой модерации сообщений.

    Класс загружает сохранённую модель из реестра моделей,
    выполняет предобработку текста и возвращает результат проверки
    в структурированном виде.
    """

    def __init__(
        self,
        registry_path: str = "models/model_registry.json",
        model_name: str | None = None,
    ) -> None:
        self.registry_path = self._resolve_project_path(registry_path)
        self.model_name = model_name

        self.registry = self._load_registry()
        self.model_config = self._get_model_config(self.model_name)

        self.model_name = self.model_config["name"]
        self.model_path = self._resolve_project_path(self.model_config["path"])
        self.threshold = self.model_config.get("threshold", 0.5)
        self.score_type = self.model_config.get("score_type", "predict_proba")

        self.model = self._load_model()

    def process(self, text: str) -> dict[str, Any]:
        """Проверяет сообщение с помощью ML-модели."""
        normalized_text = preprocess_text_for_ml(text)
        score = self._get_score(normalized_text)

        has_violation = score >= self.threshold

        return {
            "text": text,
            "normalized_text": normalized_text,
            "model_name": self.model_name,
            "score": score,
            "threshold": self.threshold,
            "has_violation": has_violation,
            "recommended_action": (
                "send_to_moderator" if has_violation else "allow_message"
            ),
        }

    def _load_registry(self) -> dict[str, Any]:
        """Загружает JSON-реестр моделей."""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Файл реестра моделей не найден: {self.registry_path}"
            )

        with open(self.registry_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _get_model_config(self, model_name: str | None) -> dict[str, Any]:
        """Возвращает конфигурацию выбранной модели.

        Если model_name не передан, используется default_model из реестра.
        """
        models = self.registry.get("models")

        if not models:
            raise ValueError("В model_registry.json отсутствует раздел 'models'.")

        # Поддержка нового формата:
        # "models": {"model_name": {...}}
        if isinstance(models, dict):
            selected_model_name = model_name or self.registry.get("default_model")

            if not selected_model_name:
                raise ValueError(
                    "Не указана model_name и отсутствует default_model "
                    "в model_registry.json."
                )

            if selected_model_name not in models:
                raise ValueError(
                    f"Модель '{selected_model_name}' не найдена "
                    "в model_registry.json."
                )

            model_config = models[selected_model_name].copy()
            model_config["name"] = selected_model_name

            return model_config

        # Поддержка старого формата:
        # "models": [{"name": "...", ...}]
        if isinstance(models, list):
            selected_model_name = model_name or self.registry.get("default_model")

            if selected_model_name:
                for model_config in models:
                    if model_config.get("name") == selected_model_name:
                        return model_config

                raise ValueError(
                    f"Модель '{selected_model_name}' не найдена "
                    "в model_registry.json."
                )

            if len(models) == 1:
                return models[0]

            raise ValueError(
                "В реестре несколько моделей. Укажите model_name "
                "или добавьте default_model."
            )

        raise ValueError("Некорректный формат поля 'models' в model_registry.json.")

    def _load_model(self) -> Any:
        """Загружает сохранённую модель."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

        return joblib.load(self.model_path)

    def _get_score(self, normalized_text: str) -> float:
        """Возвращает числовую оценку модели."""
        texts = [normalized_text]

        if self.score_type == "decision_function":
            score = self.model.decision_function(texts)[0]
            return float(score)

        if self.score_type == "predict_proba":
            probabilities = self.model.predict_proba(texts)[0]

            if len(probabilities) == 1:
                return float(probabilities[0])

            return float(probabilities[1])

        if self.score_type == "predict":
            prediction = self.model.predict(texts)[0]
            return float(prediction)

        raise ValueError(f"Неизвестный score_type: {self.score_type}")

    @staticmethod
    def _resolve_project_path(path: str) -> Path:
        """Преобразует относительный путь в путь от корня проекта."""
        path = Path(path)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    def get_available_models(self) -> list[str]:
        """Возвращает список моделей из реестра."""
        models = self.registry.get("models", {})

        if isinstance(models, dict):
            return list(models.keys())

        if isinstance(models, list):
            return [
                model["name"]
                for model in models
                if "name" in model
            ]

        return []