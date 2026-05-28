from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TransformerModerator:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        threshold: float,
        positive_class_index: int = 1,
        max_length: int = 128,
        recommended_action: str = "send_to_moderator",
        device: str | None = None,
    ) -> None:
        self.model_path = self._resolve_project_path(model_path)
        self.model_name = model_name
        self.threshold = threshold
        self.positive_class_index = positive_class_index
        self.max_length = max_length
        self.recommended_action = recommended_action

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Папка трансформерной модели не найдена: {self.model_path}"
            )

        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"В папке модели нет config.json: {self.model_path}"
            )

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def _normalize_text(self, text: str | None) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.lower()
        text = text.replace("ё", "е")
        text = " ".join(text.split())

        return text

    def predict(self, text: str | None) -> dict[str, Any]:
        original_text = "" if text is None else str(text)
        normalized_text = self._normalize_text(original_text)

        inputs = self.tokenizer(
            normalized_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            if logits.shape[-1] == 1:
                score = torch.sigmoid(logits)[0, 0].item()
            else:
                probabilities = torch.softmax(logits, dim=-1)
                score = probabilities[0, self.positive_class_index].item()

        has_violation = score >= self.threshold

        return {
            "text": original_text,
            "normalized_text": normalized_text,
            "model_name": self.model_name,
            "score": round(float(score), 4),
            "threshold": self.threshold,
            "has_violation": has_violation,
            "violation_category": "toxicity" if has_violation else None,
            "recommended_action": (
                self.recommended_action if has_violation else "allow_message"
            ),
        }

    def process(self, text: str | None) -> dict[str, Any]:
        """Проверяет сообщение через трансформерную модель."""
        return self.predict(text)

    @staticmethod
    def _resolve_project_path(path: str) -> Path:
        path_obj = Path(path)

        if path_obj.is_absolute():
            return path_obj

        return PROJECT_ROOT / path_obj
