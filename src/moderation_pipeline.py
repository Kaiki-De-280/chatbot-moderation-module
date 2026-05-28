from pathlib import Path
import json
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rule_based_moderator import RuleBasedModerator
from src.ml_moderator import MLModerator
from src.transformer_moderator import TransformerModerator


ACTION_PRIORITY = {
    "allow_message": 0,
    "show_warning": 1,
    "send_to_moderator": 2,
    "block_message": 3,
}


class ModerationPipeline:
    """Общий pipeline модерации.

    Pipeline объединяет два слоя:
    1. rule-based слой — быстрые прозрачные правила;
    2. ML-слой — классическая ML-модель или трансформерная модель.

    Итоговое решение выбирается по самому строгому действию.
    """

    def __init__(
        self,
        rules_path: str = "rules/moderation_rules.json",
        registry_path: str = "models/model_registry.json",
        model_name: str | None = None,
    ):
        """Инициализирует rule-based и ML-модераторы."""
        self.rules_path = self._resolve_project_path(rules_path)
        self.registry_path = self._resolve_project_path(registry_path)
        self.model_name = model_name

        self.rule_moderator = RuleBasedModerator(str(self.rules_path))
        self.ml_moderator = self._load_ml_moderator()

    def process(self, text: str | None) -> dict:
        """Проверяет сообщение всеми нужными слоями."""
        original_text = "" if text is None else str(text)

        rule_result = self.rule_moderator.process(original_text)

        # Если правило сразу требует блокировки,
        # ML-модель можно не запускать.
        if rule_result["recommended_action"] == "block_message":
            return self._build_final_result(
                text=original_text,
                rule_result=rule_result,
                ml_result=None,
            )

        ml_result = self.ml_moderator.process(original_text)

        return self._build_final_result(
            text=original_text,
            rule_result=rule_result,
            ml_result=ml_result,
        )

    def _load_ml_moderator(self) -> Any:
        """Загружает ML-модератор нужного типа из реестра моделей."""
        registry = self._load_registry()
        model_config = self._get_model_config(registry)
        selected_model_name = model_config["name"]

        model_type = model_config.get("type", "sklearn")

        if model_type == "transformer":
            return TransformerModerator(
                model_path=model_config["path"],
                model_name=selected_model_name,
                threshold=float(model_config.get("threshold", 0.5)),
            )

        return MLModerator(
            registry_path=str(self.registry_path),
            model_name=selected_model_name,
        )

    def _load_registry(self) -> dict[str, Any]:
        """Загружает реестр моделей."""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Файл реестра моделей не найден: {self.registry_path}"
            )

        with open(self.registry_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _get_model_config(self, registry: dict[str, Any]) -> dict[str, Any]:
        """Возвращает конфигурацию выбранной модели."""
        models = registry.get("models")

        if not isinstance(models, dict):
            raise ValueError(
                "В текущей версии pipeline ожидает, что поле 'models' "
                "в model_registry.json является словарём."
            )

        selected_model_name = self.model_name or registry.get("default_model")

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

    def _build_final_result(
        self,
        text: str,
        rule_result: dict,
        ml_result: dict | None,
    ) -> dict:
        """Формирует общий результат модерации."""
        final_action = self._choose_final_action(rule_result, ml_result)
        has_violation = final_action != "allow_message"

        violation_category = self._choose_violation_category(
            rule_result,
            ml_result,
        )

        decision_source = self._choose_decision_source(
            final_action,
            rule_result,
            ml_result,
        )

        return {
            "text": text,
            "has_violation": has_violation,
            "violation_category": violation_category,
            "recommended_action": final_action,
            "decision_source": decision_source,
            "rule_based": rule_result,
            "ml": ml_result,
        }

    def _choose_final_action(
        self,
        rule_result: dict,
        ml_result: dict | None,
    ) -> str:
        """Выбирает самое строгое действие между rule-based и ML."""
        rule_action = rule_result["recommended_action"]

        if ml_result is None:
            return rule_action

        ml_action = ml_result["recommended_action"]

        if ACTION_PRIORITY[rule_action] >= ACTION_PRIORITY[ml_action]:
            return rule_action

        return ml_action

    def _choose_violation_category(
        self,
        rule_result: dict,
        ml_result: dict | None,
    ) -> str | None:
        """Выбирает итоговую категорию нарушения."""
        if rule_result["has_violation"]:
            return rule_result["violation_category"]

        if ml_result is not None and ml_result["has_violation"]:
            return ml_result.get("violation_category") or "toxicity"

        return None

    def _choose_decision_source(
        self,
        final_action: str,
        rule_result: dict,
        ml_result: dict | None,
    ) -> str:
        """Определяет, какой слой повлиял на итоговое решение."""
        rule_action = rule_result["recommended_action"]

        if ml_result is None:
            if rule_action == final_action:
                return "rule_based"
            return "none"

        ml_action = ml_result["recommended_action"]

        if rule_action == final_action and ml_action == final_action:
            return "rule_based+ml"

        if rule_action == final_action:
            return "rule_based"

        if ml_action == final_action:
            return "ml"

        return "none"

    @staticmethod
    def _resolve_project_path(path: str) -> Path:
        """Преобразует относительный путь в путь от корня проекта."""
        path_obj = Path(path)

        if path_obj.is_absolute():
            return path_obj

        return PROJECT_ROOT / path_obj
