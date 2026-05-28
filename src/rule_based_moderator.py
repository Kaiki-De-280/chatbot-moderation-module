import json
import re
from pathlib import Path

from .preprocessing import preprocess_text_for_rules


ACTION_PRIORITY = {
    "allow_message": 0,
    "show_warning": 1,
    "send_to_moderator": 2,
    "block_message": 3,
}


class RuleBasedModerator:
    """Правило-ориентированный слой модерации.

    Слой проверяет текст по регулярным выражениям из JSON-файла.
    Если найдено нарушение, возвращает его категорию, сработавший паттерн
    и рекомендуемое действие.
    """

    def __init__(self, rules_path: str):
        """Загружает правила модерации из JSON-файла."""
        self.rules_path = Path(rules_path)
        self.rules_data = {
            "schema_version": "1.0",
            "rules": [],
        }
        self.rules: list[dict] = []

        self.load_rules()

    def load_rules(self) -> None:
        """Загружает правила из файла."""
        if not self.rules_path.exists():
            self.rules = []
            return

        with open(self.rules_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.rules_data = data
        self.rules = data.get("rules", [])

    def save_rules(self) -> None:
        """Сохраняет текущие правила в JSON-файл."""
        self.rules_data["rules"] = self.rules
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.rules_path, "w", encoding="utf-8") as file:
            json.dump(self.rules_data, file, ensure_ascii=False, indent=2)

    def process(self, text: str | None) -> dict:
        """Проверяет текст по включённым правилам.

        Возвращает сообщение с результатами проверки, включающее:
        - исходный текст;
        - нормализованный текст;
        - наличие нарушения;
        - категорию нарушения;
        - сработавший паттерн;
        - рекомендуемое действие.
        """
        original_text = "" if text is None else str(text)
        normalized_text = preprocess_text_for_rules(original_text)

        matches = self._find_matches(normalized_text)
        main_match = self._choose_main_match(matches)

        if main_match is None:
            return {
                "text": original_text,
                "normalized_text": normalized_text,
                "has_violation": False,
                "violation_category": None,
                "matched_pattern": None,
                "recommended_action": "allow_message",
            }

        return {
            "text": original_text,
            "normalized_text": normalized_text,
            "has_violation": True,
            "violation_category": main_match["category"],
            "matched_pattern": main_match["pattern"],
            "recommended_action": main_match["action"],
        }

    def add_rule(self, rule: dict) -> None:
        """Добавляет новое правило."""
        self._validate_rule(rule)

        if self.get_rule(rule["id"]) is not None:
            raise ValueError(f"Правило с id '{rule['id']}' уже существует")

        self.rules.append(rule)
        self.save_rules()

    def add_patterns_to_rule(self, rule_id: str, patterns: list[str]) -> bool:
        """Добавляет новые регулярные выражения в существующее правило.

        Возвращает True, если правило найдено и обновлено.
        """
        rule = self.get_rule(rule_id)

        if rule is None:
            return False

        if not isinstance(patterns, list):
            raise ValueError("patterns должен быть списком")

        current_patterns = rule.get("patterns", [])

        if not isinstance(current_patterns, list):
            current_patterns = []

        for pattern in patterns:
            if not isinstance(pattern, str):
                raise ValueError("Каждый паттерн должен быть строкой")

            re.compile(pattern)

            if pattern not in current_patterns:
                current_patterns.append(pattern)

        rule["patterns"] = current_patterns
        self.save_rules()

        return True

    def delete_rule(self, rule_id: str) -> bool:
        """Удаляет правило по id."""
        old_count = len(self.rules)

        self.rules = [
            rule for rule in self.rules
            if rule.get("id") != rule_id
        ]

        deleted = len(self.rules) < old_count

        if deleted:
            self.save_rules()

        return deleted

    def delete_pattern_from_rule(self, rule_id: str, pattern: str) -> bool:
        """Удаляет регулярное выражение из существующего правила.

        Возвращает True, если правило найдено и регулярное выражение удалено.
        Если после удаления список patterns стал пустым, правило остаётся,
        но фактически перестаёт срабатывать.
        """
        rule = self.get_rule(rule_id)

        if rule is None:
            return False

        patterns = rule.get("patterns", [])

        if not isinstance(patterns, list):
            return False

        old_count = len(patterns)

        rule["patterns"] = [
            current_pattern
            for current_pattern in patterns
            if current_pattern != pattern
        ]

        deleted = len(rule["patterns"]) < old_count

        if deleted:
            self.save_rules()

        return deleted

    def enable_rule(self, rule_id: str) -> bool:
        """Включает правило по id."""
        rule = self.get_rule(rule_id)

        if rule is None:
            return False

        rule["enabled"] = True
        self.save_rules()

        return True

    def disable_rule(self, rule_id: str) -> bool:
        """Отключает правило по id."""
        rule = self.get_rule(rule_id)

        if rule is None:
            return False

        rule["enabled"] = False
        self.save_rules()

        return True

    def get_rule(self, rule_id: str) -> dict | None:
        """Возвращает правило по id."""
        for rule in self.rules:
            if rule.get("id") == rule_id:
                return rule

        return None

    def get_rules(self) -> list[dict]:
        """Возвращает список всех правил."""
        return self.rules

    def _find_matches(self, text: str) -> list[dict]:
        """Ищет все срабатывания включённых правил."""
        matches = []

        for rule in self.rules:
            if not rule.get("enabled", True):
                continue

            category = rule.get("category", "other")
            action = rule.get("action", "show_warning")
            patterns = rule.get("patterns", [])

            if action not in ACTION_PRIORITY:
                continue

            if not isinstance(patterns, list):
                continue

            for pattern in patterns:
                try:
                    match = re.search(pattern, text)
                except re.error:
                    continue

                if match:
                    matches.append({
                        "category": category,
                        "pattern": pattern,
                        "action": action,
                    })

        return matches

    def _choose_main_match(self, matches: list[dict]) -> dict | None:
        """Выбирает самое строгое сработавшее правило."""
        if not matches:
            return None

        return max(
            matches,
            key=lambda match: ACTION_PRIORITY.get(match["action"], 0),
        )

    def _validate_rule(self, rule: dict) -> None:
        """Проверяет структуру правила перед добавлением."""
        required_fields = [
            "id",
            "category",
            "enabled",
            "patterns",
            "action",
        ]

        for field in required_fields:
            if field not in rule:
                raise ValueError(f"В правиле отсутствует поле: {field}")

        if not isinstance(rule["id"], str):
            raise ValueError("Поле 'id' должно быть строкой")

        if not isinstance(rule["category"], str):
            raise ValueError("Поле 'category' должно быть строкой")

        if not isinstance(rule["enabled"], bool):
            raise ValueError("Поле 'enabled' должно быть bool")

        if not isinstance(rule["patterns"], list):
            raise ValueError("Поле 'patterns' должно быть списком")

        if rule["action"] not in ACTION_PRIORITY:
            raise ValueError("Некорректное значение action")
