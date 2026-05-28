import json
from pathlib import Path

import streamlit as st

from src.moderation_pipeline import ModerationPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_REGISTRY_PATH = PROJECT_ROOT / "models" / "model_registry.json"


def load_model_registry() -> dict:
    """Загружает реестр моделей."""
    with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_available_models() -> list[str]:
    """Возвращает список доступных моделей."""
    registry = load_model_registry()
    models = registry.get("models", {})

    if not isinstance(models, dict):
        return []

    return list(models.keys())


def get_default_model() -> str | None:
    """Возвращает модель по умолчанию."""
    registry = load_model_registry()
    return registry.get("default_model")


EXAMPLE_MESSAGES = {
    "Нормальное": "Привет, можешь помочь с заданием?",
    "Оскорбление": "Ты полный идиот",
    "Угроза": "Я тебя убью",
    "Спам-ссылка": "Зайди на https://example.com и получи подарок",
    "Спорный пример": "Ты вообще понимаешь, что пишешь?",
}


def get_model_config(model_name: str) -> dict:
    """Возвращает конфигурацию выбранной модели."""
    registry = load_model_registry()
    models = registry.get("models", {})

    if not isinstance(models, dict):
        return {}

    return models.get(model_name, {})


def show_selected_model_info(model_name: str) -> None:
    """Показывает краткую информацию о выбранной модели."""
    config = get_model_config(model_name)

    description = config.get("description", "Описание модели не указано.")
    model_type = config.get("type", "sklearn")
    threshold = config.get("threshold")

    st.caption(f"**Выбранная модель:** `{model_name}`")
    st.caption(f"**Описание:** {description}")

    with st.expander("Параметры выбранной модели"):
        left_column, right_column = st.columns(2)

        with left_column:
            st.write("**Тип модели:**", model_type)
            st.write("**Threshold:**", threshold)

        with right_column:
            st.write("**Путь:**")
            st.code(config.get("path", "Не указан"))

            if "score_type" in config:
                st.write("**Тип оценки:**", config["score_type"])


def show_example_buttons(state_key: str) -> None:
    """Показывает кнопки с готовыми примерами сообщений."""
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    st.write("**Быстрые примеры:**")

    columns = st.columns(len(EXAMPLE_MESSAGES))

    for column, (label, message) in zip(columns, EXAMPLE_MESSAGES.items()):
        if column.button(label, use_container_width=True):
            st.session_state[state_key] = message


@st.cache_resource
def load_pipeline(model_name: str) -> ModerationPipeline:
    """Загружает pipeline один раз для выбранной модели."""
    return ModerationPipeline(
        rules_path="rules/moderation_rules.json",
        registry_path="models/model_registry.json",
        model_name=model_name,
    )


def show_final_decision(result: dict) -> None:
    """Показывает главное решение pipeline."""
    action = result["recommended_action"]

    if action == "allow_message":
        st.success("Сообщение разрешено")
    elif action == "show_warning":
        st.warning("Обнаружено нарушение: рекомендуется предупреждение")
    elif action == "send_to_moderator":
        st.warning("Сообщение требует проверки модератором")
    elif action == "block_message":
        st.error("Сообщение должно быть заблокировано")
    else:
        st.info(f"Решение: {action}")

    st.write("**Есть нарушение:**", "да" if result["has_violation"] else "нет")
    st.write("**Категория нарушения:**", result["violation_category"])
    st.write("**Источник решения:**", result["decision_source"])
    st.write("**Рекомендуемое действие:**", result["recommended_action"])


def show_rule_based_result(result: dict) -> None:
    """Показывает результат rule-based слоя."""
    rule_result = result["rule_based"]

    st.write("**Есть нарушение:**", "да" if rule_result["has_violation"] else "нет")
    st.write("**Категория:**", rule_result["violation_category"])
    st.write("**Сработавший паттерн:**", rule_result["matched_pattern"])
    st.write("**Рекомендуемое действие:**", rule_result["recommended_action"])
    st.write("**Нормализованный текст:**", rule_result["normalized_text"])


def show_ml_result(result: dict) -> None:
    """Показывает результат ML-слоя."""
    ml_result = result["ml"]

    if ml_result is None:
        st.info("ML-слой не запускался, потому что rule-based слой сразу принял блокирующее решение.")
        return

    st.write("**Модель:**", ml_result["model_name"])
    st.write("**Есть нарушение:**", "да" if ml_result["has_violation"] else "нет")
    st.write("**Score:**", ml_result["score"])
    st.write("**Threshold:**", ml_result["threshold"])
    st.write("**Рекомендуемое действие:**", ml_result["recommended_action"])
    st.write("**Нормализованный текст:**", ml_result["normalized_text"])


def main() -> None:
    st.title("Модуль автоматической модерации сообщений")
    st.write(
        "Здесь можно проверить пользовательское сообщение через rule-based слой"
        "и ML-модель, после чего посмотреть итоговое решение."
    )

    available_models = get_available_models()
    default_model = get_default_model()

    if not available_models:
        st.error("В реестре моделей нет доступных моделей.")
        return

    default_index = 0

    if default_model in available_models:
        default_index = available_models.index(default_model)

    selected_model = st.selectbox(
        "Выберите ML-модель:",
        available_models,
        index=default_index,
    )

    show_selected_model_info(selected_model)
    pipeline = load_pipeline(selected_model)

    show_example_buttons("main_message_text")
    with st.form("moderation_form"):
        text = st.text_area(
            "Введите сообщение для проверки:",
            height=120,
            placeholder="Например: Привет, можешь помочь?",
            key="main_message_text",
        )
        st.caption("Подсказка: Ctrl+Enter запускает проверку.")
        submitted = st.form_submit_button("Проверить сообщение")

    if submitted:
        if not text.strip():
            st.warning("Введите текст сообщения.")
            return

        result = pipeline.process(text)

        st.divider()
        st.subheader("Итоговое решение")
        show_final_decision(result)

        st.divider()

        left_column, right_column = st.columns(2)

        with left_column:
            st.subheader("Rule-based слой")
            show_rule_based_result(result)

        with right_column:
            st.subheader("ML-слой")
            show_ml_result(result)

        with st.expander("Показать полный JSON"):
            st.code(
                json.dumps(result, ensure_ascii=False, indent=2),
                language="json",
            )


if __name__ == "__main__":
    main()
