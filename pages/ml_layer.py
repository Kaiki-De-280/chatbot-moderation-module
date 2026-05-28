import json

import streamlit as st

from app.streamlit_utils import (
    MODEL_REGISTRY_PATH,
    add_project_root_to_path,
)


add_project_root_to_path()

from src.ml_moderator import MLModerator


ML_EXAMPLE_MESSAGES = {
    "Нормальное": "Спасибо, всё понятно",
    "Оскорбление": "Ты совсем тупой что ли",
    "Угроза": "Я тебя сейчас найду и ударю",
    "Скрытая токсичность": "От тебя никакой пользы, только мешаешь",
    "Спорный пример": "Ответ опять какой-то странный",
}


def show_example_buttons(state_key: str) -> None:
    """Показывает кнопки с готовыми примерами сообщений."""
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    st.write("**Быстрые примеры:**")

    columns = st.columns(len(ML_EXAMPLE_MESSAGES))

    for column, (label, message) in zip(columns, ML_EXAMPLE_MESSAGES.items()):
        if column.button(label, use_container_width=True):
            st.session_state[state_key] = message


def load_registry() -> dict:
    """Загружает реестр моделей."""
    with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_models() -> dict:
    """Возвращает модели из реестра."""
    registry = load_registry()
    models = registry.get("models", {})

    if not isinstance(models, dict):
        return {}

    return models


def get_default_model() -> str | None:
    """Возвращает модель по умолчанию."""
    registry = load_registry()
    return registry.get("default_model")


def make_model_label(model_name: str) -> str:
    """Формирует подпись модели для selectbox."""
    models = get_models()
    config = models.get(model_name, {})

    model_type = config.get("type", "sklearn")
    threshold = config.get("threshold")

    return f"{model_name} | {model_type} | threshold={threshold}"


@st.cache_resource
def load_moderator(model_name: str):
    """Загружает выбранную ML-модель.

    Для sklearn-моделей используется MLModerator.
    Для transformer-моделей используется TransformerModerator.
    """
    models = get_models()

    if model_name not in models:
        raise ValueError(f"Модель '{model_name}' не найдена в реестре.")

    config = models[model_name]
    model_type = config.get("type", "sklearn")

    if model_type == "transformer":
        from src.transformer_moderator import TransformerModerator

        return TransformerModerator(
            model_path=config["path"],
            model_name=model_name,
            threshold=float(config.get("threshold", 0.5)),
        )

    return MLModerator(
        registry_path=str(MODEL_REGISTRY_PATH),
        model_name=model_name,
    )


def show_model_info(model_name: str) -> None:
    """Показывает краткую информацию о выбранной модели."""
    models = get_models()
    config = models[model_name]

    st.subheader("Информация о модели")

    left_column, right_column = st.columns(2)

    with left_column:
        st.write("**Название:**", model_name)
        st.write("**Тип:**", config.get("type", "sklearn"))
        st.write("**Threshold:**", config.get("threshold"))
        st.write("**Описание:**", config.get("description", "Описание не указано."))

    with right_column:
        st.write("**Путь:**")
        st.code(config.get("path", "Не указан"))

        if "score_type" in config:
            st.write("**Тип оценки:**", config["score_type"])

    with st.expander("Показать запись модели в реестре"):
        st.code(
            json.dumps(config, ensure_ascii=False, indent=2),
            language="json",
        )


def show_ml_result(result: dict) -> None:
    """Показывает результат работы ML-модели."""
    if result["has_violation"]:
        st.warning("ML-модель обнаружила признаки нарушения.")
    else:
        st.success("ML-модель не обнаружила признаков нарушения.")

    left_column, right_column = st.columns(2)

    with left_column:
        st.write("**Модель:**", result["model_name"])
        st.write("**Есть нарушение:**", "да" if result["has_violation"] else "нет")
        st.write("**Score:**", result["score"])
        st.write("**Threshold:**", result["threshold"])

    with right_column:
        st.write("**Рекомендуемое действие:**", result["recommended_action"])
        st.write("**Категория:**", result.get("violation_category"))
        st.write("**Нормализованный текст:**", result["normalized_text"])

    with st.expander("Показать JSON результата"):
        st.code(
            json.dumps(result, ensure_ascii=False, indent=2),
            language="json",
        )


def show_ml_checker(model_name: str) -> None:
    """Форма проверки сообщения только ML-слоем."""
    st.subheader("Проверка сообщения")

    show_example_buttons("ml_message_text")
    with st.form("ml_check_form"):
        text = st.text_area(
            "Введите сообщение:",
            height=120,
            placeholder="Например: Ты отвратительный человек",
            key="ml_message_text",
        )
        st.caption("Подсказка: Ctrl+Enter запускает проверку.")
        submitted = st.form_submit_button("Проверить")

    if not submitted:
        return

    if not text.strip():
        st.warning("Введите текст сообщения.")
        return

    moderator = load_moderator(model_name)
    result = moderator.process(text)

    show_ml_result(result)


def main() -> None:
    st.title("ML-слой модерации")

    st.write(
        "На этой странице можно:\n" \
        "- проверить сообщение только через выбранную ML-модель;\n"
        "- посмотреть информацию о модели."
    )

    models = get_models()

    if not models:
        st.error("В реестре моделей нет доступных моделей.")
        return

    model_names = list(models.keys())
    default_model = get_default_model()

    default_index = 0

    if default_model in model_names:
        default_index = model_names.index(default_model)

    selected_model = st.selectbox(
        "Выберите модель:",
        model_names,
        index=default_index,
        format_func=make_model_label,
    )

    if st.button("Перезагрузить модели"):
        st.cache_resource.clear()
        st.rerun()

    tab_check, tab_info = st.tabs([
        "Проверка",
        "Информация о модели",
    ])

    with tab_check:
        show_ml_checker(selected_model)

    with tab_info:
        show_model_info(selected_model)


if __name__ == "__main__":
    main()
