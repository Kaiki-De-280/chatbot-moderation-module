import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.streamlit_utils import (
    MODEL_REGISTRY_PATH,
    add_project_root_to_path,
)


add_project_root_to_path()

from src.ml_moderator import MLModerator
from src.moderation_pipeline import ModerationPipeline


LABEL_MAP = {
    "0": 0,
    "1": 1,
    "normal": 0,
    "норм": 0,
    "норма": 0,
    "допустимое": 0,
    "allow": 0,
    "allowed": 0,
    "false": 0,
    "нет": 0,
    "violation": 1,
    "toxic": 1,
    "toxicity": 1,
    "нарушение": 1,
    "недопустимое": 1,
    "ненорм": 1,
    "bad": 1,
    "true": 1,
    "да": 1,
}

DATASET_STATE_KEY = "dataset_eval_dataframe"
DATASET_NAME_STATE_KEY = "dataset_eval_filename"


def load_registry() -> dict:
    """Загружает реестр моделей."""
    with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_models() -> dict:
    """Возвращает словарь моделей из реестра."""
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
    """Формирует подпись модели для списка."""
    models = get_models()
    config = models.get(model_name, {})

    description = config.get("description", "")
    threshold = config.get("threshold")
    model_type = config.get("type", "sklearn")

    if description:
        return f"{model_name} | {model_type} | threshold={threshold} | {description}"

    return f"{model_name} | {model_type} | threshold={threshold}"


def save_dataset_to_session(dataframe: pd.DataFrame, filename: str) -> None:
    """Сохраняет загруженный датасет в состояние сессии."""
    st.session_state[DATASET_STATE_KEY] = dataframe
    st.session_state[DATASET_NAME_STATE_KEY] = filename


def get_dataset_from_session() -> pd.DataFrame | None:
    """Возвращает датасет из состояния сессии."""
    return st.session_state.get(DATASET_STATE_KEY)


def get_dataset_name_from_session() -> str | None:
    """Возвращает имя загруженного датасета."""
    return st.session_state.get(DATASET_NAME_STATE_KEY)


def clear_dataset_from_session() -> None:
    """Очищает сохранённый датасет."""
    st.session_state.pop(DATASET_STATE_KEY, None)
    st.session_state.pop(DATASET_NAME_STATE_KEY, None)


@st.cache_resource
def load_ml_moderator(model_name: str):
    """Загружает выбранную ML-модель."""
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


@st.cache_resource
def load_pipeline(model_name: str) -> ModerationPipeline:
    """Загружает pipeline с выбранной моделью."""
    return ModerationPipeline(
        rules_path="rules/moderation_rules.json",
        registry_path="models/model_registry.json",
        model_name=model_name,
    )


def normalize_label(value: Any) -> int:
    """Приводит значение label к 0 или 1."""
    if pd.isna(value):
        raise ValueError("label содержит пустое значение")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in [0, 1]:
            return value

    if isinstance(value, float):
        if value in [0.0, 1.0]:
            return int(value)

    value_str = str(value).strip().lower()

    if value_str in LABEL_MAP:
        return LABEL_MAP[value_str]

    raise ValueError(f"Не удалось преобразовать label к 0/1: {value}")


def prepare_dataset(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    max_rows: int,
) -> pd.DataFrame:
    """Готовит датасет к проверке."""
    df = dataframe[[text_column, label_column]].copy()
    df = df.dropna(subset=[text_column, label_column])

    df[text_column] = df[text_column].astype(str)
    df["true_label"] = df[label_column].apply(normalize_label)

    if max_rows > 0:
        df = df.head(max_rows)

    return df


def get_prediction_result(
    text: str,
    mode: str,
    model_name: str,
) -> dict:
    """Получает результат проверки для одного сообщения."""
    if mode == "Только ML-модель":
        moderator = load_ml_moderator(model_name)
        return moderator.process(text)

    pipeline = load_pipeline(model_name)
    return pipeline.process(text)


def extract_prediction_info(result: dict, mode: str) -> dict:
    """Извлекает нужные поля из результата проверки."""
    if mode == "Только ML-модель":
        return {
            "pred_label": int(result["has_violation"]),
            "has_violation": result["has_violation"],
            "recommended_action": result["recommended_action"],
            "decision_source": "ml",
            "model_name": result.get("model_name"),
            "score": result.get("score"),
            "threshold": result.get("threshold"),
            "violation_category": result.get("violation_category"),
        }

    ml_result = result.get("ml") or {}

    return {
        "pred_label": int(result["has_violation"]),
        "has_violation": result["has_violation"],
        "recommended_action": result["recommended_action"],
        "decision_source": result.get("decision_source"),
        "model_name": ml_result.get("model_name"),
        "score": ml_result.get("score"),
        "threshold": ml_result.get("threshold"),
        "violation_category": result.get("violation_category"),
    }


def evaluate_dataframe(
    dataframe: pd.DataFrame,
    text_column: str,
    mode: str,
    model_name: str,
) -> pd.DataFrame:
    """Прогоняет датасет через выбранный режим проверки."""
    rows = []
    progress_bar = st.progress(0)
    status = st.empty()

    total = len(dataframe)

    for index, row in enumerate(dataframe.itertuples(index=False), start=1):
        text = getattr(row, text_column)
        true_label = getattr(row, "true_label")

        result = get_prediction_result(
            text=text,
            mode=mode,
            model_name=model_name,
        )

        prediction_info = extract_prediction_info(result, mode)

        rows.append({
            "text": text,
            "true_label": true_label,
            **prediction_info,
        })

        progress_bar.progress(index / total)
        status.write(f"Обработано сообщений: {index} / {total}")

    status.empty()

    return pd.DataFrame(rows)


def add_error_type(results: pd.DataFrame) -> pd.DataFrame:
    """Добавляет тип исхода: TN, FP, FN, TP."""
    results = results.copy()

    def get_type(row) -> str:
        if row["true_label"] == 0 and row["pred_label"] == 0:
            return "TN"
        if row["true_label"] == 0 and row["pred_label"] == 1:
            return "FP"
        if row["true_label"] == 1 and row["pred_label"] == 0:
            return "FN"
        return "TP"

    results["result_type"] = results.apply(get_type, axis=1)

    return results


def show_metrics(results: pd.DataFrame) -> None:
    """Показывает метрики качества."""
    y_true = results["true_label"]
    y_pred = results["pred_label"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    st.subheader("Метрики")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1-score", f"{f1:.4f}")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    matrix_df = pd.DataFrame(
        matrix,
        index=["Факт: normal", "Факт: violation"],
        columns=["Предсказано: normal", "Предсказано: violation"],
    )

    st.subheader("Матрица ошибок")
    st.dataframe(matrix_df, use_container_width=True)


def show_error_analysis(results: pd.DataFrame) -> None:
    """Показывает таблицы ошибок."""
    st.subheader("Анализ ошибок")

    false_positive = results[results["result_type"] == "FP"]
    false_negative = results[results["result_type"] == "FN"]

    tab_fp, tab_fn, tab_all = st.tabs([
        f"FP: ложные срабатывания ({len(false_positive)})",
        f"FN: пропущенные нарушения ({len(false_negative)})",
        "Все результаты",
    ])

    with tab_fp:
        st.dataframe(false_positive, use_container_width=True, hide_index=True)

    with tab_fn:
        st.dataframe(false_negative, use_container_width=True, hide_index=True)

    with tab_all:
        st.dataframe(results, use_container_width=True, hide_index=True)


def show_download_button(results: pd.DataFrame) -> None:
    """Кнопка скачивания результатов."""
    csv_data = results.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Скачать результаты CSV",
        data=csv_data,
        file_name="moderation_dataset_results.csv",
        mime="text/csv",
    )


def show_dataset_preview(dataframe: pd.DataFrame) -> None:
    """Показывает предпросмотр датасета с выбором диапазона строк."""
    st.subheader("Предпросмотр датасета")

    total_rows, total_columns = dataframe.shape

    st.write(
        f"Размер таблицы: {total_rows} строк, {total_columns} столбцов"
    )

    left_column, right_column = st.columns(2)

    with left_column:
        start_row = st.number_input(
            "Начальная строка предпросмотра",
            min_value=0,
            max_value=max(total_rows - 1, 0),
            value=0,
            step=10,
        )

    with right_column:
        preview_rows = st.number_input(
            "Количество строк в предпросмотре",
            min_value=5,
            max_value=min(500, total_rows),
            value=min(50, total_rows),
            step=10,
        )

    end_row = min(start_row + preview_rows, total_rows)

    st.caption(f"Показаны строки с {start_row} по {end_row - 1}")

    preview_dataframe = dataframe.iloc[start_row:end_row]

    st.dataframe(
        preview_dataframe,
        use_container_width=True,
        height=420,
    )


def main() -> None:
    st.title("Тестирование на датасете")

    st.write(
        "На этой странице можно:\n" \
        "- загрузить CSV-файл;\n"
        "- прогнать сообщения через выбранную модель или полный pipeline;\n"
        "- просмотреть результаты тестирования."
    )

    separator_name = st.selectbox(
        "Разделитель CSV",
        [
            "Запятая (,)",
            "Точка с запятой (;)",
            "Табуляция",
        ],
    )

    separator_map = {
        "Запятая (,)": ",",
        "Точка с запятой (;)": ";",
        "Табуляция": "\t",
    }

    separator = separator_map[separator_name]

    uploaded_file = st.file_uploader(
        "Загрузить CSV-файл",
        type=["csv", "txt"],
        key="dataset_eval_uploader",
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            dataframe = pd.read_csv(uploaded_file, sep=separator)
        except Exception as error:
            st.error(f"Не удалось прочитать файл: {error}")
            return

        if dataframe.empty:
            st.error("Файл загружен, но таблица пустая.")
            return

        save_dataset_to_session(dataframe, uploaded_file.name)
        st.success(f"Датасет загружен: {uploaded_file.name}")

    dataframe = get_dataset_from_session()
    dataset_name = get_dataset_name_from_session()

    if dataframe is None:
        st.info(
            "Ожидаемый минимальный формат файла: две колонки — text и label. "
            "Label должен принимать значения 0/1."
        )
        return

    st.info(f"Текущий датасет в сессии: {dataset_name}")

    if st.button("Очистить загруженный датасет"):
        clear_dataset_from_session()
        st.rerun()

    # st.subheader("Предпросмотр датасета")
    # st.write(f"Размер таблицы: {dataframe.shape[0]} строк, {dataframe.shape[1]} столбцов")
    # st.dataframe(dataframe.head(10), use_container_width=True)
    show_dataset_preview(dataframe)

    columns = list(dataframe.columns)

    default_text_index = columns.index("text") if "text" in columns else 0
    default_label_index = columns.index("label") if "label" in columns else min(1, len(columns) - 1)

    text_column = st.selectbox(
        "Колонка с текстом",
        columns,
        index=default_text_index,
    )

    label_column = st.selectbox(
        "Колонка с label",
        columns,
        index=default_label_index,
    )

    models = get_models()

    if not models:
        st.error("В реестре моделей нет доступных моделей.")
        return

    model_names = list(models.keys())
    default_model = get_default_model()

    default_model_index = 0

    if default_model in model_names:
        default_model_index = model_names.index(default_model)

    selected_model = st.selectbox(
        "Выберите модель",
        model_names,
        index=default_model_index,
        format_func=make_model_label,
    )

    mode = st.radio(
        "Режим проверки",
        [
            "Только ML-модель",
            "Полный pipeline",
        ],
    )

    max_rows = st.number_input(
        "Максимальное количество строк для проверки",
        min_value=1,
        max_value=100_000,
        value=1000,
        step=100,
    )

    st.warning(
        "Для трансформерных моделей лучше начинать с небольшого количества строк, "
        "например 100–300."
    )

    if st.button("Рассчитать метрики", type="primary"):
        try:
            prepared_df = prepare_dataset(
                dataframe=dataframe,
                text_column=text_column,
                label_column=label_column,
                max_rows=int(max_rows),
            )
        except Exception as error:
            st.error(f"Ошибка подготовки датасета: {error}")
            return

        if prepared_df.empty:
            st.error("После подготовки не осталось строк для проверки.")
            return

        with st.spinner("Выполняется проверка сообщений..."):
            results = evaluate_dataframe(
                dataframe=prepared_df,
                text_column=text_column,
                mode=mode,
                model_name=selected_model,
            )

        results = add_error_type(results)

        st.success("Проверка завершена.")

        show_metrics(results)
        show_error_analysis(results)
        show_download_button(results)


if __name__ == "__main__":
    main()
