import json
import re

import pandas as pd
import streamlit as st

from app.streamlit_utils import (
    RULES_PATH,
    add_project_root_to_path,
    read_file_text,
)


add_project_root_to_path()

from src.rule_based_moderator import RuleBasedModerator


CATEGORY_OPTIONS = [
    "spam",
    "insult",
    "threat",
    "obscenity",
    "aggression",
    "other",
]

ACTION_OPTIONS = [
    "show_warning",
    "send_to_moderator",
    "block_message",
]


def load_moderator() -> RuleBasedModerator:
    """Создаёт объект rule-based модератора."""
    return RuleBasedModerator(str(RULES_PATH))


def rules_to_dataframe(rules: list[dict]) -> pd.DataFrame:
    """Преобразует список правил в таблицу."""
    rows = []

    for rule in rules:
        patterns = rule.get("patterns", [])

        if not patterns:
            rows.append({
                "id": rule.get("id"),
                "category": rule.get("category"),
                "enabled": rule.get("enabled"),
                "action": rule.get("action"),
                "pattern": None,
            })
            continue

        for pattern in patterns:
            rows.append({
                "id": rule.get("id"),
                "category": rule.get("category"),
                "enabled": rule.get("enabled"),
                "action": rule.get("action"),
                "pattern": pattern,
            })

    return pd.DataFrame(rows)


def compile_patterns(patterns: list[str]) -> list[str]:
    """Проверяет, что регулярные выражения корректны."""
    errors = []

    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as error:
            errors.append(f"{pattern} → {error}")

    return errors


def show_rules_table(moderator: RuleBasedModerator) -> None:
    """Показывает список всех правил."""
    st.subheader("Список правил")

    rules = moderator.get_rules()

    if not rules:
        st.info("Правила не найдены.")
        return

    dataframe = rules_to_dataframe(rules)

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def show_add_rule_form(moderator: RuleBasedModerator) -> None:
    """Форма добавления нового правила или паттернов в существующее правило."""
    st.subheader("Добавить правило или регулярное выражение")

    rules = moderator.get_rules()
    rule_ids = [
        rule.get("id")
        for rule in rules
        if rule.get("id")
    ]

    mode = st.radio(
        "Режим добавления",
        [
            "Создать новое правило",
            "Добавить регулярное выражение в существующее правило",
        ],
    )

    if mode == "Создать новое правило":
        with st.form("add_new_rule_form"):
            rule_id = st.text_input(
                "ID правила",
                placeholder="Например: strong_insult_keywords",
            )

            category = st.selectbox(
                "Категория нарушения",
                CATEGORY_OPTIONS,
            )

            action = st.selectbox(
                "Рекомендуемое действие",
                ACTION_OPTIONS,
            )

            enabled = st.checkbox(
                "Правило включено",
                value=True,
            )

            patterns_text = st.text_area(
                "Регулярные выражения",
                height=150,
                placeholder=(
                    "Каждое регулярное выражение с новой строки.\n"
                    "Например:\n"
                    "\\bидиот\\b"
                ),
            )

            submitted = st.form_submit_button("Создать правило")

        if not submitted:
            return

        patterns = [
            line.strip()
            for line in patterns_text.splitlines()
            if line.strip()
        ]

        if not rule_id.strip():
            st.error("Укажите ID правила.")
            return

        if not patterns:
            st.error("Добавьте хотя бы одно регулярное выражение.")
            return

        pattern_errors = compile_patterns(patterns)

        if pattern_errors:
            st.error("Есть ошибки в регулярных выражениях:")
            for error in pattern_errors:
                st.code(error)
            return

        new_rule = {
            "id": rule_id.strip(),
            "category": category,
            "enabled": enabled,
            "patterns": patterns,
            "action": action,
        }

        try:
            moderator.add_rule(new_rule)
        except ValueError as error:
            st.error(str(error))
            return

        st.success(f"Правило '{rule_id}' создано.")
        st.rerun()

    else:
        if not rule_ids:
            st.info("Нет существующих правил.")
            return

        selected_rule_id = st.selectbox(
            "Выберите существующее правило",
            rule_ids,
        )

        selected_rule = moderator.get_rule(selected_rule_id)

        st.info(
            "Регулярки будут добавлены в выбранное правило. "
            "Категория и действие возьмутся из этого правила."
        )

        st.write("**Категория правила:**", selected_rule.get("category"))
        st.write("**Действие правила:**", selected_rule.get("action"))

        with st.form("append_patterns_form"):
            patterns_text = st.text_area(
                "Новые регулярные выражения",
                height=150,
                placeholder=(
                    "Каждое регулярное выражение с новой строки.\n"
                    "Например:\n"
                    "\\bдурак\\b"
                ),
            )

            submitted = st.form_submit_button("Добавить регулярки")

        if not submitted:
            return

        patterns = [
            line.strip()
            for line in patterns_text.splitlines()
            if line.strip()
        ]

        if not patterns:
            st.error("Добавьте хотя бы одно регулярное выражение.")
            return

        pattern_errors = compile_patterns(patterns)

        if pattern_errors:
            st.error("Есть ошибки в регулярных выражениях:")
            for error in pattern_errors:
                st.code(error)
            return

        try:
            moderator.add_patterns_to_rule(selected_rule_id, patterns)
        except ValueError as error:
            st.error(str(error))
            return
        except re.error as error:
            st.error(f"Ошибка в регулярном выражении: {error}")
            return

        st.success(f"Регулярки добавлены в правило '{selected_rule_id}'.")
        st.rerun()


def show_delete_rule_form(moderator: RuleBasedModerator) -> None:
    """Форма удаления правила."""
    st.subheader("Удалить правило")

    rules = moderator.get_rules()
    rule_ids = [
        rule.get("id")
        for rule in rules
        if rule.get("id")
    ]

    if not rule_ids:
        st.info("Нет правил для удаления.")
        return

    selected_rule_id = st.selectbox(
        "Выберите правило",
        rule_ids,
    )

    confirm_delete = st.checkbox(
        "Я понимаю, что правило будет удалено из JSON-файла",
    )

    if st.button("Удалить правило", type="secondary"):
        if not confirm_delete:
            st.warning("Подтвердите удаление правила.")
            return

        deleted = moderator.delete_rule(selected_rule_id)

        if deleted:
            st.cache_resource.clear()
            st.success(f"Правило '{selected_rule_id}' удалено.")
            st.rerun()

        st.error("Правило не найдено.")


def show_delete_pattern_form(moderator: RuleBasedModerator) -> None:
    """Форма удаления отдельного регулярного выражения из правила."""
    st.subheader("Удалить регулярное выражение")

    rules = moderator.get_rules()

    rules_with_patterns = [
        rule
        for rule in rules
        if rule.get("id") and isinstance(rule.get("patterns"), list) and rule.get("patterns")
    ]

    if not rules_with_patterns:
        st.info("Нет правил с регулярными выражениями.")
        return

    rule_ids = [
        rule["id"]
        for rule in rules_with_patterns
    ]

    selected_rule_id = st.selectbox(
        "Выберите правило",
        rule_ids,
        key="delete_pattern_rule_select",
    )

    selected_rule = moderator.get_rule(selected_rule_id)

    if selected_rule is None:
        st.error("Правило не найдено.")
        return

    patterns = selected_rule.get("patterns", [])

    if not patterns:
        st.info("В выбранном правиле нет регулярных выражений.")
        return

    st.write("**Категория правила:**", selected_rule.get("category"))
    st.write("**Действие правила:**", selected_rule.get("action"))

    selected_pattern = st.selectbox(
        "Выберите регулярное выражение",
        patterns,
        key="delete_pattern_select",
    )

    st.write("**Выбранное регулярное выражение:**")
    st.code(selected_pattern)

    confirm_delete = st.checkbox(
        "Я понимаю, что регулярное выражение будет удалено из JSON-файла",
        key="confirm_delete_pattern",
    )

    if st.button("Удалить регулярное выражение", type="secondary"):
        if not confirm_delete:
            st.warning("Подтвердите удаление регулярного выражения.")
            return

        deleted = moderator.delete_pattern_from_rule(
            selected_rule_id,
            selected_pattern,
        )

        if deleted:
            st.cache_resource.clear()
            st.success("Регулярное выражение удалено.")
            st.rerun()

        st.error("Регулярное выражение не найдено.")


def show_export_block() -> None:
    """Показывает путь к файлу правил и кнопку выгрузки."""
    st.subheader("Файл правил")

    st.write("**Путь к файлу:**")
    st.code(str(RULES_PATH))

    if not RULES_PATH.exists():
        st.warning("Файл правил не найден.")
        return

    rules_json = read_file_text(RULES_PATH)

    st.download_button(
        label="Выгрузить JSON",
        data=rules_json,
        file_name="moderation_rules.json",
        mime="application/json",
    )

    with st.expander("Показать JSON-файл"):
        st.code(rules_json, language="json")


def show_rule_based_checker(moderator: RuleBasedModerator) -> None:
    """Отдельная проверка сообщения только через rule-based слой."""
    st.subheader("Проверка сообщения")

    with st.form("rule_based_check_form"):
        text = st.text_area(
            "Введите сообщение:",
            height=120,
            placeholder="Например: Зайди на https://example.com",
        )
        st.caption("Подсказка: Ctrl+Enter запускает проверку.")
        submitted = st.form_submit_button("Проверить сообщение")

    if not submitted:
        return

    if not text.strip():
        st.warning("Введите текст сообщения.")
        return

    result = moderator.process(text)

    if result["has_violation"]:
        st.warning("Нарушение найдено.")
    else:
        st.success("Нарушений не найдено.")

    left_column, right_column = st.columns(2)

    with left_column:
        st.write("**Категория:**", result["violation_category"])
        st.write("**Рекомендуемое действие:**", result["recommended_action"])

    with right_column:
        st.write("**Сработавший паттерн:**", result["matched_pattern"])
        st.write("**Нормализованный текст:**", result["normalized_text"])

    with st.expander("Показать JSON результата"):
        st.code(
            json.dumps(result, ensure_ascii=False, indent=2),
            language="json",
        )


def main() -> None:
    st.title("Rule-based слой модерации")

    st.write(
        "На этой странице можно:\n" \
        "- просматривать правила;\n" \
        "- добавлять новые правила;\n" \
        "- удалять правила;\n" \
        "- проверять сообщения только через данный слой;\n"\
        "- выгрузить JSON-файл с правилами."
    )

    moderator = load_moderator()

    tab_rules, tab_check, tab_edit, tab_file = st.tabs([
        "Список правил",
        "Проверка",
        "Добавить / удалить",
        "Файл правил",
    ])

    with tab_rules:
        show_rules_table(moderator)

    with tab_check:
        show_rule_based_checker(moderator)

    with tab_edit:
        left_column, right_column = st.columns(2)

        with left_column:
            show_add_rule_form(moderator)

        with right_column:
            show_delete_pattern_form(moderator)
            st.divider()
            show_delete_rule_form(moderator)

    with tab_file:
        show_export_block()


if __name__ == "__main__":
    main()
