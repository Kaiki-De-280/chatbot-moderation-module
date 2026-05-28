import streamlit as st


main_page = st.Page(
    "pages/main_page.py",
    title="Главная",
    icon="🛡️",
    default=True,
)

rule_based_page = st.Page(
    "pages/rule_based_layer.py",
    title="Правило-ориентированный слой",
    icon="📋",
)

ml_page = st.Page(
    "pages/ml_layer.py",
    title="ML-слой",
    icon="🤖",
)

dataset_page = st.Page(
    "pages/dataset_evaluation.py",
    title="Тестирование на датасете",
    icon="📊",
)

page = st.navigation([
    main_page,
    rule_based_page,
    ml_page,
    dataset_page,
])

page.run()
