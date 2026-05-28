import re

# Удаляемые шаблоны для rule-based предобработки
EXTRA_SPACES_PATTERN = r"\s+"

# Удаляемые шаблоны для ML-предобработки.
URL_PATTERN = r"https?://\S+|www\.\S+"
EMAIL_PATTERN = r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
UNSUPPORTED_CHARS_PATTERN = r"[^а-яa-z0-9\s]"
# + EXTRA_SPACES_PATTERN


def normalize_text(text: str | None) -> str:
    """Базовая нормализация текста.

    Используется как общая часть предобработки:
    - None заменяется на пустую строку;
    - текст приводится к строковуому типу;
    - символ 'ё' заменяется на 'е';
    - текст приводится к нижнему регистру.

    Функция ничего не удаляет из текста.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = text.replace("ё", "е")

    return text


def preprocess_text_for_rules(text: str | None) -> str:
    """Предобработка текста для rule-based слоя.

    Включает в себя:
    - приведение к нижнему регистру;
    - замена 'ё' на 'е';
    - удаление лишних пробелов.
    """
    text = normalize_text(text)
    text = re.sub(EXTRA_SPACES_PATTERN, " ", text).strip()

    return text


def preprocess_text_for_ml(text: str | None) -> str:
    """Предобработка текста для ML-моделей.

    Включает в себя:
    - приведение к нижнему регистру;
    - замена 'ё' на 'е';
    - удаление URL;
    - удаление email;
    - удаление неподдерживаемых символов;
    - удаление лишних пробелов.
    """
    text = normalize_text(text)

    text = re.sub(URL_PATTERN, " ", text)
    text = re.sub(EMAIL_PATTERN, " ", text)
    text = re.sub(UNSUPPORTED_CHARS_PATTERN, " ", text)
    text = re.sub(EXTRA_SPACES_PATTERN, " ", text).strip()

    return text
