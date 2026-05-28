from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = PROJECT_ROOT / "rules" / "moderation_rules.json"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "models" / "model_registry.json"


def add_project_root_to_path() -> None:
    """Добавляет корень проекта в sys.path."""
    project_root_str = str(PROJECT_ROOT)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def read_file_text(path: Path) -> str:
    """Читает текстовый файл."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
