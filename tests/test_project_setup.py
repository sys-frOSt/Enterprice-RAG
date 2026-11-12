from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_environment_template_contains_required_settings() -> None:
    template = (ROOT / ".env.example").read_text(encoding="utf-8")

    settings = ("GROQ_API_KEY", "GROQ_MODEL", "QDRANT_URL", "QDRANT_API_KEY", "COLLECTION_NAME")
    for setting in settings:
        assert f"{setting}=" in template
