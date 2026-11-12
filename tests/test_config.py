import pytest

from app.config import Settings


def test_startup_validation_requires_groq_key() -> None:
    settings = Settings(groq_api_key="")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        settings.validate_startup()

