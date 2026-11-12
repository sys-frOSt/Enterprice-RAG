from app.config import Settings
from app.llm import groq


def test_groq_uses_groq_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatGroq:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(groq, "ChatGroq", FakeChatGroq)
    settings = Settings(groq_api_key="groq-test", groq_model="llama-test")

    groq.create_groq(settings)

    assert captured["api_key"] == "groq-test"
    assert captured["model"] == "llama-test"
    assert captured["temperature"] == 0
