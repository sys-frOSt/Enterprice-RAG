from langchain_groq import ChatGroq

from app.config import Settings


def create_groq(settings: Settings) -> ChatGroq:
    settings.validate_startup()
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
        timeout=settings.request_timeout_seconds,
        max_retries=3,
    )
