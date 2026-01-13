# src/utils/models.py
from functools import lru_cache
from config import MODEL_PROVIDER, RESPONSE_MODEL, RERANK_MODEL


def _create_chat_model(model_name: str, temperature: float = 0.0):
    """Factory function to create chat model based on provider."""

    if MODEL_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, temperature=temperature)

    elif MODEL_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)

    elif MODEL_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)

    elif MODEL_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


# Response model (cached)
@lru_cache(maxsize=1)
def get_response_model():
    return _create_chat_model(RESPONSE_MODEL, temperature=0.0)


# Rerank model (cached)
@lru_cache(maxsize=1)
def get_rerank_model():
    return _create_chat_model(RERANK_MODEL, temperature=0.0)
