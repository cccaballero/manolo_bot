"""Embeddings factory auto-picking a provider from LLMConfig (Phase 1)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from manolo_bot.ai.config import LLMConfig

# NOTE: `models/text-embedding-004` was retired by Google in Jan 2026 and now
# returns 404. `gemini-embedding-001` is the current stable text embedding model.
GOOGLE_DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
OLLAMA_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


def build_embeddings(llm_config: LLMConfig, model: str | None = None) -> Embeddings:
    """Build an embeddings implementation from an LLMConfig.

    The backend always follows the configured LLM provider
    (google -> openai -> ollama by available credentials); only the model
    name can be overridden — e.g. an OpenAI-compatible endpoint (LM Studio,
    Ollama, vLLM) that does not host OpenAI's own embedding models.

    :param llm_config: LLM configuration with provider credentials/models.
    :param model: Optional embedding model name for the detected provider
        (also settable via ``RAG_EMBEDDING_MODEL``).
    :raises ValueError: When no provider is configured.
    """
    if getattr(llm_config, "google_api_key", ""):
        return GoogleGenerativeAIEmbeddings(
            model=model or GOOGLE_DEFAULT_EMBEDDING_MODEL,
            google_api_key=llm_config.google_api_key,
        )
    if getattr(llm_config, "openai_api_key", "") or getattr(llm_config, "openai_api_base_url", ""):
        import os

        kwargs: dict = {"model": model or OPENAI_DEFAULT_EMBEDDING_MODEL}
        if llm_config.openai_api_key:
            kwargs["openai_api_key"] = llm_config.openai_api_key
        if llm_config.openai_api_base_url:
            kwargs["openai_api_base"] = llm_config.openai_api_base_url
        if "openai_api_key" not in kwargs and not os.environ.get("OPENAI_API_KEY"):
            # Local OpenAI-compatible servers often need no key, but the
            # client requires one to be set.
            kwargs["openai_api_key"] = "not-needed"
        return OpenAIEmbeddings(**kwargs)
    if getattr(llm_config, "ollama_model", ""):
        # NOTE: the chat model (e.g. gemma3) is not an embedding model;
        # default to nomic-embed-text unless explicitly overridden.
        return OllamaEmbeddings(model=model or OLLAMA_DEFAULT_EMBEDDING_MODEL)
    raise ValueError("No embeddings provider configured (need google/openai/ollama settings)")
