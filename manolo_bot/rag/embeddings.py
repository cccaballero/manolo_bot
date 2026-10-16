"""Embeddings factory auto-picking a provider from EmbeddingsConfig (or LLMConfig)."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from manolo_bot.ai.config import EmbeddingsConfig, LLMConfig

# NOTE: `models/text-embedding-004` was retired by Google in Jan 2026 and now
# returns 404. `gemini-embedding-001` is the current stable text embedding model.
GOOGLE_DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
OLLAMA_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


def build_embeddings(
    llm_config: LLMConfig | None = None,
    embeddings_config: EmbeddingsConfig | None = None,
    model: str | None = None,
) -> Embeddings:
    """Build an embeddings implementation.

    Granularity of control, most specific first:

    :param embeddings_config: Preferred embeddings-specific configuration.
        ``provider`` selects the backend: ``"google"``, ``"openai"`` or
        ``"ollama"``. An empty ``provider`` means auto: credentials are probed
        in order google -> openai -> ollama. Each provider uses its own key
        fields (``google_api_key`` / ``openai_api_key`` + ``openai_api_base_url``
        / ``ollama_model``). For OpenAI-compatible endpoints that need no key
        (LM Studio, Ollama, vLLM), a missing ``openai_api_key`` falls back to
        the ``OPENAI_API_KEY`` env var, then to the placeholder ``"not-needed"``
        (a ``base_url`` alone is allowed). An explicit ``provider`` with no
        usable credentials raises ValueError.
    :param llm_config: Deprecated, retained for backwards compatibility:
        derives the embeddings provider from the LLM config
        (google -> openai or base URL -> ollama by available credentials).
        Ignored when ``embeddings_config`` is given.
    :param model: Optional embedding model name override for the selected
        provider (also settable via ``RAG_EMBEDDING_MODEL``).
    :raises ValueError: When no provider is configured (or an unknown
        provider name is given in ``embeddings_config``).
    """
    if embeddings_config is not None:
        return _build_from_embeddings_config(embeddings_config, model=model)
    if llm_config is not None:
        return _build_from_llm_config(llm_config, model=model)
    raise ValueError(
        "No embeddings provider configured (pass embeddings_config with "
        "google/openai/ollama settings, or a legacy llm_config)"
    )


def _build_from_embeddings_config(
    embeddings_config: EmbeddingsConfig,
    model: str | None = None,
) -> Embeddings:
    """Build embeddings from an EmbeddingsConfig (explicit or auto provider)."""
    provider = (embeddings_config.provider or "").strip().lower()
    if provider == "":
        # Auto: probe credentials in order google -> openai -> ollama.
        if embeddings_config.google_api_key:
            provider = "google"
        elif embeddings_config.openai_api_key or embeddings_config.openai_api_base_url:
            provider = "openai"
        elif embeddings_config.ollama_model:
            provider = "ollama"
        else:
            raise ValueError("No embeddings provider configured (need google/openai/ollama settings)")

    if provider == "google":
        if not embeddings_config.google_api_key:
            raise ValueError("Embeddings provider 'google' requires google_api_key")
        return GoogleGenerativeAIEmbeddings(
            model=embeddings_config.model or model or GOOGLE_DEFAULT_EMBEDDING_MODEL,
            google_api_key=embeddings_config.google_api_key,
        )
    if provider == "openai":
        kwargs: dict = {"model": embeddings_config.model or model or OPENAI_DEFAULT_EMBEDDING_MODEL}
        if embeddings_config.openai_api_key:
            kwargs["openai_api_key"] = embeddings_config.openai_api_key
        if embeddings_config.openai_api_base_url:
            kwargs["openai_api_base"] = embeddings_config.openai_api_base_url
        if "openai_api_key" not in kwargs and not os.environ.get("OPENAI_API_KEY"):
            # Local OpenAI-compatible servers often need no key, but the
            # client requires one to be set.
            kwargs["openai_api_key"] = "not-needed"
        return OpenAIEmbeddings(**kwargs)
    if provider == "ollama":
        # NOTE: the chat model (e.g. gemma3) is not an embedding model;
        # default to nomic-embed-text unless explicitly overridden.
        return OllamaEmbeddings(model=embeddings_config.model or model or OLLAMA_DEFAULT_EMBEDDING_MODEL)
    raise ValueError(f"Unknown embeddings provider: {provider!r}")


def _build_from_llm_config(llm_config: LLMConfig, model: str | None = None) -> Embeddings:
    """Legacy path: derive the embeddings provider from an LLMConfig.

    Kept byte-for-byte compatible with the pre-EmbeddingsConfig behavior so
    existing callers and tests keep working.
    """
    if getattr(llm_config, "google_api_key", ""):
        return GoogleGenerativeAIEmbeddings(
            model=model or GOOGLE_DEFAULT_EMBEDDING_MODEL,
            google_api_key=llm_config.google_api_key,
        )
    if getattr(llm_config, "openai_api_key", "") or getattr(llm_config, "openai_api_base_url", ""):
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
