"""Embeddings factory auto-picking a provider from EmbeddingsConfig (or LLMConfig)."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from manolo_bot.ai.config import EmbeddingsConfig, LLMConfig

# NOTE: `models/text-embedding-004` was retired by Google in Jan 2026 (404).
# `gemini-embedding-001` is sunset May 14 2028; default is now gemini-embedding-2 (GA Apr 2026).
GOOGLE_DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"
# Ollama's docs now promote `embeddinggemma` (was `nomic-embed-text`).
OLLAMA_DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
# OpenAI's `text-embedding-ada-002` is legacy; default is now 3-small (5× cheaper).
OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def build_embeddings(
    llm_config: LLMConfig | None = None,
    embeddings_config: EmbeddingsConfig | None = None,
    model: str | None = None,
) -> Embeddings:
    """Build an embeddings implementation.

    Granularity of control, most specific first:

    :param embeddings_config: Preferred embeddings-specific configuration.
        No provider field: the provider is inferred from which settings are
        present, in order google -> openai -> ollama (mirroring the LLM
        config pattern). Each provider uses its own key and model fields
        (``google_api_key`` / ``google_embedding_model``, ``openai_api_key`` +
        ``openai_api_base_url`` + ``openai_embedding_model``,
        ``ollama_model``). For OpenAI-compatible endpoints that need no key
        (LM Studio, Ollama, vLLM), a missing ``openai_api_key`` falls back to
        the ``OPENAI_API_KEY`` env var, then to the placeholder ``"not-needed"``
        (a ``base_url`` alone is allowed).
    :param llm_config: Deprecated, retained for backwards compatibility:
        derives the embeddings provider from the LLM config
        (google -> openai or base URL -> ollama by available credentials).
        Ignored when ``embeddings_config`` is given.
    :param model: Optional embedding model name override for the selected
        provider (generic fallback for any provider).
    :raises ValueError: When no provider is configured (pass at least one
        provider's key/model in ``embeddings_config``, or a legacy
        ``llm_config``).
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
    """Build embeddings from an EmbeddingsConfig.

    The provider is inferred from which settings are present (google ->
    openai -> ollama), mirroring the LLM config pattern — there is no explicit
    provider field.
    """
    if embeddings_config.google_api_key:
        return GoogleGenerativeAIEmbeddings(
            model=embeddings_config.google_embedding_model or model or GOOGLE_DEFAULT_EMBEDDING_MODEL,
            google_api_key=embeddings_config.google_api_key,
        )
    if embeddings_config.openai_api_key or embeddings_config.openai_api_base_url:
        kwargs: dict = {"model": embeddings_config.openai_embedding_model or model or OPENAI_DEFAULT_EMBEDDING_MODEL}
        if embeddings_config.openai_api_key:
            kwargs["openai_api_key"] = embeddings_config.openai_api_key
        if embeddings_config.openai_api_base_url:
            kwargs["openai_api_base"] = embeddings_config.openai_api_base_url
        if "openai_api_key" not in kwargs and not os.environ.get("OPENAI_API_KEY"):
            # Local OpenAI-compatible servers often need no key, but the
            # client requires one to be set.
            kwargs["openai_api_key"] = "not-needed"
        return OpenAIEmbeddings(**kwargs)
    if embeddings_config.ollama_model:
        return OllamaEmbeddings(model=embeddings_config.ollama_model)
    raise ValueError("No embeddings provider configured (need google/openai/ollama settings)")


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
