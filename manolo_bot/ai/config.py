from dataclasses import dataclass, field


@dataclass
class BotConfig:
    bot_uuid: str
    bot_name: str
    bot_username: str
    bot_token: str
    user_id: int
    agent_instructions: str | None = None

    allowed_chat_ids: list = field(default_factory=list)
    bot_instructions: str = ""
    bot_instructions_character: str = ""
    bot_instructions_extra: str = ""

    simulate_typing: bool = True
    simulate_typing_wpm: int = 100
    simulate_typing_max_time: int = 10

    use_tools: bool = False

    # MCP Configuration
    enable_mcp: bool = False
    mcp_servers_config: dict = field(default_factory=dict)

    context_max_tokens: int = 4096
    context_summarization: bool = True
    summary_max_tokens: int = 512
    summary_keep_messages: int = 6
    preferred_language: str = "English"
    add_no_answer: bool = False
    is_image_multimodal: bool = False
    is_audio_multimodal: bool = False
    is_document_multimodal: bool = False
    is_group_assistant: bool = False
    agent_mode: bool = False

    # Agent history persistence policy ("final-only" | "full-trace"). Thin
    # Telegram-side plumbing: main.py maps this string to a policy instance.
    # NOTE: the policy lives INSIDE the truncated queue, so "full-trace" does
    # not make tool traces durable against truncation (that needs an
    # outside-queue sidecar). Kept open to a future pinned/never-truncate class.
    history_policy: str = "final-only"

    # Web content retrieval configuration
    web_content_request_timeout: int = 10
    max_document_size: int = 2 * 1024 * 1024
    max_voice_size: int = 2 * 1024 * 1024

    # Header authentication for attachment downloads (image/audio/document
    # file URLs). Some platforms require auth headers to fetch file URLs.
    attachment_headers: dict = field(default_factory=dict)

    can_use_tavily_search: bool = False

    # Tavily Extract: prefer server-side fetching for web content / URL
    # documents and YouTube transcripts, falling back to the direct path on
    # failure. ``tavily_api_key`` comes from the env boundary
    # (TAVILY_API_KEY) and is shared with Tavily Search.
    can_use_tavily_extract: bool = False
    can_use_tavily_extract_for_youtube: bool = False
    tavily_api_key: str = ""

    # Stable Diffusion configuration
    sdapi_url: str = ""
    sdapi_params: dict = field(default_factory=dict)
    sdapi_negative_prompt: str = ""


@dataclass
class LLMConfig:
    google_api_key: str
    google_api_model: str
    openai_api_key: str
    openai_api_model: str
    openai_api_base_url: str
    ollama_model: str

    rate_limiter_requests_per_second: float = 0.25
    rate_limiter_check_every_n_seconds: float = 0.1
    rate_limiter_max_bucket_size: int = 10


@dataclass
class EmbeddingsConfig:
    """Embeddings provider configuration.

    Mirrors :class:`LLMConfig` for the RAG knowledge base: there is no
    provider field — the provider is inferred from which settings are present
    (google -> openai -> ollama). Per-provider model fields follow the
    ``*_API_MODEL`` / ``OLLAMA_MODEL`` pattern of the chat config.
    """

    google_api_key: str = ""
    google_embedding_model: str = ""
    openai_api_key: str = ""
    openai_api_base_url: str = ""
    openai_embedding_model: str = ""
    ollama_model: str = ""
