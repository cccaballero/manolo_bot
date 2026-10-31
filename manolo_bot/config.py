import os
import tempfile

from envmodel import (
    BooleanField,
    EnvModel,
    FloatField,
    IntegerField,
    JsonField,
    StringField,
    StringListField,
)


class Config(EnvModel):
    google_api_key = StringField("GOOGLE_API_KEY")
    google_api_model = StringField("GOOGLE_API_MODEL", default="gemini-2.0-flash")
    openai_api_key = StringField("OPENAI_API_KEY")
    openai_api_model = StringField("OPENAI_API_MODEL")
    openai_api_base_url = StringField("OPENAI_API_BASE_URL")
    ollama_model = StringField("OLLAMA_MODEL")
    bot_name = StringField("TELEGRAM_BOT_NAME", required=True)
    bot_username = StringField("TELEGRAM_BOT_USERNAME", required=True)
    bot_token = StringField("TELEGRAM_BOT_TOKEN", required=True)
    bot_uuid = StringField("BOT_UUID", default="default-bot-uuid")
    user_id = IntegerField("USER_ID", default=0)
    context_max_tokens = IntegerField("CONTEXT_MAX_TOKENS", default=4096)
    enable_context_summarization = BooleanField("ENABLE_CONTEXT_SUMMARIZATION", default=True)
    summary_max_tokens = IntegerField("SUMMARY_MAX_TOKENS", default=512)
    summary_keep_messages = IntegerField("SUMMARY_KEEP_MESSAGES", default=6)
    preferred_language = StringField("PREFERRED_LANGUAGE", default="Spanish")
    add_no_answer = BooleanField("ADD_NO_ANSWER", default=False)
    rate_limiter_requests_per_second = FloatField("RATE_LIMITER_REQUESTS_PER_SECOND", default=0.25)
    rate_limiter_check_every_n_seconds = FloatField("RATE_LIMITER_CHECK_EVERY_N_SECONDS", default=0.1)
    rate_limiter_max_bucket_size = IntegerField("RATE_LIMITER_MAX_BUCKET_SIZE", default=10)
    is_image_multimodal = BooleanField("IMAGE_MULTIMODAL", default=False)
    is_audio_multimodal = BooleanField("AUDIO_MULTIMODAL", default=False)
    is_document_multimodal = BooleanField("DOCUMENT_MULTIMODAL", default=False)
    is_group_assistant = BooleanField("ENABLE_GROUP_ASSISTANT", default=False)
    agent_mode = BooleanField("AGENT_MODE", default=False)
    ai_mode = StringField("AI_MODE", default="", allowed_values=["", "llm", "agent", "deep_agent"])
    agent_instructions = StringField("AGENT_INSTRUCTIONS")
    # Agent history persistence policy, parsed at the env boundary (Telegram
    # adapter only; library callers pass policy instances directly).
    agent_history_policy = StringField(
        "AGENT_HISTORY_POLICY", default="final-only", allowed_values=["final-only", "full-trace"]
    )
    allow_private_chats = BooleanField("ALLOW_PRIVATE_CHATS", default=True)

    @property
    def effective_ai_mode(self) -> str:
        """Resolve the effective AI mode, handling backward compatibility with AGENT_MODE.

        Precedence:
          1. ``AI_MODE`` if set to a non-empty value
          2. ``"agent"`` when ``AGENT_MODE`` is True (backward-compat)
          3. ``"agent"`` by default (documented default)
        """
        if self.ai_mode:
            return self.ai_mode
        # Backward compatibility: AGENT_MODE=True selects "agent".
        # The default is "agent" (per docs in env.example).
        return "agent"

    # Web content retrieval configuration
    web_content_request_timeout = IntegerField("WEB_CONTENT_REQUEST_TIMEOUT_SECONDS", default=10)

    sdapi_url = StringField("WEBUI_SD_API_URL")
    default_sdapi_params = {
        "steps": 1,
        "cfg_scale": 1,
        "width": 512,
        "height": 512,
        "timestep_spacing": "trailing",
    }
    sdapi_params = JsonField(
        "WEBUI_SD_API_PARAMS",
        default=default_sdapi_params,
    )
    sdapi_negative_prompt = StringField("WEBUI_SD_API_NEGATIVE_PROMPT")
    allowed_chat_ids = StringListField("TELEGRAM_ALLOWED_CHATS")
    bot_instructions = StringField("TELEGRAM_BOT_INSTRUCTIONS")
    bot_instructions_character = StringField("TELEGRAM_BOT_INSTRUCTIONS_CHARACTER")
    bot_instructions_extra = StringField("TELEGRAM_BOT_INSTRUCTIONS_EXTRA")
    simulate_typing = BooleanField("SIMULATE_TYPING", default=False)
    simulate_typing_wpm = IntegerField("SIMULATE_TYPING_WPM", default=100)
    simulate_typing_max_time = IntegerField("SIMULATE_TYPING_MAX_TIME", default=10)

    use_tools = BooleanField("USE_TOOLS", default=True)

    # MCP Configuration
    enable_mcp = BooleanField("ENABLE_MCP", default=False)
    mcp_servers_config = JsonField(
        "MCP_SERVERS_CONFIG",
        default={},
        warning="MCP_SERVERS_CONFIG not set. MCP will be disabled.",
    )

    # File size limits
    max_document_size = IntegerField("MAX_DOCUMENT_SIZE_BYTES", default=2 * 1024 * 1024)  # 2MB
    max_voice_size = IntegerField("MAX_VOICE_SIZE_BYTES", default=2 * 1024 * 1024)  # 2MB

    # Document storage
    document_storage_path = StringField(
        "DOCUMENT_STORAGE_PATH", default=os.path.join(tempfile.gettempdir(), "manolo_bot", "documents")
    )

    # Deep Agent workspace (virtual filesystem root, separate from document storage)
    deep_agent_workspace_path = StringField(
        "DEEP_AGENT_WORKSPACE_PATH", default=os.path.join(tempfile.gettempdir(), "manolo_bot", "workspace")
    )
    deep_agent_backend = StringField("DEEP_AGENT_BACKEND", default="local_fs", allowed_values=["in_memory", "local_fs"])

    # Deep Agent skills (comma-separated paths; entries may be `<path>::LABEL=<text>`)
    deep_agent_skills_paths = StringListField("DEEP_AGENT_SKILLS_PATHS", default=[])

    # Deep Agent memory (per-chat AGENTS.md files rooted under this directory)
    deep_agent_memory_path = StringField(
        "DEEP_AGENT_MEMORY_PATH", default=os.path.join(tempfile.gettempdir(), "manolo_bot", "memory")
    )
    deep_agent_memory_add_cache_control = BooleanField("DEEP_AGENT_MEMORY_ADD_CACHE_CONTROL", default=False)

    # RAG (Retrieval-Augmented Generation) — agent/deep_agent modes only
    rag_enabled = BooleanField("RAG_ENABLED", default=False)
    rag_backend = StringField("RAG_BACKEND", default="in_memory", allowed_values=["in_memory", "local_fs", "redis"])
    rag_sources = StringListField("RAG_SOURCES", default=[])
    rag_store_path = StringField("RAG_STORE_PATH", default=os.path.join(tempfile.gettempdir(), "manolo_bot", "rag"))
    rag_top_k = IntegerField("RAG_TOP_K", default=5)
    rag_chunk_size = IntegerField("RAG_CHUNK_SIZE", default=1000)
    rag_chunk_overlap = IntegerField("RAG_CHUNK_OVERLAP", default=200)
    rag_reindex = StringField("RAG_REINDEX", default="auto", allowed_values=["auto", "always", "never"])
    rag_google_api_key = StringField("RAG_GOOGLE_API_KEY", default="")
    rag_google_embedding_model = StringField("RAG_GOOGLE_EMBEDDING_MODEL", default="")
    rag_openai_api_key = StringField("RAG_OPENAI_API_KEY", default="")
    rag_openai_api_base_url = StringField("RAG_OPENAI_API_BASE_URL", default="")
    rag_openai_embedding_model = StringField("RAG_OPENAI_EMBEDDING_MODEL", default="")
    rag_ollama_model = StringField("RAG_OLLAMA_MODEL", default="")
    rag_max_file_bytes = IntegerField("RAG_MAX_FILE_BYTES", default=10 * 1024 * 1024)  # 10MB

    # Scheduler (agent/deep_agent modes only). The app composes a store + engine
    # into a SchedulerService at startup and injects it explicitly into the
    # agents, mirroring the RAG/history-policy pattern.
    scheduler_enabled = BooleanField("SCHEDULER_ENABLED", default=False)
    scheduler_store = StringField(
        "SCHEDULER_STORE", default="in_memory", allowed_values=["in_memory", "local_fs", "redis"]
    )
    scheduler_engine = StringField("SCHEDULER_ENGINE", default="asyncio", allowed_values=["asyncio", "manual"])
    scheduler_store_path = StringField(
        "SCHEDULER_STORE_PATH", default=os.path.join(tempfile.gettempdir(), "manolo_bot", "scheduler")
    )
    scheduler_timezone = StringField("SCHEDULER_TIMEZONE", default="UTC")
    scheduler_poll_interval_seconds = FloatField("SCHEDULER_POLL_INTERVAL_SECONDS", default=5.0)
    scheduler_max_actions_per_chat = IntegerField("SCHEDULER_MAX_ACTIONS_PER_CHAT", default=20)
    scheduler_max_consecutive_failures = IntegerField("SCHEDULER_MAX_CONSECUTIVE_FAILURES", default=5)
    scheduler_catchup_grace_seconds = IntegerField("SCHEDULER_CATCHUP_GRACE_SECONDS", default=0)

    logging_level = StringField(
        "LOGGING_LEVEL", default="INFO", allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )

    use_tavily_search = BooleanField("USE_TAVILY_SEARCH", default=False)
    # Tavily Extract: optional server-side fetching for web/URL-document content
    # and YouTube transcripts. Both flags are independent and fall back to the
    # direct implementations on failure or when no key is configured.
    use_tavily_extract = BooleanField("USE_TAVILY_EXTRACT", default=False)
    use_tavily_extract_for_youtube = BooleanField("USE_TAVILY_EXTRACT_FOR_YOUTUBE", default=False)
    # One key for both Tavily features (search + extract).
    tavily_api_key = StringField("TAVILY_API_KEY", default="")
    # Apify: optional YouTube transcripts through the johnvc/YoutubeTranscripts
    # Actor. Falls back to the direct implementation on failure or without a token.
    use_apify_for_youtube = BooleanField("USE_APIFY_FOR_YOUTUBE", default=False)
    apify_api_token = StringField("APIFY_API_TOKEN", default="")

    storage_type = StringField("STORAGE_TYPE", default="memory", allowed_values=["memory", "redis"])
    redis_url = StringField("REDIS_URL", default="redis://localhost:6379/0")
