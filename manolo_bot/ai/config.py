import os
import tempfile
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

    # Web content retrieval configuration
    web_content_request_timeout: int = 10
    max_document_size: int = 2 * 1024 * 1024
    max_voice_size: int = 2 * 1024 * 1024

    can_use_tavily_search: bool = False

    # Stable Diffusion configuration
    sdapi_url: str = ""
    sdapi_params: dict = field(default_factory=dict)
    sdapi_negative_prompt: str = ""

    # RAG configuration (agent/deep_agent modes only; LLMBot ignores these)
    rag_enabled: bool = False
    rag_backend: str = "in_memory"
    rag_sources: list = field(default_factory=list)
    rag_store_path: str = os.path.join(tempfile.gettempdir(), "manolo_bot", "rag")
    rag_top_k: int = 5
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_reindex: str = "auto"
    rag_embedding_model: str = ""
    rag_max_file_bytes: int = 10 * 1024 * 1024  # 10MB; 0 = unlimited


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
