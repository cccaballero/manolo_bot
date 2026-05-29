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
    preferred_language: str = "English"
    add_no_answer: bool = False
    is_image_multimodal: bool = False
    is_audio_multimodal: bool = False
    is_group_assistant: bool = False
    agent_mode: bool = False

    # Web content retrieval configuration
    web_content_request_timeout: int = 10

    can_use_tavily_search: bool = False

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
