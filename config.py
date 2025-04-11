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
    context_max_tokens = IntegerField("CONTEXT_MAX_TOKENS", default=4096)
    preferred_language = StringField("PREFERRED_LANGUAGE", default="Spanish")
    add_no_answer = BooleanField("ADD_NO_ANSWER", default=False)
    rate_limiter_requests_per_second = FloatField("RATE_LIMITER_REQUESTS_PER_SECOND", default=0.25)
    rate_limiter_check_every_n_seconds = FloatField("RATE_LIMITER_CHECK_EVERY_N_SECONDS", default=0.1)
    rate_limiter_max_bucket_size = IntegerField("RATE_LIMITER_MAX_BUCKET_SIZE", default=10)
    is_image_multimodal = BooleanField("ENABLE_MULTIMODAL", default=False)
    is_group_assistant = BooleanField("ENABLE_GROUP_ASSISTANT", default=False)
    sdapi_url = StringField(
        "WEBUI_SD_API_URL", warning="WEBUI_SD_API_URL environment variable not set. Image generation disabled."
    )
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
        warning="Could not load WEBUI_SD_API_PARAMS. Defaults for SDXL Turbo model will be used.",
    )
    sdapi_negative_prompt = StringField("WEBUI_SD_API_NEGATIVE_PROMPT")
    allowed_chat_ids = StringListField("TELEGRAM_ALLOWED_CHATS")
    bot_instructions = StringField("TELEGRAM_BOT_INSTRUCTIONS")
    bot_instructions_character = StringField("TELEGRAM_BOT_INSTRUCTIONS_CHARACTER")
    bot_instructions_extra = StringField("TELEGRAM_BOT_INSTRUCTIONS_EXTRA")
    simulate_typing = BooleanField("SIMULATE_TYPING", default=False)
    simulate_typing_wpm = IntegerField("SIMULATE_TYPING_WPM", default=100)
    simulate_typing_max_time = IntegerField("SIMULATE_TYPING_MAX_TIME", default=10)

    use_tools = BooleanField("USE_TOOLS", default=False)

    logging_level = StringField(
        "LOGGING_LEVEL", default="INFO", allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
