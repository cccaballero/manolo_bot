# Telegram Chat Bot using LLM

This is an experimental Telegram chat bot that uses a configurable LLM model to generate responses. With this bot, you can have engaging and realistic conversations with an artificial intelligence model.

## Getting Started

### Prerequisites

First, you need to install the required packages using [uv](https://docs.astral.sh/uv/):

```shell
uv sync --no-dev
```

### Configuration

You can copy and rename the provided `env.example` to `.env` and edit the file according to your data

You can create a bot on Telegram and get its API token by following the [official instructions](https://core.telegram.org/bots#how-do-i-create-a-bot).

To use the bot in a group, you have to use the @BotFather bot to [set the Group Privacy off](https://stackoverflow.com/questions/50204633/allow-bot-to-access-telegram-group-messages/50236522#50236522). This allows the bot to access all group messages.

#### Required environment variables.

You can use the `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `OPENAI_API_BASE_URL` or `OLLAMA_MODEL` for selecting the required
LLM provider.
The `OPENAI_API_BASE_URL` will look for an OpenAI API like, as the LM Studio API

- <b>Note:</b> When `GOOGLE_API_KEY` option is selected, the default model will be Gemini 2.0 Flash.

`TELEGRAM_BOT_NAME`: Your Telegram bot name

`TELEGRAM_BOT_USERNAME`: Your Telegram bot username

`TELEGRAM_BOT_TOKEN`: Your Telegram bot token

#### Selecting OpenAI Model.

`OPENAI_API_MODEL`: LLM to use for OpenAI or OpenAI-like API; if not provided, the default model will be used.

#### Selecting Google API Model.

`GOOGLE_API_MODEL`: LLM to use for Google API; if not provided, the default model will be used.

#### Enabling image Generation with Stable Diffusion

`WEBUI_SD_API_URL`: you can define a Stable Diffusion Web UI API URL for image generation. If this option is enabled the bot will answer image generation requests using Stable Diffusion generated images.

`WEBUI_SD_API_PARAMS`: A JSON string containing Stable Diffusion Web UI API params. If not provided, default parameters for the SDXL Turbo model will be used.

#### Setting custom bot character instructions

`TELEGRAM_BOT_INSTRUCTIONS_CHARACTER`: You can define a custom character for the bot instructions. 
This will override the default bot character. For example: `You are a software engineer, geek and nerd, user of linux and free software technologies.`

#### Setting extra bot instructions

`TELEGRAM_BOT_INSTRUCTIONS_EXTRA`: You can include extra LLM system instructions using this variable.

#### Setting custom bot instructions

`TELEGRAM_BOT_INSTRUCTIONS`: You can define custom LLM system instructions using this variable. 
This will override the default instructions, and the custom bot character instructions.

#### Limiting Bot interaction

`TELEGRAM_ALLOWED_CHATS`: You can use a comma-separated list of allowed chat IDs to limit bot interaction to those chats.

#### Enable multimodal capabilities

`ENABLE_MULTIMODAL`: Enable multimodal capabilities for images (True, False). The selected model must support multimodal capabilities.

#### Enable group assistant

`ENABLE_GROUP_ASSISTANT`: Enable group assistant for group chats (True, False). The bot will respond to group chats with a question mark. The default value is False.

#### Enable rate limiting

`RATE_LIMITER_REQUESTS_PER_SECOND`: The number of requests per second allowed by the bot.
`RATE_LIMITER_CHECK_EVERY_N_SECONDS`: The number of seconds between rate limit checks.
`RATE_LIMITER_MAX_BUCKET_SIZE`: The maximum bucket size for rate limiting.

#### Set preferred language

`PREFERRED_LANGUAGE`: The preferred language for the bot. (English, Spanish, etc.)

#### Set context max tokens

`CONTEXT_MAX_TOKENS`: The maximum number of tokens allowed for the bot's context.

#### Simulate typing human behavior

`SIMULATE_TYPING`: Enable simulating human typing behavior. The default is False. This typing simulation will influence the bot's response time in all chats.

`SIMULATE_TYPING_WPM`: The words per minute for simulating human typing behavior. Default is 100.

`SIMULATE_TYPING_MAX_TIME`: The maximum time in seconds for simulating human typing behavior. Default is 10 seconds (we usually don't want to wait too long).

#### Tools usage

`ENABLE_TOOLS`: Enable tool usage (True, False). Default is False. When tool usage is enabled, the bot will use the LLM's tools capabilities. When tool usage is disabled, the bot will use the prompt-based pseudo-tools implementation.

#### Logging Level

`LOGGING_LEVEL`: Sets the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL), defaulting to INFO.

#### Prompt Guardian

`ENABLE_PROMPT_GUARDIAN`: Enable prompt guardian (True, False). Default is False. When prompt guardian is enabled, the bot will use the prompt guardian to prevent basic jailbreak attacks.

Prompt Guardian uses the `meta-llama/Prompt-Guard-86M` model to detect prompt injection attacks. To install the necessary dependencies, use the following command:
```shell
uv sync --group promptguardian
```

The Prompt Guard model will be downloaded automatically from Hugging Face Hub. Ensure you have a Hugging Face API key and have accepted the model's terms and conditions [here](https://huggingface.co/meta-llama/Prompt-Guard-86M).

### Available Commands

The bot supports the following commands:

- `/flushcontext` - Clears the conversation context for the current chat. In group chats, only admins can use this command. The bot will respond with a confirmation message in the configured language.

### Running the Bot

You can run the bot using the following command:

```shell
uv run main.py
```
or

```shell
python main.py
```

## Developers information

Use `uv sync --dev` to install the development dependencies.

### Pre-commit hooks

After installing the development dependencies, to install pre-commit scripts, including ruff checks, you can run
the following command:

```shell
pre-commit install
```

### Running tests

You can run the tests using the following command:

```shell
uv run python -m unittest discover
```

## Contributing

If you'd like to contribute to this project, feel free to submit a pull request. We're always open to new ideas or improvements to the code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
