# Docker Support for manolo_bot

This document explains how to use the Docker image for running manolo_bot.

## Environment Variables

The Docker image accepts the following environment variables for configuration:

### Required Environment Variables

- `TELEGRAM_BOT_NAME`: The name of your Telegram bot
- `TELEGRAM_BOT_TOKEN`: The token for your Telegram bot (from BotFather)
- `TELEGRAM_BOT_USERNAME`: The username of your Telegram bot

### LLM API Keys (at least one is required)

- `GOOGLE_API_KEY`: Your Google AI API key (for Gemini)
- `OPENAI_API_KEY`: Your OpenAI API key
- `OLLAMA_MODEL`: Ollama model to use (if using Ollama)

### Optional Environment Variables

#### LLM Configuration
- `GOOGLE_API_MODEL`: Google AI model to use (default: "gemini-2.0-flash")
- `OPENAI_API_MODEL`: OpenAI model to use
- `OPENAI_API_BASE_URL`: Base URL for OpenAI-like API (for example, LM Studio API)
- `CONTEXT_MAX_TOKENS`: Maximum tokens for context (default: 4096)

#### Bot Behavior
- `PREFERRED_LANGUAGE`: Preferred language for the bot (default: "Spanish")
- `ADD_NO_ANSWER`: Whether to add "no answer" to the context (default: false)
- `ENABLE_MULTIMODAL`: Enable multimodal capabilities (default: false)
- `ENABLE_GROUP_ASSISTANT`: Enable group assistant mode (default: false)
- `USE_TOOLS`: Enable tools for the bot (default: false)
- `TELEGRAM_ALLOWED_CHATS`: Comma-separated list of allowed chat IDs
- `TELEGRAM_BOT_INSTRUCTIONS`: Custom instructions for the bot
- `TELEGRAM_BOT_INSTRUCTIONS_CHARACTER`: Character instructions for the bot
- `TELEGRAM_BOT_INSTRUCTIONS_EXTRA`: Extra instructions for the bot

#### Typing Simulation
- `SIMULATE_TYPING`: Enable typing simulation (default: false)
- `SIMULATE_TYPING_WPM`: Words per minute for typing simulation (default: 100)
- `SIMULATE_TYPING_MAX_TIME`: Maximum time for typing simulation in seconds (default: 10)

#### Rate Limiting
- `RATE_LIMITER_REQUESTS_PER_SECOND`: Requests per second for rate limiting (default: 0.25)
- `RATE_LIMITER_CHECK_EVERY_N_SECONDS`: Check interval for rate limiting (default: 0.1)
- `RATE_LIMITER_MAX_BUCKET_SIZE`: Maximum bucket size for rate limiting (default: 10)

#### Web Content
- `WEB_CONTENT_REQUEST_TIMEOUT_SECONDS`: Timeout for web content requests in seconds (default: 10)

#### Stable Diffusion (for image generation)
- `WEBUI_SD_API_URL`: Stable Diffusion Web UI API URL
- `WEBUI_SD_API_PARAMS`: Stable Diffusion Web UI API params (JSON string)
- `WEBUI_SD_API_NEGATIVE_PROMPT`: Negative prompt for Stable Diffusion

#### Logging
- `LOGGING_LEVEL`: Logging level (default: "INFO", options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

## Running with Docker

### Using Docker Run

```bash
docker run -d \
  --name manolo_bot \
  --restart unless-stopped \
  -e TELEGRAM_BOT_NAME="Your Bot Name" \
  -e TELEGRAM_BOT_TOKEN="your_bot_token" \
  -e TELEGRAM_BOT_USERNAME="your_bot_username" \
  -e GOOGLE_API_KEY="your_google_api_key" \
  -e LOGGING_LEVEL="INFO" \
  manolo_bot:latest
```

### Using Docker Compose

1. Create a `.env` file with your configuration:

```
TELEGRAM_BOT_NAME=Your Bot Name
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_BOT_USERNAME=your_bot_username
GOOGLE_API_KEY=your_google_api_key
LOGGING_LEVEL=INFO
```

2. Run with docker-compose:

```bash
docker-compose up -d
```

## Building the Docker Image

```bash
docker build -t manolo_bot:latest .
```

## Logs

Logs are output to stdout/stderr, which can be viewed with:

```bash
docker logs manolo_bot
```

For persistent logs, you can mount a volume:

```bash
docker run -d \
  --name manolo_bot \
  -v ./logs:/app/logs \
  -e TELEGRAM_BOT_NAME="Your Bot Name" \
  -e TELEGRAM_BOT_TOKEN="your_bot_token" \
  -e TELEGRAM_BOT_USERNAME="your_bot_username" \
  -e GOOGLE_API_KEY="your_google_api_key" \
  manolo_bot:latest
```

