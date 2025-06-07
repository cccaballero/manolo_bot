# Docker Support for manolo_bot

This document explains how to use the Docker image for running manolo_bot.

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

