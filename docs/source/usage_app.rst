Using as an App
===============

`manolo-bot` can be run as a standalone Telegram bot.

Configuration
-------------

The bot is configured using environment variables. You can create a `.env` file based on the provided `env.example`.

Key environment variables:

* `TELEGRAM_BOT_TOKEN`: Your Telegram Bot token from BotFather.
* `GOOGLE_API_KEY`: API key for Google Gemini (optional).
* `OPENAI_API_KEY`: API key for OpenAI (optional).
* `OLLAMA_BASE_URL`: Base URL for Ollama local models (optional).
* `REDIS_URL`: Redis connection URL for persistent storage (optional).

Running the Bot
---------------

If you installed using `uv`, you can run the bot with:

.. code-block:: shell

   uv run manolo-bot

Alternatively, if you installed the package, the `manolo-bot` command should be available in your path:

.. code-block:: shell

   manolo-bot

Using with Docker
-----------------

You can also run the bot using Docker and Docker Compose:

.. code-block:: shell

   docker-compose up -d
