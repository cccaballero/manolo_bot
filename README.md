# manolo-bot: AI-powered Telegram Chat Bot and Library

[![Documentation Status](https://readthedocs.org/projects/manolo-bot/badge/?version=latest)](https://manolo-bot.readthedocs.io/en/latest/?badge=latest)

`manolo-bot` is an AI-powered Telegram Chat Bot and Library built with Python, leveraging modern LLM frameworks and the
Model Context Protocol (MCP).
It is designed to be both a standalone application and a reusable library for building your own AI-powered bots.

## Documentation

Full documentation is available at [https://manolo-bot.readthedocs.io/](https://manolo-bot.readthedocs.io/)

## Installation

### From PyPI (Recommended)

You can install `manolo-bot` directly from PyPI:

```shell
pip install manolo-bot
```

### From Source

If you want to run the bot from the source code, you need to install the required packages
using [uv](https://docs.astral.sh/uv/):

```shell
uv sync --no-dev
```

## Configuration

You can copy and rename the provided `env.example` to `.env` and edit the file according to your data

You can create a bot on Telegram and get its API token by following
the [official instructions](https://core.telegram.org/bots#how-do-i-create-a-bot).

To use the bot in a group, you have to use the @BotFather bot
to [set the Group Privacy off](https://stackoverflow.com/questions/50204633/allow-bot-to-access-telegram-group-messages/50236522#50236522).
This allows the bot to access all group messages.

#### Required environment variables.

You can use the `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `OPENAI_API_BASE_URL` or `OLLAMA_MODEL` for selecting the required
LLM provider.
The `OPENAI_API_BASE_URL` will look for an OpenAI API like, as the LM Studio API

- <b>Note:</b> When `GOOGLE_API_KEY` option is selected, the default model will be Gemini 2.0 Flash.

`TELEGRAM_BOT_NAME`: Your Telegram bot name

`TELEGRAM_BOT_USERNAME`: Your Telegram bot username

`TELEGRAM_BOT_TOKEN`: Your Telegram bot token

#### Advanced Identity Settings

`BOT_UUID`: A unique identifier for this bot instance (default: `default-bot-uuid`). Used to isolate conversation
history in storage.

`USER_ID`: Your Telegram User ID (default: `0`). Used for internal tracking and metadata.

#### Selecting OpenAI Model.

`OPENAI_API_MODEL`: LLM to use for OpenAI or OpenAI-like API; if not provided, the default model will be used.

#### Selecting Google API Model.

`GOOGLE_API_MODEL`: LLM to use for Google API; if not provided, the default model will be used.

#### Enabling Agent Mode

`AGENT_MODE`: Enable agent mode (True, False). Default is False. When agent mode is enabled, the bot will use **agentic
capabilities**. This means the bot will use the LLM as a reasoning engine, allowing it to iterate through multiple
steps (like searching the internet and analyzing results) to complete complex tasks.

`AGENT_INSTRUCTIONS`: (Optional) Custom instructions to guide the agent's behavior and reasoning when in agent mode.

#### Enabling image Generation with Stable Diffusion

`WEBUI_SD_API_URL`: you can define a Stable Diffusion Web UI API URL for image generation. If this option is enabled the
bot will answer image generation requests using Stable Diffusion generated images.

`WEBUI_SD_API_PARAMS`: A JSON string containing Stable Diffusion Web UI API params. If not provided, default parameters
for the SDXL Turbo model will be used.

#### Setting custom bot character instructions

`TELEGRAM_BOT_INSTRUCTIONS_CHARACTER`: You can define a custom character for the bot instructions.
This will override the default bot character. For example:
`You are a software engineer, geek and nerd, user of linux and free software technologies.`

#### Setting extra bot instructions

`TELEGRAM_BOT_INSTRUCTIONS_EXTRA`: You can include extra LLM system instructions using this variable.

#### Setting custom bot instructions

`TELEGRAM_BOT_INSTRUCTIONS`: You can define custom LLM system instructions using this variable.
This will override the default instructions, and the custom bot character instructions.

#### Limiting Bot interaction

`TELEGRAM_ALLOWED_CHATS`: You can use a comma-separated list of allowed chat IDs to limit bot interaction to those
chats.

`ALLOW_PRIVATE_CHATS`: Enable or disable direct bot interaction in private chats (True, False). Default is True.

`ADD_NO_ANSWER`: If `True`, the bot will reply with "NO_ANSWER" if it doesn't understand a message or isn't sure if it
should respond (True, False). Default is False.

#### Enable multimodal capabilities

`IMAGE_MULTIMODAL`: Enable multimodal capabilities for images (True, False). The selected model must support multimodal
capabilities.

`AUDIO_MULTIMODAL`: **(Experimental)** Enable multimodal capabilities for audio/voice messages (True, False). Currently,
this feature only works with the Gemini API.

`DOCUMENT_MULTIMODAL`: Enable document analysis for PDF, DOCX, and TXT files (True, False). Default is False.
When enabled, the bot can extract text from uploaded documents and use it as context.
For large documents, it is recommended to use a model with a large context window (like Gemini) and increase
`CONTEXT_MAX_TOKENS`.
The document text is stored separately from chat history to keep memory clean.

#### Enable group assistant

`ENABLE_GROUP_ASSISTANT`: Enable group assistant for group chats (True, False). The bot will respond to group chats with
a question mark. The default value is False.

#### File size limits and document storage

`MAX_DOCUMENT_SIZE_BYTES`: Maximum size of documents the bot will process (default: 2MB).
`MAX_VOICE_SIZE_BYTES`: Maximum size of voice messages the bot will process (default: 2MB).
`DOCUMENT_STORAGE_PATH`: Directory where extracted document text is stored. Defaults to a system temporary directory.

#### Enable rate limiting

`RATE_LIMITER_REQUESTS_PER_SECOND`: The number of requests per second allowed by the bot.
`RATE_LIMITER_CHECK_EVERY_N_SECONDS`: The number of seconds between rate limit checks.
`RATE_LIMITER_MAX_BUCKET_SIZE`: The maximum bucket size for rate limiting.

#### Set preferred language

`PREFERRED_LANGUAGE`: The preferred language for the bot. (English, Spanish, etc.)

#### Set context max tokens

`CONTEXT_MAX_TOKENS`: The maximum number of tokens allowed for the bot's context.

#### Web Content Retrieval Configuration

`WEB_CONTENT_REQUEST_TIMEOUT_SECONDS`: Timeout in seconds for HTTP requests when retrieving web content. Default is 10
seconds.

#### Simulate typing human behavior

`SIMULATE_TYPING`: Enable simulating human typing behavior. The default is False. This typing simulation will influence
the bot's response time in all chats.

`SIMULATE_TYPING_WPM`: The words per minute for simulating human typing behavior. Default is 100.

`SIMULATE_TYPING_MAX_TIME`: The maximum time in seconds for simulating human typing behavior. Default is 10 seconds (we
usually don't want to wait too long).

#### Tools usage

`USE_TOOLS`: Enable tool usage (True, False). Default is False. When tool usage is enabled, the bot will use the LLM's
tools capabilities. When tool usage is disabled, the bot will use the prompt-based pseudo-tools implementation.

#### Search Configuration

The bot uses DuckDuckGo by default for web searches. You can optionally enable **Tavily Search** for more advanced
search capabilities.

`USE_TAVILY_SEARCH`: Enable Tavily Search (True, False). Default is False.

`TAVILY_SEARCH_KEY`: Your Tavily API key. Required if `USE_TAVILY_SEARCH` is set to True.

#### Storage Configuration

`STORAGE_TYPE`: Sets the storage type for conversation context (memory, redis). Default is memory.

`REDIS_URL`: The Redis URL for storage when `STORAGE_TYPE` is set to redis. Default is `redis://localhost:6379/0`.

### MCP (Model Context Protocol) Support

manolo_bot supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting to external tool
servers.

#### Enabling MCP

Set the following environment variables:

`ENABLE_MCP`: Enable MCP support (True, False). Default is False.

`MCP_SERVERS_CONFIG`: MCP server configuration in JSON format.

#### MCP Server Configuration

MCP servers are configured via the `MCP_SERVERS_CONFIG` environment variable, which accepts a JSON object mapping server
names to their configurations.

**stdio transport example:**

```json
{
  "math": {
    "command": "python",
    "args": [
      "/path/to/math_server.py"
    ],
    "transport": "stdio"
  }
}
```

**streamable_http transport example:**

```json
{
  "weather": {
    "url": "http://localhost:8000/mcp/",
    "transport": "streamable_http"
  }
}
```

**Multiple servers:**

```json
{
  "math": {
    "command": "python",
    "args": [
      "/path/to/math_server.py"
    ],
    "transport": "stdio"
  },
  "weather": {
    "url": "http://localhost:8000/mcp/",
    "transport": "streamable_http"
  }
}
```

**Notes:**

- MCP tools are loaded alongside custom tools defined in `ai/tools.py`
- If tool name conflicts occur, MCP tools will override custom tools (a warning is logged)
- The bot will start successfully even if MCP initialization fails (graceful degradation)
- MCP is only loaded when both `ENABLE_MCP=True` and valid `MCP_SERVERS_CONFIG` are provided

#### Logging Level

`LOGGING_LEVEL`: Sets the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL), defaulting to INFO.

### Available Commands

The bot supports the following commands:

- `/flushcontext` - Clears the conversation context for the current chat. In group chats, only admins can use this
  command. The bot will respond with a confirmation message in the configured language.

### Running the Bot

You can run the bot using the following command:

```shell
uv run manolo-bot
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
uv run python -m unittest discover tests
```

## Library Usage

`manolo-bot` can also be used as a library to build your own AI assistants. It provides a clean abstraction for LLM
providers, message storage, and agentic capabilities.

### Using LLMAgent (Recommended)

The `LLMAgent` is the most powerful way to use the library. It allows the bot to use tools and iterate through multiple
steps to solve complex queries.

```python
import asyncio
from manolo_bot.ai.llmagent import LLMAgent
from manolo_bot.ai.llmbot import LLMBuilder
from manolo_bot.ai.config import BotConfig, LLMConfig
from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage
from manolo_bot.ai.tools import get_tools


async def main():
    # 1. Configure LLM (Google, OpenAI, or Ollama)
    llm_config = LLMConfig(google_api_key="your_api_key")
    llm = LLMBuilder(llm_config).get_llm()

    # 2. Define Bot identity
    bot_config = BotConfig(bot_uuid="my-bot", bot_name="Assistant")

    # 3. Setup Storage for a specific conversation
    storage = MemoryMessagesStorage(bot_uuid="my-bot", chat_id=123)
    await storage.refresh_messages()

    # 4. Initialize Agent with tools
    tools = get_tools()
    agent = LLMAgent(
        llm=llm,
        config=bot_config,
        system_instructions="You are a helpful assistant.",
        storage=storage,
        tools=tools
    )

    # 5. Interact
    # The agent can search the web, analyze content, etc.
    response = await agent.answer_message(chat_id=123, message="What is the current price of Bitcoin?")
    print(f"Agent: {response.content}")

    # 6. Persistent changes (if using Redis)
    await storage.commit()


if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Tools

You can provide your own tools when initializing `LLMAgent` or `LLMBot`:

```python
from langchain_core.tools import tool
from manolo_bot.ai.llmagent import LLMAgent


@tool
def my_tool(query: str) -> str:
    """Description of my tool."""
    return "Result"


agent = LLMAgent(..., tools=[my_tool])
```

### Using LLMBot (Simple)

For simpler use cases or when using models that don't support tool calling, you can use the basic `LLMBot`. It provides
a direct chat interface without iterative reasoning.

```python
from manolo_bot.ai.llmbot import LLMBot

# ... (same setup as above)

bot = LLMBot(
    llm=llm,
    config=bot_config,
    system_instructions="You are a simple assistant.",
    storage=storage
)

response = await bot.answer_message(chat_id=123, message="Hello!")
```

For more advanced usage and full API details, please refer to
the [Full Documentation](https://manolo-bot.readthedocs.io/).

## Contributing

If you'd like to contribute to this project, feel free to submit a pull request. We're always open to new ideas or
improvements to the code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
