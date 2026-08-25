Using as an App
===============

`manolo-bot` is ready to use as a standalone Telegram chat bot. It handles message queuing, multimodal inputs (images/voice), and tool execution out of the box.

Quick Start: Your First Telegram Bot
------------------------------------

This guide will help you get a basic Telegram bot up and running in minutes.

Step 1: Get a Telegram Bot Token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  Open Telegram and search for `@BotFather <https://t.me/botfather>`_.
2.  Send the command ``/newbot``.
3.  Follow the instructions to choose a name and a username for your bot.
4.  BotFather will give you an **API Token**. Keep this safe!

Step 2: Get an AI API Key
~~~~~~~~~~~~~~~~~~~~~~~~~

For this tutorial, we recommend using Google Gemini as it's easy to set up:

1.  Go to `Google AI Studio <https://aistudio.google.com/>`_.
2.  Create a free **API Key**.

Step 3: Install and Run
~~~~~~~~~~~~~~~~~~~~~~~

1.  **Install the package**:

    .. code-block:: shell

       pip install manolo-bot

2.  **Create a configuration file**:
    Create a file named ``.env`` in your current folder and paste your keys:

    .. code-block:: text

       TELEGRAM_BOT_TOKEN=your_telegram_token_here
       TELEGRAM_BOT_NAME=MyAwesomeBot
       TELEGRAM_BOT_USERNAME=my_awesome_bot
       GOOGLE_API_KEY=your_google_api_key_here
       AI_MODE=agent

3.  **Run the bot**:

    .. code-block:: shell

       manolo-bot

4.  **Start Chatting**:
    Open Telegram, find your bot by its username, and send it a message like "Hello!".

Security and Privacy
--------------------

By default, your bot will respond to anyone who sends it a message. To prevent unauthorized use, you can restrict it to specific users or groups.

Restricting to Specific Chats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``TELEGRAM_ALLOWED_CHATS`` variable to provide a comma-separated list of IDs. The bot will ignore any message coming from a chat ID not in this list.

.. code-block:: text

   # Only respond to these two users/groups
   TELEGRAM_ALLOWED_CHATS=12345678, -100123456789

**How to find a Chat ID?**
You can use a bot like `@userinfobot <https://t.me/userinfobot>`_ or `@MissRose_bot <https://t.me/MissRose_bot>`_ (send ``/id`` in a group) to find the ID of a user or a group.

Private vs. Group Chats
~~~~~~~~~~~~~~~~~~~~~~~

* `ALLOW_PRIVATE_CHATS`: Set to `False` to prevent the bot from responding in direct messages, forcing it to be used only in allowed groups.
* `ENABLE_GROUP_ASSISTANT`: If `True`, the bot will proactively respond to messages containing a `?` in groups, even if not directly mentioned. This is useful for support or FAQ bots.

Advanced Configuration
----------------------

The bot is configured entirely via environment variables. You can set these in your terminal or, more conveniently, in a `.env` file in the project root.

LLM Providers
~~~~~~~~~~~~~

You must provide at least one API key or a local model configuration.

* `GOOGLE_API_KEY`: API key for Google Gemini. If set, the bot defaults to `gemini-2.0-flash`.
* `OPENAI_API_KEY`: API key for OpenAI.
* `OPENAI_API_BASE_URL`: Use this for OpenAI-compatible services (like LM Studio or LocalAI).
* `OLLAMA_MODEL`: Name of a model running on your local Ollama instance (e.g., `llama3`).

**Advanced Model Selection:**

* `OPENAI_API_MODEL`: Override the default OpenAI model (e.g., `gpt-4o`).
* `GOOGLE_API_MODEL`: Override the default Gemini model (e.g., `gemini-1.5-pro`).

Telegram Bot Settings
~~~~~~~~~~~~~~~~~~~~~

To get these values, talk to `@BotFather` on Telegram.

* `TELEGRAM_BOT_TOKEN`: **Required.** Your unique bot token.
* `TELEGRAM_BOT_NAME`: The display name you gave your bot.
* `TELEGRAM_BOT_USERNAME`: The @username of your bot.
* `BOT_UUID`: A unique identifier for this bot instance (default: `default-bot-uuid`). Used to isolate conversation history in storage.
* `USER_ID`: Your Telegram User ID (default: `0`). Used for internal tracking and metadata.

Bot Persona and Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Customize how the bot behaves and speaks.

* `TELEGRAM_BOT_INSTRUCTIONS`: Full system instructions for the LLM. Overrides all other instructions.
* `TELEGRAM_BOT_INSTRUCTIONS_CHARACTER`: Defines the bot's "personality" (e.g., "You are a helpful assistant").
* `TELEGRAM_BOT_INSTRUCTIONS_EXTRA`: Additional instructions appended to the main system prompt.
* `PREFERRED_LANGUAGE`: The language the bot should prefer (default: `Spanish`).

Agent and Tools
~~~~~~~~~~~~~~~

.. note::
   Using **Agent Mode** is highly recommended for a better experience, as it allows the bot to reason and use tools effectively.

* `AI_MODE`: Selects the bot's AI mode (default: `agent`).

  * `agent`: LangGraph-based agent with an automatic tool loop. The bot uses the LLM to reason about your request and can perform multiple iterations (like searching the internet, reading web pages, and refining its search) until it completes the task.
  * `deep_agent`: The full `Deep Agents <https://www.langchain.com/deep-agents>`_ harness on top of the agent mode. Adds to-do list planning, a virtual filesystem, and sub-agent support for complex, multi-step tasks.
  * `llm`: Simple LLM with a manual tool loop. Best for models without tool-calling support or simpler use cases.

* `AGENT_MODE`: (Deprecated) Old boolean switch for agent mode. Use `AI_MODE` instead. When `AI_MODE` is not set, `AGENT_MODE=True` selects `agent`.
* `USE_TOOLS`: Set to `True` to allow the bot to use tools. In non-agent mode, it uses tools in a more direct, single-step way.
* `AGENT_INSTRUCTIONS`: Custom rules that guide how the agent should reason and prioritize its actions when in an agent mode.
* `DEEP_AGENT_WORKSPACE_PATH`: Directory used as the virtual filesystem root for the `deep_agent` mode (only used by the `local_fs` backend). Defaults to a system temporary directory (e.g. ``/tmp/manolo_bot/workspace``). Separate from `DOCUMENT_STORAGE_PATH`, which handles temporarily uploaded documents.

  .. warning::
     The default workspace lives under the system temp directory, which on Linux is world-readable.
     Files the deep-agent writes there (chat working notes, file extracts, data echoed back from tool
     results) can be read by any local user on a shared host. On multi-tenant or shared machines set
     this to a path only the bot process can read, e.g.
     ``$HOME/.local/share/manolo_bot/workspace`` with mode ``0700``.

* `DEEP_AGENT_BACKEND`: Filesystem backend type for the `deep_agent` mode (`in_memory` or `local_fs`, default `local_fs`). The `local_fs` backend persists scratch files per chat and refuses to delete anything outside the workspace. The `in_memory` backend keeps the virtual filesystem in process memory, scoped per ``(bot_uuid, chat_id)``; state is cleared on ``clean_context()`` or process restart.
* `DEEP_AGENT_SKILLS_PATHS`: Comma-separated list of skill source paths for the `deep_agent` mode. Each entry is a bare path to a directory whose subdirectories contain `SKILL.md` files, or `<path>::LABEL=<text>` to give the source a label. Skills are loaded into the agent's system prompt via progressive disclosure (metadata at startup, full `SKILL.md` bodies on demand via `read_file`). Empty by default; no skills middleware is added when unset. See the Agent Skills spec at https://agentskills.io/specification for the `SKILL.md` frontmatter format (`name` and `description` required).
* `DEEP_AGENT_MEMORY_PATH`: Root directory for per-chat long-term memory in the `deep_agent` mode. Each chat gets its own independent, seeded `AGENTS.md` file at `DEEP_AGENT_MEMORY_PATH/bot_uuid/chat_id/AGENTS.md` — no information leaks between chats. Memory files are **fully loaded into the agent's system prompt on every turn**, so keep them concise — unlike skills (progressive disclosure), every token in a memory file costs tokens on every message. Defaults to a system temporary directory (e.g. `/tmp/manolo_bot/memory`). The bot constructs a per-chat `MemoryFilesystemDeepAgentBackend` (alongside the main backend) whose default process-local `InMemoryStore` is forwarded to `create_deep_agent(store=...)`. `/flushcontext` wipes the chat's memory along with its workspace.
* `DEEP_AGENT_MEMORY_ADD_CACHE_CONTROL`: Set to `True` to add an Anthropic prompt-cache breakpoint on the memory block (default `False`). No-op on non-Anthropic models. Fed to `LLMDeepAgent(memory_add_cache_control=...)` at agent construction.

Search Configuration
~~~~~~~~~~~~~~~~~~~~

The bot can search the web to answer your questions. By default, it uses DuckDuckGo (no API key required). For better results, you can enable **Tavily Search**:

* `USE_TAVILY_SEARCH`: Set to `True` to use Tavily instead of DuckDuckGo.
* `TAVILY_SEARCH_KEY`: Your Tavily API key (get one at `tavily.com <https://tavily.com/>`_).

Image and Voice
~~~~~~~~~~~~~~~

* `IMAGE_MULTIMODAL`: Set to `True` to allow the bot to "see" images you send or reply to.
* `AUDIO_MULTIMODAL`: **(Experimental)** Set to `True` to allow the bot to "hear" voice messages. Currently only supported by Google Gemini.
* `DOCUMENT_MULTIMODAL`: Set to `True` to allow the bot to "read" uploaded documents (PDF, DOCX, TXT).
* `WEBUI_SD_API_URL`: If you have a Stable Diffusion Web UI running, provide its URL here to enable the `/generate_image` capability.
* `WEBUI_SD_API_PARAMS`: A JSON string of parameters for the Stable Diffusion API (e.g., `{"steps": 20, "width": 512}`).
* `WEBUI_SD_API_NEGATIVE_PROMPT`: Words or concepts you want Stable Diffusion to avoid.

Document and Voice Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When `DOCUMENT_MULTIMODAL` is enabled, the bot can process uploaded files. It extracts the text content, cleans it, and stores it in a temporary storage so the LLM can reference it during the conversation.

Similarly, when `AUDIO_MULTIMODAL` is enabled, the bot can process voice messages, but only up to the configured maximum size.

* `MAX_DOCUMENT_SIZE_BYTES`: Maximum size of documents the bot will process (default: `2097152` bytes / 2MB).
* `MAX_VOICE_SIZE_BYTES`: Maximum size of voice messages the bot will process (default: `2097152` bytes / 2MB).
* `DOCUMENT_STORAGE_PATH`: Directory where extracted document text is stored. Defaults to a system temporary directory (`/tmp/manolo_bot/documents` on Linux).

Storage and Persistence
~~~~~~~~~~~~~~~~~~~~~~~

* `STORAGE_TYPE`:
    * `memory` (default): Conversation history is lost when the bot stops.
    * `redis`: Conversation history is saved in a Redis database.
* `REDIS_URL`: Connection string if using Redis (e.g., `redis://localhost:6379/0`).

Model Context Protocol (MCP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`manolo-bot` can connect to external tool servers using MCP.

* `ENABLE_MCP`: Set to `True` to enable.
* `MCP_SERVERS_CONFIG`: A JSON string defining your MCP servers.

**Example Config:**

.. code-block:: json

   {
     "weather": {
       "url": "http://localhost:8000/mcp/",
       "transport": "streamable_http"
     }
   }

Interaction and Behavior
~~~~~~~~~~~~~~~~~~~~~~~~

* `ADD_NO_ANSWER`: If `True`, the bot will reply with "NO_ANSWER" if it doesn't understand a message or isn't sure if it should respond.
* `SIMULATE_TYPING`: If `True`, the bot will simulate typing before sending a response.
* `SIMULATE_TYPING_WPM`: Typing speed in words per minute (default: `100`).
* `SIMULATE_TYPING_MAX_TIME`: Maximum time in seconds to simulate typing (default: `10`).

Advanced Settings
~~~~~~~~~~~~~~~~~

Fine-tune the bot's performance and logging.

* `LOGGING_LEVEL`: Set the verbosity of logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
* `CONTEXT_MAX_TOKENS`: Maximum number of tokens to keep in conversation memory (default: `4096`).
* `WEB_CONTENT_REQUEST_TIMEOUT_SECONDS`: Timeout for tools that fetch web content (default: `10`).
* `RATE_LIMITER_REQUESTS_PER_SECOND`: Max requests per second (default: `0.25`).
* `RATE_LIMITER_CHECK_EVERY_N_SECONDS`: Interval between rate limit checks (default: `0.1`).
* `RATE_LIMITER_MAX_BUCKET_SIZE`: Token bucket size for rate limiting (default: `10`).

Available Commands
------------------

* `/flushcontext`: Use this in a chat to clear the bot's memory for that specific conversation. In groups, this is restricted to administrators.

Running with Docker
-------------------

For production deployment, using Docker Compose (V2) is recommended:

.. code-block:: shell

   docker compose up -d

If you encounter a ``KeyError: 'ContainerConfig'`` when using the older ``docker-compose`` tool, please upgrade to Docker Compose V2.

This will start both the bot and a Redis instance for persistent storage.
