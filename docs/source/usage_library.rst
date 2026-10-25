Using as a Library
==================

`manolo-bot` is designed as a modular library, allowing you to integrate its AI capabilities into any Python application—not just Telegram bots. This section explains the core concepts and how to use the library components.

Core Concepts
-------------

To build an AI assistant with `manolo-bot`, you need to understand four main components:

1. **LLM Configuration**: Defining which AI model to use and providing the necessary credentials.
2. **Bot Configuration**: Defining the bot's identity and behavioral settings.
3. **Storage**: Managing the conversation history (context) so the bot can "remember" previous messages.
4. **The Bot/Agent**: The main engine that processes messages using the LLM and storage.

LLM Configuration and Builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `LLMConfig` class holds all settings for the AI provider (OpenAI, Google, or Ollama). The `LLMBuilder` then takes this configuration and returns a compatible LangChain model object.

.. code-block:: python

   from manolo_bot.ai.config import LLMConfig
   from manolo_bot.ai.llmbot import LLMBuilder

   # Example: Google Gemini
   llm_config = LLMConfig(
       google_api_key="your_api_key",
       google_api_model="",
       openai_api_key="",
       openai_api_model="",
       openai_api_base_url="",
       ollama_model="",
   )
   llm = LLMBuilder(llm_config).get_llm()

Bot Configuration
~~~~~~~~~~~~~~~~~

The `BotConfig` class defines settings like the bot's name, UUID (for unique identification in storage), and features like multimodal support.

.. code-block:: python

   from manolo_bot.ai.config import BotConfig

   bot_config = BotConfig(
       bot_uuid="my-unique-bot-id",
       bot_name="MyAssistant",
       bot_username="myassistant_bot",
       bot_token="123456:ABC",
       user_id=0,
   )

Storage
~~~~~~~

The `storage` component is responsible for persisting conversation history. `manolo-bot` uses a "per-chat" storage model. Each user or conversation should have its own storage instance identified by a `chat_id`.

* **MemoryMessagesStorage**: Stores messages in RAM. Useful for testing or simple CLI tools. History is lost when the process restarts.
* **RedisMessagesStorage**: Stores messages in a Redis database. Recommended for production and persistent history.

.. code-block:: python

   from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage

   # chat_id is a unique identifier for the current conversation (e.g., a user ID)
   storage = MemoryMessagesStorage(bot_uuid="my-unique-bot-id", chat_id=12345)

History policy (what gets remembered)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Storage decides *where* history lives; the history policy decides *what* from each
agent turn gets stored. By default only the final answer is persisted (``FinalOnlyPolicy``).
Pass ``history_policy=FullTracePolicy()`` to also keep intermediate tool calls and results —
useful for debugging or multi-step continuity, at the cost of higher token usage. Truncation
always drops ``AI → Tool`` blocks atomically, so replayed history stays valid for the next turn.

.. code-block:: python

   from manolo_bot.ai.history_policy import FullTracePolicy
   from manolo_bot.ai.llmagent import LLMAgent

   agent = LLMAgent(
       ...,
       history_policy=FullTracePolicy(),
   )

To define your own strategy, subclass ``BaseHistoryPolicy`` and implement ``select(tail)``,
which receives only the new messages produced by the turn and returns the subset to persist.
``LLMDeepAgent`` accepts the same parameter. No bot configuration is needed — the policy is
a plain object injected by the caller.

Main Component: LLMAgent (Recommended)
--------------------------------------

The `LLMAgent` is the most powerful and feature-rich component in `manolo-bot`. It implements **Agentic behavior**, meaning it uses the LLM as a "reasoning engine" to complete tasks.

What is "Agentic" behavior?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike a standard LLM that generates a single response, an **Agent** iterates through a loop:

1.  **Reasoning**: The bot analyzes the user's request.
2.  **Tool Selection**: It decides if it needs external information (e.g., using a Search tool).
3.  **Iteration**: It executes the tool, observes the result, and then *iterates*.
4.  **Completion**: It continues this loop until it has enough information to provide a final answer.

Implementation Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from manolo_bot.ai.llmagent import LLMAgent
   from manolo_bot.ai.llmbot import LLMBuilder
   from manolo_bot.ai.config import LLMConfig, BotConfig
   from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage
   from manolo_bot.ai.tools import get_tools

   async def main():
       # 1. Setup LLM (Must support Tool Calling)
       llm_config = LLMConfig(
           google_api_key="your_key",
           google_api_model="",
           openai_api_key="",
           openai_api_model="",
           openai_api_base_url="",
           ollama_model="",
       )
       llm = LLMBuilder(llm_config).get_llm()

       # 2. Setup Bot Identity
       bot_config = BotConfig(
           bot_uuid="bot-1",
           bot_name="Assistant",
           bot_username="assistant_bot",
           bot_token="123456:ABC",
           user_id=0,
       )

       # 3. Setup Storage
       chat_id = 1001
       storage = MemoryMessagesStorage(bot_uuid="bot-1", chat_id=chat_id)
       await storage.refresh_messages()

       # 4. Initialize the Agent with Tools
       tools = get_tools()
        agent = LLMAgent(
            llm=llm,
            bot_config=bot_config,
            system_instructions="You are a helpful assistant.",
            messages_storage=storage,
            tools=tools
        )

       # 5. Use the agent as an async context manager (required for MCP)
       async with agent:
           # 6. Answer a message
           # The agent will automatically decide when to use search, etc.
           response = await agent.answer_message(chat_id=chat_id, message="What happened in the news today?")
           print(f"Agent Result: {response.content}")

       await storage.commit()

if __name__ == "__main__":
        asyncio.run(main())

Advanced Component: LLMDeepAgent (Deep Agents Harness)
-------------------------------------------------------

The `LLMDeepAgent` extends `LLMAgent` with the full `Deep Agents <https://www.langchain.com/deep-agents>`_ harness.
It adds three capabilities on top of the agent loop:

* **To-do list planning**: A no-op todo list tool that helps the agent keep track of multi-step tasks.
* **Virtual filesystem**: Read, write, edit, and search files on a pluggable backend (in-memory or local filesystem).
* **Sub-agents**: The ability to spawn isolated sub-agents for focused subtasks.

This makes it ideal for complex, long-horizon tasks that benefit from planning and scratch space.

The virtual filesystem uses a pluggable backend, similar to how message and document storage work:

* ``MemoryDeepAgentBackend``: In-memory virtual filesystem. State is scoped per ``(bot_uuid, chat_id)`` and shared across instances in the same process; cleared on ``clean_context()`` or process restart.
* ``FilesystemDeepAgentBackend``: Persistent virtual filesystem stored under ``workspace_path/bot_uuid/chat_id``. ``clear()`` resolves the chat path and refuses to delete anything outside the configured workspace.

If no backend is provided, ``LLMDeepAgent`` falls back to an in-memory ``StateBackend``.

Skills (progressive disclosure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The deep agent loads Agent Skills (capability bundles described by ``SKILL.md`` files) via ``deepagents``' ``SkillsMiddleware``. Two optional parameters on ``LLMDeepAgent.__init__`` mirror the backend-injection pattern used by ``backend=``:

* ``skills_paths``: A sequence of source paths. Each entry is a bare path (``str``) or a ``(path, label)`` tuple. Bare paths get a default label derived from the final path component. The optional ``<path>::LABEL=<text>`` syntax is also accepted and split into a tuple under the hood, so the same string format works for env-var-driven configuration.
* ``skills_backend``: An instance of :class:`BaseSkillsBackend` (typically :class:`SkillsFilesystemDeepAgentBackend`) — explicitly injected by the caller. ``LLMDeepAgent`` does **not** instantiate a skills backend itself: if you don't pass one, ``SkillsMiddleware`` is omitted entirely, even when ``skills_paths`` is set. A ``WARNING`` is logged so the misconfiguration is loud, not silent.

When ``skills_paths`` is not set, no ``SkillsMiddleware`` is added and behavior is identical to a deep-agent instance without skills. Configuration (e.g. the ``DEEP_AGENT_SKILLS_PATHS`` env var) flows in through the caller, which reads it and passes it explicitly — ``LLMDeepAgent`` does not read configuration itself.

Skills are operator-provided, not per-chat state. They are intentionally **not** cleared by ``clean_context()`` — wiping operator content when a user runs ``/flushcontext`` would be destructive. ``SkillsMiddleware`` owns its own per-session lifecycle via the ``before_agent`` hook, so a fresh chat context simply re-reads skill metadata on the next session.

The skills backend
++++++++++++++++++

Skills are *global* — shared across all chats and bot instances in the same process. The canonical wrapper is :class:`SkillsFilesystemDeepAgentBackend` (a :class:`BaseSkillsBackend` subclass), which wraps ``deepagents.backends.filesystem.FilesystemBackend`` with ``virtual_mode=False`` so absolute skill source paths (e.g. ``/etc/manolo_bot/skills``) resolve directly on the host filesystem. The unrestricted read access is safe because the wrapper is only handed to ``SkillsMiddleware``, which only calls ``ls()`` and ``download_files()`` (read-only) at the configured source paths.

For a standalone bot, construct one instance at module load and pass it to every agent:

.. code-block:: python

   from manolo_bot.storage.deep_agent_backends.skills_filesystem_backend import (
       SkillsFilesystemDeepAgentBackend,
   )

   skills_backend = SkillsFilesystemDeepAgentBackend()
   llm_bot = LLMDeepAgent(
       ...,
       skills_paths=["/etc/manolo_bot/skills", ("$HOME/.local/share/manolo_bot/skills", "User")],
       skills_backend=skills_backend,
   )

To use a custom workspace path (e.g. relative paths anchored at a project root):

.. code-block:: python

   skills_backend = SkillsFilesystemDeepAgentBackend(workspace_path="/opt/mybot/skills")
   llm_bot = LLMDeepAgent(
       ...,
       skills_paths=["/opt/mybot/skills", ("/etc/manolo_bot/skills", "System")],
       skills_backend=skills_backend,
   )

For a custom backend (e.g. Redis-backed, S3, in-memory), subclass :class:`BaseSkillsBackend` and implement ``.backend`` (a ``BackendProtocol``), ``.routes(sources)`` (CompositeBackend route entries so the agent's runtime tools can reach skill content — see `Skill/memory-file routing for the agent's runtime tools`_ below) and an async ``.clear()``. ``clear()`` should typically be a no-op — skills are operator-provided and must never be wiped by ``clean_context()``:

.. code-block:: python

   from manolo_bot.storage.deep_agent_backends.base import BaseSkillsBackend

   class RedisSkillsBackend(BaseSkillsBackend):
       def __init__(self, client):
           self._client = client
           self._backend = ...  # your BackendProtocol implementation

       @property
       def backend(self):
           return self._backend

       def routes(self, sources):
           # Map each source prefix to a backend rooted at the routed location
           # (CompositeBackend strips the prefix before delegating). Return {}
           # if skill content is served exclusively through SkillsMiddleware.
           return {f"{s.rstrip('/')}/": self._backend for s in sources}

       async def clear(self):
           pass  # operator-provided content; never wipe

Skill/memory-file routing for the agent's runtime tools
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

When ``skills_paths`` or ``memory_paths`` is configured, ``LLMDeepAgent`` wraps the agent's main filesystem backend with a :class:`deepagents.backends.composite.CompositeBackend` whose routes come from the injected wrappers' ``routes()`` factories — the path→backend mapping is storage-specific knowledge owned by each wrapper, not by the agent. The filesystem wrappers (:class:`SkillsFilesystemDeepAgentBackend`, :class:`MemoryFilesystemDeepAgentBackend`) return one route per skill source directory / memory file's parent directory: a sandboxed :class:`FilesystemBackend` rooted at that path with ``virtual_mode=True``. This lets the agent's runtime ``read_file`` / ``write_file`` / ``edit_file`` / ``ls`` / ``glob`` / ``grep`` tools reach those files at their absolute paths — without it, the chat-scoped backend's ``virtual_mode=True`` root would reject any path outside the chat workspace and ``read_file`` calls to skill or memory paths would silently fail with "file not found" (writes would silently land in in-memory state instead of disk). The per-chat scratch behaviour for non-skill/non-memory paths is unchanged.

For skills this routing is read-only access to capability bundles. For memory it also enables **durable write-back**: the agent persists learnings via ``edit_file`` / ``write_file``, and because the memory file's parent directory is routed to a real ``FilesystemBackend``, those updates land on the actual ``AGENTS.md`` file on disk — the same file ``MemoryMiddleware`` loads at the start of every turn. Repeated parent directories (multiple memory files in one directory, or a skill source and a memory file sharing a directory) are deduped into a single route.

Custom backends (Redis, S3, DB-backed, virtual filesystems) implement ``routes(sources)`` to expose their content to the runtime tools: keys are virtual path prefixes ending in ``"/"`` and values are backends rooted at the routed location (``CompositeBackend`` strips the prefix before delegating, so each value must resolve the stripped path inside its own root — e.g. a ``StoreBackend``-style backend). A backend may return ``{}`` if its content is served exclusively through the middleware (``SkillsMiddleware`` / ``MemoryMiddleware``) and the runtime tools should not reach it.

.. code-block:: python

   import asyncio
   from manolo_bot.ai.llmdeepagent import LLMDeepAgent
   from manolo_bot.ai.llmbot import LLMBuilder
   from manolo_bot.ai.config import LLMConfig, BotConfig
   from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage
   from manolo_bot.storage.deep_agent_backends.memory_backend import MemoryDeepAgentBackend

   async def main():
       # 1. Setup LLM (Must support Tool Calling)
       llm_config = LLMConfig(
           google_api_key="your_key",
           google_api_model="",
           openai_api_key="",
           openai_api_model="",
           openai_api_base_url="",
           ollama_model="",
       )
       llm = LLMBuilder(llm_config).get_llm()

       # 2. Setup Bot Identity
       bot_config = BotConfig(
           bot_uuid="bot-1",
           bot_name="Assistant",
           bot_username="assistant_bot",
           bot_token="123456:ABC",
           user_id=0,
       )

       # 3. Setup Storage
       chat_id = 1001
       storage = MemoryMessagesStorage(bot_uuid="bot-1", chat_id=chat_id)
       await storage.refresh_messages()

       # 4. Setup the deep agent filesystem backend
       backend = MemoryDeepAgentBackend(bot_uuid="bot-1", chat_id=chat_id)
       # For a persistent workspace instead:
       # backend = FilesystemDeepAgentBackend("bot-1", chat_id, "/path/to/workspace")

       # 5. Initialize the Deep Agent
       agent = LLMDeepAgent(
           llm=llm,
           bot_config=bot_config,
           system_instructions="You are a helpful assistant.",
           messages_storage=storage,
           backend=backend,
       )
       await agent.initialize_async_resources()

       # 6. Answer a message
       response = await agent.answer_message(
           chat_id=chat_id,
           message="Research and summarise the latest developments in AI agents."
       )
       print(f"Agent: {response.content}")

       await storage.commit()

   if __name__ == "__main__":
       asyncio.run(main())

To enable skills in your own bot, construct a skills backend and pass it explicitly:

.. code-block:: python

   from manolo_bot.ai.llmdeepagent import LLMDeepAgent
   from manolo_bot.storage.deep_agent_backends.skills_filesystem_backend import (
       SkillsFilesystemDeepAgentBackend,
   )

   skills_backend = SkillsFilesystemDeepAgentBackend()
   llm_bot = LLMDeepAgent(
       llm,
       bot_config,
       system_instructions,
       messages_storage,
       backend=backend,
       skills_paths=[
           "/etc/manolo_bot/skills",
           ("/opt/skills/research", "Research"),
       ],
       skills_backend=skills_backend,
   )

Memory (long-term memory)
~~~~~~~~~~~~~~~~~~~~~~~~~

The deep agent can load long-term memory from a per-chat ``AGENTS.md`` file via ``deepagents``' ``MemoryMiddleware``. Two optional parameters on ``LLMDeepAgent.__init__`` control it, mirroring the backend-injection pattern used by ``backend=`` and ``skills_backend=``:

* ``memory_backend``: An instance of :class:`BaseMemoryBackend` (typically a per-chat :class:`MemoryFilesystemDeepAgentBackend`) — explicitly injected by the caller. ``LLMDeepAgent`` does **not** instantiate a memory backend itself; the injected wrapper supplies the backend used to read the memory file, the chat-scoped source path (derived automatically when ``memory_paths`` is not given), and the langgraph store forwarded to ``create_deep_agent(store=...)``. The ``MemoryMiddleware`` itself is constructed by the agent, exactly like ``SkillsMiddleware``. If you don't pass one, ``MemoryMiddleware`` is omitted entirely. A ``WARNING`` is logged so the misconfiguration is loud, not silent.
* ``memory_paths``: Optional explicit sequence of memory file paths (advanced use) that overrides the wrapper-derived source. When ``None`` and a ``memory_backend`` is present, the agent uses the wrapper's chat-scoped ``AGENTS.md``.
* ``memory_add_cache_control``: ``bool``, default ``False``. Passed through to the ``MemoryMiddleware`` the agent constructs; adds an Anthropic prompt-cache breakpoint on the memory block. No-op on non-Anthropic models.

.. warning::
   Unlike skills (progressive disclosure — metadata at startup, full bodies on demand), memory files are **fully loaded into the system prompt on every turn**. Keep them concise: every token in a memory file is paid on every message, so prefer skills for large capability bundles and reserve memory for small, durable facts and preferences.

The memory backend
++++++++++++++++++

Memory is **per-chat** — each chat gets its own independent, seeded ``AGENTS.md`` file under ``DEEP_AGENT_MEMORY_PATH``, so no information leaks between chats. The canonical wrapper is :class:`MemoryFilesystemDeepAgentBackend` (a :class:`BaseMemoryBackend` subclass), scoped by ``(bot_uuid, chat_id)`` exactly like :class:`FilesystemDeepAgentBackend` and sharing its two-layer security model (``bot_uuid`` validated against ``[A-Za-z0-9_-]+``, plus a realpath containment check). On construction it creates the chat memory directory (``memory_root/bot_uuid/chat_id``) and seeds a minimal **non-empty** ``AGENTS.md`` template — deepagents' ``MemoryMiddleware`` skips empty sources (HTML comments are stripped) and would otherwise render "(No memory loaded)" while hiding the source path from the agent. It also owns the langgraph store: by default a process-local ``InMemoryStore()`` created at construction time, or any ``BaseStore`` you inject via ``store=``. Like the skills wrapper, the memory wrapper separates two backend roles: ``backend`` is a ``virtual_mode=False`` ``FilesystemBackend`` handed to ``MemoryMiddleware`` (whose raw absolute source paths must pass through unchanged), while ``routes()`` provides a separate ``virtual_mode=True`` instance for the agent's ``CompositeBackend`` (whose route targets receive leading-slash-stripped keys).

For a standalone bot, construct one instance per chat (typically inside ``instance_llm_bot``, alongside the main backend) and pass it to that chat's agent:

.. code-block:: python

   from manolo_bot.ai.llmdeepagent import LLMDeepAgent
   from manolo_bot.storage.deep_agent_backends.memory_filesystem_backend import (
       MemoryFilesystemDeepAgentBackend,
   )

   # One instance per chat — pass it to that chat's agent.
   memory_backend = MemoryFilesystemDeepAgentBackend(
       bot_uuid="bot-1", chat_id=1001, memory_root="/var/lib/manolo_bot/memory"
   )
   llm_bot = LLMDeepAgent(
       llm,
       bot_config,
       system_instructions,
       messages_storage,
       backend=backend,
       memory_backend=memory_backend,
       memory_add_cache_control=True,
   )

For a custom backend (e.g. Redis-backed, S3, in-memory), subclass :class:`BaseMemoryBackend` with a per-chat constructor contract ``(bot_uuid, chat_id, ...)`` and implement ``.backend`` (a ``BackendProtocol``), ``.routes(sources)`` (CompositeBackend route entries so the agent's runtime tools can reach the memory file — see `Skill/memory-file routing for the agent's runtime tools`_ below) and an async ``.clear()``. Add a ``.store`` property (a langgraph ``BaseStore`` or ``None``) when your backend needs persistent agent state. ``clear()`` semantics are implementation-defined: the filesystem implementation deletes the chat's memory directory (``/flushcontext`` wipes chat memory along with the workspace):

.. code-block:: python

   from manolo_bot.storage.deep_agent_backends.base import BaseMemoryBackend

   class RedisMemoryBackend(BaseMemoryBackend):
       def __init__(self, bot_uuid, chat_id, client):
           self.bot_uuid = bot_uuid
           self.chat_id = chat_id
           self._client = client
           self._backend = ...  # your BackendProtocol implementation
           self._store = ...    # your BaseStore implementation, or None

       @property
       def backend(self):
           return self._backend

       @property
       def store(self):
           return self._store

       def routes(self, sources):
           # Map the chat's memory path prefix to a backend rooted at the
           # routed location (CompositeBackend strips the prefix before
           # delegating). Unlike skills, memory routes must support WRITE
           # (durable edit_file write-back), not just read. Return {} if memory
           # is served exclusively through MemoryMiddleware.
           return {f"/memory/{self.bot_uuid}/{self.chat_id}/": self._backend}

       async def clear(self):
           # Per-chat state: drop this chat's memory keys.
           ...

Memory is per-chat state and is cleared by ``clean_context()`` — ``/flushcontext`` wipes the chat's memory along with its workspace.

Memory files are also routed into the agent's main backend ``CompositeBackend`` (see `Skill/memory-file routing for the agent's runtime tools`_ above): the memory backend wrapper's ``routes()`` factory maps the chat memory directory to a real ``FilesystemBackend`` route, so the agent's runtime ``read_file`` / ``write_file`` / ``edit_file`` tools reach the actual file on disk. This is what makes durable write-back work — learnings the agent saves via ``edit_file`` persist to the same seeded ``AGENTS.md`` file ``MemoryMiddleware`` loads every turn. Token-cost behavior is unchanged from the notes above.

Simple Alternative: LLMBot
--------------------------

The `LLMBot` is a simpler implementation designed for:
*   **Simple Models**: LLMs that do not support tool calling or have limited reasoning capabilities.
*   **Direct Interaction**: When you only need a straightforward chat interface without multi-step reasoning.
*   **Performance**: It is slightly faster as it doesn't perform multiple iterations.

.. code-block:: python

   from manolo_bot.ai.llmbot import LLMBot

   # ... (Setup is identical to LLMAgent, but without tools)

   bot = LLMBot(
       llm=llm,
       bot_config=bot_config,
       system_instructions="You are a simple chatbot.",
       messages_storage=storage
   )

   async with bot:
       response = await bot.answer_message(chat_id=chat_id, message="Hello!")

Multimodal Support
------------------

`manolo-bot` supports processing images, voice messages, and documents.

Images
~~~~~~

To process an image, use the ``answer_image_message`` method. It requires a publicly accessible URL to the image.

.. code-block:: python

   response = await bot.answer_image_message(
       chat_id=chat_id,
       text="What is in this image?",
       image="https://example.com/image.jpg"
   )

Attachment download headers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some platforms require HTTP header authentication to fetch attachment file URLs. Set ``attachment_headers`` on ``BotConfig`` — the headers are sent on every attachment download (``LLMBot._download_file`` and ``LLMAgent.answer_image_message``'s inline download; ``LLMDeepAgent`` inherits the behavior). The default ``{}`` preserves the current behavior and Telegram file URLs are unaffected.

.. code-block:: python

   from manolo_bot.ai.config import BotConfig

   bot_config = BotConfig(
       ...,
       attachment_headers={"Authorization": "Bearer <token>"},
   )

For custom schemes, override the ``LLMBot._download_headers()`` hook, which returns a copy of that mapping:

.. code-block:: python

   from manolo_bot.ai.llmbot import LLMBot


   class CustomAuthBot(LLMBot):
       def _download_headers(self) -> dict:
           return {"Authorization": "Custom scheme"}

Voice
~~~~~

Voice message support depends on the LLM backend (e.g., Google Gemini).

.. code-block:: python

   response = await bot.answer_voice_message(
       chat_id=chat_id,
       text="Summarize this voice message",
       audio="https://example.com/voice.ogg"
   )

Documents
~~~~~~~~~

`manolo-bot` can extract text from PDF, DOCX, and TXT files. It stores the extracted text in a specialized `document_storage`.

.. code-block:: python

   response = await bot.answer_document_message(
       chat_id=chat_id,
       text="What is the summary of this report?",
       document_url="https://example.com/report.pdf",
       filename="report.pdf"
   )

Model Context Protocol (MCP)
----------------------------

The Model Context Protocol (MCP) allows your bot to connect to external tool servers. To use MCP, you must enable it in the `BotConfig` and provide a configuration dictionary.

.. code-block:: python

   from manolo_bot.ai.config import BotConfig

   mcp_config = {
       "mcpServers": {
           "everything": {
               "command": "npx",
               "args": ["-y", "@modelcontextprotocol/server-everything"],
               "transport": "stdio"
           }
       }
   }

   bot_config = BotConfig(
       ...,
       enable_mcp=True,
       mcp_servers_config=mcp_config
   )

When `enable_mcp` is True, you **must** use the bot/agent as an async context manager to ensure connections are properly established and closed:

.. code-block:: python

   async with agent:
       # MCP tools are automatically loaded and available to the agent
       response = await agent.answer_message(chat_id, "Use an MCP tool to...")

Custom Tools
------------

You can easily provide your own tools to both ``LLMAgent`` and ``LLMBot``. This allows you to extend the bot's capabilities with your own domain-specific logic.

To add custom tools, use the ``@tool`` decorator from ``langchain_core.tools`` and pass a list of tools to the constructor.

.. code-block:: python

   from langchain_core.tools import tool
   from manolo_bot.ai.llmagent import LLMAgent

   @tool
   def get_stock_price(symbol: str) -> str:
       """Gets the current stock price for a given symbol."""
       # Your custom logic here
       return f"The price of {symbol} is $150.00"

   # Initialize with custom tools
   custom_tools = [get_stock_price]
   agent = LLMAgent(
       ...,
       tools=custom_tools
   )

.. note::
   If you provide a ``tools`` list, it will **replace** the default built-in tools. If you want to **extend** the default tools, you can use the ``get_tools`` function:

   .. code-block:: python

      from manolo_bot.ai.tools import get_tools
      
      all_tools = get_tools(bot_config) + [get_stock_price]
      agent = LLMAgent(..., tools=all_tools)

Retrieval-Augmented Generation (RAG)
------------------------------------

A local document knowledge base can be attached to ``LLMAgent`` (or ``LLMDeepAgent``).
Build a backend via the factory, index ``RAGSource`` records — descriptions may contain
commas, since records never go through the comma-separated env parsing — and pass the
backend explicitly; the agent appends the ``rag_search`` tool itself, plus one
``rag_search_<slug>`` tool per source. The backend's ``as_instructions()`` block is
injected into the system prompt automatically; subclass the backend and override it
to customize the wording.

.. code-block:: python

   import asyncio
   from langchain_core.messages import AIMessage, SystemMessage
   from manolo_bot.ai.config import BotConfig, EmbeddingsConfig, LLMConfig
   from manolo_bot.ai.llmagent import LLMAgent
   from manolo_bot.ai.llmbot import LLMBuilder
   from manolo_bot.rag.embeddings import build_embeddings
   from manolo_bot.rag.factory import build_rag_backend
   from manolo_bot.rag.sources import RAGSource
   from manolo_bot.storage.messages.memory_storage import MemoryMessagesStorage


   async def main():
       llm_config = LLMConfig(
           google_api_key="your_api_key",
           google_api_model="",
           openai_api_key="",
           openai_api_model="",
           openai_api_base_url="",
           ollama_model="",
       )
       llm = LLMBuilder(llm_config).get_llm()
       bot_config = BotConfig(
           bot_uuid="my-bot",
           bot_name="Assistant",
           bot_username="assistant_bot",
           bot_token="123456:ABC",
           user_id=0,
       )
       storage = MemoryMessagesStorage(bot_uuid="my-bot", chat_id=123)
       await storage.refresh_messages()

       sources = [
           RAGSource("docs/handbook.md", "Employee handbook, covering policies and benefits"),
           RAGSource("docs/policies/", "Company policies"),
       ]
       # Embeddings can use a different provider than the chat model.
       emb_config = EmbeddingsConfig(google_api_key="your_api_key", google_embedding_model="models/gemini-embedding-2")
       backend = build_rag_backend("in_memory", build_embeddings(embeddings_config=emb_config), bot_uuid="my-bot")
       await backend.build_or_load()
       await backend.ingest(sources)

       system_instructions = [
           SystemMessage(content="You are a helpful assistant."),
           AIMessage(content="ok!"),
       ]
       agent = LLMAgent(
           llm=llm,
           bot_config=bot_config,
           system_instructions=system_instructions,
           messages_storage=storage,
           rag_backend=backend,
       )
       response = await agent.answer_message(chat_id=123, message="What does the handbook say?")
       print(f"Agent: {response.content}")


   if __name__ == "__main__":
       asyncio.run(main())

``build_embeddings(llm_config)`` is deprecated but still supported for backwards
compatibility: it derives the embeddings provider from the ``LLMConfig`` exactly as
before, so existing code keeps working unchanged. To decouple embeddings from the
chat provider, build an explicit ``EmbeddingsConfig`` — for example, chat on Gemini
while embedding with a local Ollama or a local OpenAI-compatible endpoint:

.. code-block:: python

   from manolo_bot.ai.config import EmbeddingsConfig
   # Chat on Gemini, embeddings on a local Ollama server:
   emb_config = EmbeddingsConfig(ollama_model="embeddinggemma")
   backend = build_rag_backend("in_memory", build_embeddings(embeddings_config=emb_config), bot_uuid="my-bot")
   # Or on a local OpenAI-compatible endpoint:
   emb_config = EmbeddingsConfig(openai_api_base_url="http://localhost:1234/v1", openai_embedding_model="text-embedding-3-small")
   backend = build_rag_backend("in_memory", build_embeddings(embeddings_config=emb_config), bot_uuid="my-bot")

``EmbeddingsConfig`` mirrors ``LLMConfig``: pass the settings for the provider you
want to use. If settings for more than one provider are present, Google is used
first, then OpenAI, then Ollama. If a RAG-specific API key is not set, the chat
model's key for that provider is used; a missing OpenAI key falls back to the
``OPENAI_API_KEY`` environment variable or the ``"not-needed"`` placeholder, so
key-less local servers (LM Studio, vLLM, Ollama) work out of the box. Default models:
Google ``models/gemini-embedding-2``, OpenAI ``text-embedding-3-small``, Ollama
``embeddinggemma``.

The backend doubles as a small document API: ``query()`` retrieves passages without an
agent, ``remove()`` drops documents, and ``list_sources()`` reports what is indexed.
Ingest is idempotent — re-ingesting unchanged files costs no embedding calls — and
per-file isolated, so one bad file can't fail the whole index.

For a custom index (e.g. Chroma, pgvector), subclass ``BaseRAGBackend`` from
``manolo_bot.rag.base`` and implement ``build_or_load()``, ``ingest()``, ``query()``,
``clear()``, ``remove()``, ``list_sources()``, ``needs_reindex()``, ``find_removed()``,
``as_tool()`` and ``as_source_tools()``; wire it up through ``build_rag_backend()``.

Scheduler
---------

The scheduler adds scheduled actions scoped to an opaque string token to ``LLMAgent`` and
``LLMDeepAgent``. The token is not necessarily a numeric Telegram chat id: it can be a Telegram
chat id (``"42"``), another platform's id (``"C012345"``) or a composite key such as
``"dashboard:{user}:{agent}:{uuid}"``. It is composed from two independent seams and injected
explicitly, mirroring RAG:

- a **store** decides *what* is due and persists actions (`in_memory`, `local_fs`, or `redis`);
- an **engine** decides *when* to check;
- ``SchedulerService`` composes them and exposes the agent tools plus ``sweep()`` / ``fire()``.

.. code-block:: python

   from langchain_core.messages import AIMessage, SystemMessage
   from manolo_bot.scheduler.engines import build_scheduler_engine
   from manolo_bot.scheduler.service import SchedulerService
   from manolo_bot.scheduler.stores import build_scheduler_store

   async def deliver(action):
       # Run the action through your agent and send the reply.
       # Raise on failure so the service records it.
       ...

   store = build_scheduler_store(
       "local_fs",
       bot_uuid=bot_config.bot_uuid,
       store_path="/tmp/manolo_bot/scheduler",
       default_timezone="UTC",
   )
   engine = build_scheduler_engine("asyncio", interval_seconds=5)
   scheduler = SchedulerService(store, engine, handler=deliver)

   system_instructions = [
       SystemMessage(content="You are a helpful assistant."),
       AIMessage(content="ok!"),
   ]
   agent = LLMAgent(
       ...,
       system_instructions=system_instructions,
       scheduler=scheduler,
       schedule_scope="dashboard:{user}:{agent}:{uuid}",  # None = messages_storage.chat_id
   )
   await scheduler.start()
   # ...
   await scheduler.stop()

``scheduler=`` appends ``schedule_action``, ``list_scheduled_actions`` and
``delete_scheduled_action`` to the agent's tools (an MCP/custom tool with the same name wins).
It also injects ``scheduler.as_instructions()`` and ``await scheduler.active_instructions(scope)``
(the same scope the tools use) into the system prompt; subclass ``SchedulerService`` and override
those methods to customize or suppress them.
The tools never take the scope as an argument, so actions cannot leak across scopes. They read
it from the run's ``RunnableConfig`` metadata — ``metadata["schedule_scope"]`` first, then
``metadata["chat_id"]``. With ``schedule_scope=`` set on the agent, that value is added to the
metadata and also used for ``await scheduler.active_instructions(...)``; with the default
``None``, the tools fall back to the per-run chat id while the prompt block uses
``messages_storage.chat_id``, preserving classic per-chat behavior. Setting the same scope for
both keeps the tools and the active-schedules prompt block in agreement.

This separation lets a consumer keep one conversation storage key while scheduling per real
chat — build one agent per chat (or per scope) over the same storage key:

.. code-block:: python

   # Conversation history is shared under one key...
   storage = MemoryMessagesStorage(bot_uuid="my-bot", chat_id="my-bot:shared")
   # ...while schedules belong to this real chat's opaque scope.
   agent = LLMAgent(
       ...,
       messages_storage=storage,
       scheduler=scheduler,
       schedule_scope="C012345",
   )

The Telegram app never sets ``schedule_scope``, so it keeps using the numeric chat id.

A scheduled prompt can ask the agent to stay silent when a condition is not met by replying
``NO_ANSWER``. The agent turn still runs and is persisted, but no chat message is sent — useful
for recurring checks such as “every day, check whether there is a match tomorrow; only send a
preview if there is one”.

The scope's active schedules are embedded in the agent's system prompt (when the Telegram app
builds it), so the agent still knows about them after ``/flushcontext``. When an action fires its
id is included in the trigger marker, allowing the agent to cancel that exact action without
listing first.

Use ``build_scheduler_engine("manual")`` when an external process owns timing; call
``await scheduler.sweep()`` from that process (or library endpoint) instead of running an
in-process loop.

For a custom persistence backend, subclass ``BaseSchedulerStore`` from
``manolo_bot.scheduler.base`` and implement ``build_or_load()``, ``add()``, ``list()``,
``delete()``, ``claim_due()``, ``claim_one()``, ``mark_delivered()``, ``record_failure()``,
``clear()`` and ``list_all()``; reuse the state-machine helpers in that module so claim and
failure semantics stay identical. Register it in ``build_scheduler_store()``.

For a custom trigger, subclass ``BaseSchedulerEngine`` and call ``dispatch.sweep()`` (or
``dispatch.fire(chat_id, action_id)``); register it in ``build_scheduler_engine()``.
Out-of-process engines (system cron, Celery workers) work through the same service but need a
shared store and a shared message store, since their worker runs outside the bot process.

Dynamic System Instructions
---------------------------

You can make your system instructions dynamic by using placeholders and a mapping dictionary. This is useful for injecting real-time information, such as the current date, user-specific data, or any other context that changes over time.

To use this feature:
1. Include a placeholder in your system instructions string (e.g., ``{current_time}``).
2. Pass a ``system_instructions_mapping`` dictionary to the ``LLMBot`` or ``LLMAgent`` constructor.
3. The keys in the mapping should match your placeholders, and the values should be callable functions that take the ``bot`` instance as their only argument.

.. code-block:: python

   import datetime
   from manolo_bot.ai.llmbot import LLMBot
   from langchain_core.messages import SystemMessage

   def get_datetime(bot) -> str:
       return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

   instructions = [SystemMessage(content="You are a helpful assistant. The current time is {time}.")]
   
   mapping = {
       "{time}": get_datetime
   }

   bot = LLMBot(
       llm=llm,
       bot_config=bot_config,
       system_instructions=instructions,
       messages_storage=storage,
       system_instructions_mapping=mapping
   )

   # Every time bot.system_instructions is accessed, the placeholder will be updated
   # with the result of the callable function.
