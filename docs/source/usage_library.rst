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
   llm_config = LLMConfig(google_api_key="your_api_key")
   llm = LLMBuilder(llm_config).get_llm()

Bot Configuration
~~~~~~~~~~~~~~~~~

The `BotConfig` class defines settings like the bot's name, UUID (for unique identification in storage), and features like multimodal support.

.. code-block:: python

   from manolo_bot.ai.config import BotConfig

   bot_config = BotConfig(
       bot_uuid="my-unique-bot-id",
       bot_name="MyAssistant"
   )

Storage
~~~~~~~

The `storage` component is responsible for persisting conversation history. `manolo-bot` uses a "per-chat" storage model. Each user or conversation should have its own storage instance identified by a `chat_id`.

* **MemoryMessagesStorage**: Stores messages in RAM. Useful for testing or simple CLI tools. History is lost when the process restarts.
* **RedisMessagesStorage**: Stores messages in a Redis database. Recommended for production and persistent history.

.. code-block:: python

   from manolo_bot.storage.messages.memory import MemoryMessagesStorage

   # chat_id is a unique identifier for the current conversation (e.g., a user ID)
   storage = MemoryMessagesStorage(bot_uuid="my-unique-bot-id", chat_id=12345)

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
   from manolo_bot.storage.messages.memory import MemoryMessagesStorage
   from manolo_bot.ai.tools import get_tools

   async def main():
       # 1. Setup LLM (Must support Tool Calling)
       llm_config = LLMConfig(google_api_key="your_key")
       llm = LLMBuilder(llm_config).get_llm()

       # 2. Setup Bot Identity
       bot_config = BotConfig(bot_uuid="bot-1", bot_name="Assistant")

       # 3. Setup Storage
       chat_id = 1001
       storage = MemoryMessagesStorage(bot_uuid="bot-1", chat_id=chat_id)
       await storage.refresh_messages()

       # 4. Initialize the Agent with Tools
       tools = get_tools()
       agent = LLMAgent(
           llm=llm,
           config=bot_config,
           system_instructions="You are a helpful assistant.",
           storage=storage,
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
