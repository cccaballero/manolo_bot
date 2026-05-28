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

   from manolo_bot.config import LLMConfig
   from manolo_bot import LLMBuilder

   # Example: Google Gemini
   llm_config = LLMConfig(google_api_key="your_api_key")
   llm = LLMBuilder(llm_config).get_llm()

Bot Configuration
~~~~~~~~~~~~~~~~~

The `BotConfig` class defines settings like the bot's name, UUID (for unique identification in storage), and features like multimodal support.

.. code-block:: python

   from manolo_bot.config import BotConfig

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

   from manolo_bot.storage.memory_storage import MemoryMessagesStorage

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
   from manolo_bot import LLMAgent, LLMBuilder
   from manolo_bot.config import LLMConfig, BotConfig
   from manolo_bot.storage.memory_storage import MemoryMessagesStorage
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

       # 5. Answer a message
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

   from manolo_bot import LLMBot

   # ... (Setup is identical to LLMAgent, but without tools)

   bot = LLMBot(
       llm=llm,
       config=bot_config,
       system_instructions="You are a simple chatbot.",
       storage=storage
   )

   response = await bot.answer_message(chat_id=chat_id, message="Hello!")
