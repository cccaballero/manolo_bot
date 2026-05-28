Using as a Library
==================

You can integrate `manolo-bot`'s core functionality into your own Python applications.

Core Components
---------------

The main classes exposed by the library are:

* `LLMBot`: The base class for LLM-powered chat bots.
* `LLMBuilder`: A factory class to create LLM instances (Gemini, OpenAI, Ollama).
* `LLMAgent`: An advanced agent capable of using tools and MCP servers.
* `BaseMessagesStorage`: Abstract base class for message persistence.

Basic Example
-------------

Here is how to initialize an `LLMBot` using Google Gemini:

.. code-block:: python

   from manolo_bot import LLMBot, LLMBuilder
   from manolo_bot.config import LLMConfig, BotConfig
   from manolo_bot.storage.memory_storage import MemoryMessagesStorage

   # Configure LLM
   llm_config = LLMConfig(
       llm_provider="google",
       google_api_key="your_google_api_key"
   )
   llm = LLMBuilder(llm_config).get_llm()

   # Configure Bot
   bot_config = BotConfig(
       bot_uuid="my-bot-id",
       bot_name="My Assistant"
   )

   # Storage
   storage = MemoryMessagesStorage()

   # Initialize Bot
   bot = LLMBot(
       llm=llm,
       config=bot_config,
       system_instructions="You are a helpful assistant.",
       storage=storage
   )

   # Process a message
   # response = await bot.process_text("Hello!")

Using the Agent with Tools
--------------------------

To use the advanced agent with tools:

.. code-block:: python

   from manolo_bot import LLMAgent
   from manolo_bot.ai.tools import get_tools

   tools = get_tools()
   agent = LLMAgent(
       llm=llm,
       config=bot_config,
       system_instructions="You are a tool-using assistant.",
       storage=storage,
       tools=tools
   )
