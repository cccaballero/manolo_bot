# Telegram Chat Bot using LLM
 
This is an experimental Telegram chat bot that uses a configurable LLM model to generate responses. With this bot, you can have engaging and realistic conversations with an artificial intelligence model. 
 
## Getting Started 
 
### Prerequisites 
 
First, you need to install the required packages using pip:
pip install -r requirements.txt

### Configuration 
 
You can copy and rename the provided `env.example` to `.env` and edit the file according your data

You can create a bot on Telegram and get its API token by following the [official instructions](https://core.telegram.org/bots#how-do-i-create-a-bot).

For use the bot on a group you have to use the @BotFather bot to [set the Group Privacy off](https://stackoverflow.com/questions/50204633/allow-bot-to-access-telegram-group-messages/50236522#50236522). This let the bot to access all the group messages.

#### Required environment variables.

You can use the `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `OPENAI_API_BASE_URL` or `OLLAMA_MODEL` for selecting the required 
LLM provider.
The `OPENAI_API_BASE_URL` will look for an OpenAI API like, as the LM Studio API

- <b>Note:</b> When `GOOGLE_API_KEY` option is selected the default model used will be Gemini 2.0 Flash.

`TELEGRAM_BOT_NAME`: Your Telegram bot name

`TELEGRAM_BOT_USERNAME`: Your Telegram bot name

`TELEGRAM_BOT_TOKEN`: Your Telegram bot token

#### Selecting OpenAI Model.

`OPENAI_API_MODEL`: LLM to use for OpenAI or OpenAI-like API, if not provided the default model will be used.

#### Selecting Google API Model.

`GOOGLE_API_MODEL`: LLM to use for Google API, if not provided the default model will be used.

#### Enabling image Generation with Stable Diffusion

`WEBUI_SD_API_URL`: you can define a Stable Diffusion Web UI API URL for image generation. If this option is enabled the bot will answer image generation requests using Stable Diffusion generated images.

`WEBUI_SD_API_PARAMS`: A JSON string containing Stable Diffusion Web UI API params. If not provided default params for SDXL Turbo model will me used.

#### Setting custom bot character instructions

`TELEGRAM_BOT_INSTRUCTIONS_CHARACTER`: You can define a custom character for the bot instructions. 
This will override the default bot character. For example: `You are a software engineer, geek and nerd, user of linux and free software technologies.`

#### Setting extra bot instructions

`TELEGRAM_BOT_INSTRUCTIONS_EXTRA`: You can include extra LLM system instructions using this variable.

#### Setting custom bot instructions

`TELEGRAM_BOT_INSTRUCTIONS`: You can define custom LLM system instructions using this variable. 
This will override the default instructions, and the custom bot character instructions.

#### Limiting Bot interaction

`TELEGRAM_ALLOWED_CHATS`: You can use a comma separated allowed chat IDs for limiting bot interaction to those chats.

#### Enable multimodal capabilities

`ENABLE_MULTIMODAL`: Enable multimodal capabilities for images (True, False). The selected model must support multimodal capabilities.

#### Enable group assistant

`ENABLE_GROUP_ASSISTANT`: Enable group assistant for group chats (True, False). The bot will answer to group chats with a question mark. Default is False.

#### Enable rate limiting

`RATE_LIMITER_REQUESTS_PER_SECOND`: The number of requests per second allowed by the bot.
`RATE_LIMITER_CHECK_EVERY_N_SECONDS`: The number of seconds between rate limit checks.
`RATE_LIMITER_MAX_BUCKET_SIZE`: The maximum bucket size for rate limiting.

#### Set preferred language

`PREFERRED_LANGUAGE`: The preferred language for the bot. (English, Spanish, etc.)

#### Set context max tokens

`CONTEXT_MAX_TOKENS`: The maximum number of tokens allowed for the bot's context.

### Running the Bot 
 
You can run the bot using the following command:

```shell
python main.py
```

## Developers information

Use `requirements-dev.txt` to install the development dependencies.

### Pre-commit hooks

After installing the development dependencies, For installing pre-commit scripts including ruff checks, you can run 
the following command:

```shell
pre-commit install
```

### Running tests

You can run the tests using the following command:

```shell
python -m unittest discover
```

## Contributing 
 
If you'd like to contribute to this project, feel free to submit a pull request. We're always open to new ideas or improvements to the code.  
 
## License 
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.