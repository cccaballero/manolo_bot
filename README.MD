# Telegram Chat Bot using LLM
 
This is an experimental Telegram chat bot that uses a configurable LLM model to generate responses. With this bot, you can have engaging and realistic conversations with an artificial intelligence model. 
 
## Getting Started 
 
### Prerequisites 
 
First, you need to install the required packages using pip:
pip install -r requirements.txt

### Configuration 
 
You can copy and rename the provided `env.example` to `.env` and edit the file according your data

You can create a bot on Telegram and get its API token by following the [official instructions](https://core.telegram.org/bots#how-do-i-create-a-bot).

#### Required environment variables.

You can use the `GOOGLE_API_KEY`, `OPENAI_API_KEY` or `OPENAI_API_BASE_URL` for selecting the required LLM provider.
The `OPENAI_API_BASE_URL` will look for an OpenAI API like, as the LM Studio API

<b>Note:</b> When `GOOGLE_API_KEY` option is selected the model used will be Gemini Pro and multimodal capabilities will be enabled for images.

`TELEGRAM_BOT_NAME`: Your Telegram bot name

`TELEGRAM_BOT_USERNAME`: Your Telegram bot name

`TELEGRAM_BOT_TOKEN`: Your Telegram bot token

#### Selecting OpenAI Model.

`OPENAI_API_MODEL`: LLM to use for OpenAI or OpenAI-like API, if not provided the default model will be used.

#### Enabling image Generation with Stable Diffusion

`WEBUI_SD_API_URL`: you can define a Stable Diffusion Web UI API URL for image generation. If this option is enabled the bot will answer image generation requests using Stable Diffusion generated images.

`WEBUI_SD_API_PARAMS`: A JSON string containing Stable Diffusion Web UI API params. If not provided default params for SDXL Turbo model will me used.

#### Setting custom bot instructions

`TELEGRAM_BOT_INSTRUCTIONS`: You can define custom LLM system instructions using this variable.

#### Limiting Bot interaction

`TELEGRAM_ALLOWED_CHATS`: You can use a comma separated allowed chat IDs for limiting bot interaction to those chats.

### Running the Bot 
 
You can run the bot using the following command:
```
python main.py
```

## Contributing 
 
If you'd like to contribute to this project, feel free to submit a pull request. We're always open to new ideas or improvements to the code.  
 
## License 
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.