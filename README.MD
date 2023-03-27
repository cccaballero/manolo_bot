# Telegram Chat Bot using OpenAI ChatGPT 
 
This is an experimental Telegram chat bot that uses the OpenAI ChatGPT model to generate responses. With this bot, you can have engaging and realistic conversations with an artificial intelligence model. 
 
## Getting Started 
 
### Prerequisites 
 
First, you need to install the required packages using pip:
pip install -r requirements.txt
### Configuration 
 
You need to provide the following environment variables.

```
OPENAI_API_KEY='your_openai_api_key'
TELEGRAM_BOT_NAME='your_telegram_bot_name'
TELEGRAM_BOT_USERNAME='your_telegram_bot_username'
TELEGRAM_BOT_TOKEN='your_telegram_bot_api_token'
```

You can create a bot on Telegram and get its API token by following the [official instructions](https://core.telegram.org/bots#how-do-i-create-a-bot). You can also get an OpenAI API key by creating an account on their [website](https://platform.openai.com/signup/).

You can also use the following not required environment variables:

```
TELEGRAM_BOT_INSTRUCTIONS='Your bot instructions'
```
For setting the ChatGPT API system user instructions

```
TELEGRAM_ALLOWED_CHATS='chatid1, chatid2'
```
A comma-separated list of allowed telegram chat ids

### Running the Bot 
 
You can run the bot using the following command:
```
python main.py
```

## Contributing 
 
If you'd like to contribute to this project, feel free to submit a pull request. We're always open to new ideas or improvements to the code.  
 
## License 
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.