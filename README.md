# Multi-Modal LangChain agents in Production
[![Open in a VS Code Dev Container](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/steamship-packages/langchain-telegram-chatbot)
[![Steamship](https://raw.githubusercontent.com/steamship-core/python-client/main/badge.svg)](https://www.steamship.com/build?utm_source=github&utm_medium=badge&utm_campaign=awesome_gpt_prompts&utm_id=awesome_gpt_prompts)

This starter project contains the necessary scaffolding to deploy LangChain Agents with memory and connect them to Telegram.

Get started:
https://twitter.com/eniascailliau/status/1658544730324492303

Add/Edit voice: 
https://twitter.com/eniascailliau/status/1658841969211088905

## 🚀 Features

- 🧠 Support for OpenAI GPT-4 and GPT-3.5 
- 🔗 Embeddable chat window
- 🔌 Connect your chatbot to Telegram
- 🔈 Give your Agent a voice
- 💸 Moneytize your agent 

## Quick-start guide 🛠️

Getting your agent online only takes 4 steps. If not, shoot me a message on [Discord](https://steamship.com/discord). Happy to help you out. 

1. Clone the repository
2. Add your agent to `src/api.py`
3. Install required dependencies: `pip install --upgrade steamship_langchain`
4. Run `ship deploy && ship use`



## Getting started 

To run your companion locally:

```
pip install -r requirements.txt
python main.py 
```

To deploy your companion & connect it to Telegram:

```
pip install steamship
ship deploy && ship use 
```

You will need to fetch a Telegram key to connect your companion to Telegram. [This guide](/docs/register-telegram-bot.md) will show you how.


## Development 😎


**..in a Local VS Code Container**

Just click here: [![Open in a VS Code Dev Container](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/steamship-core/hackathon-starter)

**..in a Web VS Code Container**

[![Open in a VS Code Dev Container](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/steamship-packages/langchain-telegram-chatbot)


**..on localhost with your own IDE**

Clone this repository, then set up a Python virtual environment with:

```bash
pip install -r requirements.txt
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Happy building! 🎉
