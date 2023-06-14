from typing import List, Type

from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from pydantic import Field
from steamship import Steamship
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import TelegramTransportConfig, TelegramTransport
from steamship.invocable import Config
from steamship.utils.repl import AgentREPL
from steamship_langchain.llms import OpenAIChat
from steamship_langchain.memory import ChatMessageHistory

from agent.base import LangChainAgent, LangChainTelegramBot
from agent.tools.image import GenerateImageTool
from agent.tools.my_tool import MyTool
from agent.tools.search import SearchTool

MODEL_NAME = "gpt-4"  # or "gpt-4.0"
TEMPERATURE = 0.7
VERBOSE = True


class MyAgent(LangChainAgent):

    def get_agent(self, client: Steamship, chat_id: str) -> AgentExecutor:
        llm = OpenAIChat(
            client=client,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            verbose=VERBOSE,
        )

        tools = self.get_tools(client=client, chat_id=chat_id)

        memory = self.get_memory(client=client, chat_id=chat_id)

        return initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            agent_kwargs={
                # "prefix": PREFIX,
                # "suffix": SUFFIX,
                # "format_instructions": FORMAT_INSTRUCTIONS,
            },
            verbose=VERBOSE,
            memory=memory,
        )

    def get_memory(self, client: Steamship, chat_id: str):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=ChatMessageHistory(
                client=client, key=f"history-{chat_id or 'default'}"
            ),
            return_messages=True,
        )
        return memory

    def get_tools(self, client: Steamship, chat_id: str) -> List[Tool]:
        return [
            SearchTool(client),
            MyTool(client),
            GenerateImageTool(client),
            # VideoMessageTool(client),
        ]


class ChatbotConfig(TelegramTransportConfig):
    bot_token: str = Field(
        description="Your telegram bot token.\nLearn how to create one here: "
                    "https://github.com/steamship-packages/langchain-agent-production-starter/blob/main/docs/register-telegram-bot.md"
    )
    elevenlabs_api_key: str = Field(
        default="", description="Optional API KEY for ElevenLabs Voice Bot"
    )
    elevenlabs_voice_id: str = Field(
        default="", description="Optional voice_id for ElevenLabs Voice Bot"
    )
    chat_ids: str = Field(
        default="", description="Comma separated list of whitelisted chat_id's"
    )
    use_gpt4: bool = Field(
        True,
        description="If True, use GPT-4. Use GPT-3.5 if False. "
                    "GPT-4 generates better responses at higher cost and latency.",
    )


class MyBot(LangChainTelegramBot):

    USED_MIXIN_CLASSES = [TelegramTransport, SteamshipWidgetTransport]

    def __init__(self, **kwargs):
        super().__init__(agent=MyAgent(), **kwargs)
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client, agent_service=self, agent=self._agent
            )
        )

        self.add_mixin(
            TelegramTransport(
                client=self.client,
                config=self.config,
                agent_service=self,
                agent=self._agent
            )
        )

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return ChatbotConfig


if __name__ == "__main__":
    AgentREPL(
        MyBot,
        method="prompt",
        agent_package_config={"botToken": "not-a-real-token-for-local-testing"},
    ).run()
