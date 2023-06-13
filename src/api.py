from typing import List, Type

from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from steamship import Steamship
from steamship.agents.mixins.transports.telegram import TelegramTransportConfig
from steamship.invocable import Config
from steamship.utils.repl import AgentREPL
from steamship_langchain.llms import OpenAIChat
from steamship_langchain.memory import ChatMessageHistory

from agent.base import LangChainAgent, LangChainTelegramBot
from agent.tools.image import GenerateImageTool
from agent.tools.my_tool import MyTool
from agent.tools.search import SearchTool
from prompts import SUFFIX, FORMAT_INSTRUCTIONS, PREFIX

MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4.0"
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
                "prefix": PREFIX,
                "suffix": SUFFIX,
                "format_instructions": FORMAT_INSTRUCTIONS,
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


class MyBotConfig(TelegramTransportConfig):
    pass


class MyBot(LangChainTelegramBot):

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return MyBotConfig

    def __init__(self, **kwargs):
        super().__init__(agent=MyAgent(), **kwargs)


if __name__ == "__main__":
    AgentREPL(
        MyBot,
        method="prompt",
        agent_package_config={"botToken": "not-a-real-token-for-local-testing"},
    ).run()
