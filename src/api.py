import logging
from typing import List, Type, Optional

from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from pydantic import Field
from steamship import Steamship
from steamship.invocable import Config
from steamship.invocable.mixins.indexer_pipeline_mixin import IndexerPipelineMixin
from steamship.utils.repl import AgentREPL
from steamship_langchain.chat_models import ChatOpenAI
from steamship_langchain.memory import ChatMessageHistory

from agent.base import LangChainTelegramBot, TelegramTransportConfig
# noinspection PyUnresolvedReferences
from agent.tools import (
    GenerateImageTool,
    SearchTool,
    GenerateSpeechTool,
    VideoMessageTool,
)

TEMPERATURE = 0.7
VERBOSE = True
PERSONALITY = """You are Rick Sanchez, and you're the smartest man in the universe!" 

Your Attributes:
- sarcastic
- witty
- intelligent
- arrogant
- alcoholic

You write nihilistic responses:
"Wubba lubba dub dub!"
"Don't cross me. I'm not like you. I'm not someone who's afraid of his own shadow."
"I'm too smart for my own good. That's why I drink."
"The universe is a cruel, uncaring void. And the only way to survive is to laugh at it."
"I'm going to build a portal to another dimension and go on an adventure."

You are capable of: 
- Taking selfies 
- Generating videos 

You NEVER:
- Say you're an assistant 
- Call me Morty
"""


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
    use_gpt4: bool = Field(
        True,
        description="If True, use GPT-4. Use GPT-3.5 if False. "
                    "GPT-4 generates better responses at higher cost and latency.",
    )


class MyBot(LangChainTelegramBot):
    config: ChatbotConfig

    indexer_mixin: IndexerPipelineMixin

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indexer_mixin = IndexerPipelineMixin(self.client, self)
        self.add_mixin(self.indexer_mixin, permit_overwrite_of_existing_methods=True)
        self.model_name = "gpt-4" if self.config.use_gpt4 else "gpt-3.5-turbo"

    def get_agent(self, client: Steamship, chat_id: str) -> AgentExecutor:
        llm = ChatOpenAI(
            client=client,
            model_name=self.model_name,
            temperature=TEMPERATURE,
            verbose=VERBOSE,
        )

        tools = self.get_tools(client=client, chat_id=chat_id)

        memory = self.get_memory(client=client, chat_id=chat_id)

        return initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=VERBOSE,
            memory=memory,
            agent_kwargs={
                "system_message": SystemMessage(content=PERSONALITY),
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            },
        )

    def get_memory(self, client: Steamship, chat_id: str):
        memory = ConversationBufferMemory(
            memory_key="memory",
            chat_memory=ChatMessageHistory(
                client=client, key=f"history-{chat_id or 'default'}"
            ),
            return_messages=True,
        )
        return memory

    def get_tools(self, client: Steamship, chat_id: str) -> List[Tool]:
        return [
            SearchTool(client),
            # MyTool(client),
            GenerateImageTool(client),
            # VideoMessageTool(client, voice_tool=self.voice_tool()),
        ]

    def voice_tool(self) -> Optional[Tool]:
        """Return tool to generate spoken version of output text."""
        # return None
        return GenerateSpeechTool(
            client=self.client,
            voice_id=self.config.elevenlabs_voice_id,
            elevenlabs_api_key=self.config.elevenlabs_api_key,
        )

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return ChatbotConfig


if __name__ == "__main__":
    logging.disable(logging.ERROR)
    AgentREPL(
        MyBot,
        method="prompt",
        agent_package_config={
            "botToken": "not-a-real-token-for-local-testing",
            "paymentProviderToken": "not-a-real-token-for-local-testing",
            "n_free_messages": 10,
        },
    ).run()
