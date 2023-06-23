import logging
import re
import uuid
from abc import abstractmethod
from typing import List, Optional

from langchain.agents import Tool, AgentExecutor
from langchain.memory.chat_memory import BaseChatMemory
from steamship import Block, Steamship
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import (
    TelegramTransportConfig,
    TelegramTransport,
)
from steamship.agents.schema import (
    AgentContext,
    Metadata,
    Agent,
)
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import post

from agent.utils import is_uuid, UUID_PATTERN


class LangChainTelegramBot(AgentService):
    """Deployable Multimodal Agent that illustrates a character personality with voice.

    NOTE: To extend and deploy this agent, copy and paste the code into api.py.
    """

    USED_MIXIN_CLASSES = [TelegramTransport, SteamshipWidgetTransport]
    config: TelegramTransportConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_mixin(
            SteamshipWidgetTransport(client=self.client, agent_service=self, agent=None)
        )

        self.add_mixin(
            TelegramTransport(
                client=self.client, config=self.config, agent_service=self, agent=None
            )
        )

    @abstractmethod
    def get_agent(self, client: Steamship, chat_id: str) -> AgentExecutor:
        raise NotImplementedError()

    @abstractmethod
    def get_memory(self, client: Steamship, chat_id: str) -> BaseChatMemory:
        raise NotImplementedError()

    @abstractmethod
    def get_tools(self, client: Steamship, chat_id: str) -> List[Tool]:
        raise NotImplementedError()

    def voice_tool(self) -> Optional[Tool]:
        return None

    def respond(
        self, incoming_message: Block, chat_id: str, client: Steamship
    ) -> List[Block]:

        if incoming_message.text == "/start":
            return [Block(text=f"New conversation started. chat_id: {chat_id}")]

        if incoming_message.text == "/reset":
            self.get_memory(self.client, chat_id).chat_memory.clear()
            return [Block(text="Conversation log cleared.")]

        conversation = self.get_agent(
            client=client,
            chat_id=chat_id,
        )
        response = conversation.run(input=incoming_message.text)

        def replace_markdown_with_uuid(text):
            pattern = r"(?:!\[.*?\]|)\((.*?)://?(.*?)\)"
            return re.sub(pattern, r"\2", text)

        response = replace_markdown_with_uuid(response)
        response = UUID_PATTERN.split(response)
        response = [re.sub(r"^\W+", "", el) for el in response]
        if audio_tool := self.voice_tool():
            response_messages = []
            for message in response:
                response_messages.append(message)
                if not is_uuid(message):
                    audio_uuid = audio_tool.run(message)
                    response_messages.append(audio_uuid)
        else:
            response_messages = response

        return [
            Block.get(self.client, _id=response)
            if is_uuid(response)
            else Block(text=response)
            for response in response_messages
        ]

    def run_agent(self, agent: Agent, context: AgentContext):
        chat_id = context.metadata.get("chat_id")

        incoming_message = context.chat_history.last_user_message
        output_messages = self.respond(incoming_message, chat_id, context.client)
        for func in context.emit_funcs:
            logging.info(f"Emitting via function: {func.__name__}")
            func(output_messages, context.metadata)

    @post("prompt")
    def prompt(self, prompt: str) -> str:
        """Run an agent with the provided text as the input."""

        context = AgentContext.get_or_create(self.client, {"id": str(uuid.uuid4())})
        context.chat_history.append_user_message(prompt)

        output = ""

        def sync_emit(blocks: List[Block], meta: Metadata):
            nonlocal output
            for block in blocks:
                if not block.is_text():
                    block.set_public_data(True)
                    output += f"({block.mime_type}: {block.raw_data_url})\n"
                else:
                    output += f"{block.text}\n"

        context.emit_funcs.append(sync_emit)
        self.run_agent(None, context)  # Maybe I override this
        return output
