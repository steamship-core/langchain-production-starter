import logging
import uuid
from abc import abstractmethod
from typing import List

from langchain.agents import Tool, AgentExecutor
from langchain.memory.chat_memory import BaseChatMemory
from steamship import Block, Steamship
from steamship.agents.mixins.transports.telegram import TelegramTransportConfig
from steamship.agents.schema import (
    AgentContext,
    Metadata,
    Agent,
)
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import post

from agent.utils import is_uuid, UUID_PATTERN

MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4.0"
TEMPERATURE = 0.7
VERBOSE = True


def _agent_output_to_chat_messages(
        client: Steamship, response_messages: List[str]
) -> List[Block]:
    """Transform the output of the Multi-Modal Agent into a list of ChatMessage objects.

    The response of a Multi-Modal Agent contains one or more:
    - parseable UUIDs, representing a block containing binary data, or:
    - Text

    This method inspects each string and creates a ChatMessage of the appropriate type.
    """
    return [Block.get(client, _id=response) if is_uuid(response) else Block(text=response)
            for response in response_messages]


class LangChainTelegramBot(AgentService):
    """Deployable Multimodal Agent that illustrates a character personality with voice.

    NOTE: To extend and deploy this agent, copy and paste the code into api.py.
    """
    config: TelegramTransportConfig

    @abstractmethod
    def get_agent(self, client: Steamship, chat_id: str) -> AgentExecutor:
        raise NotImplementedError()

    @abstractmethod
    def get_memory(self, client: Steamship, chat_id: str) -> BaseChatMemory:
        raise NotImplementedError()

    @abstractmethod
    def get_tools(self, client: Steamship, chat_id: str) -> List[Tool]:
        raise NotImplementedError()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def respond(self, incoming_message: Block, chat_id: str, client: Steamship) -> List[Block]:

        if incoming_message.text == "/start":
            return [Block(text="New conversation started.")]

        conversation = self.get_agent(
            client=client,
            chat_id=chat_id,
        )
        response = conversation.run(input=incoming_message.text)
        return [Block.get(self.client, _id=response) if is_uuid(response) else Block(text=response)
                for response in UUID_PATTERN.split(response)]

    def run_agent(self, agent: Agent, context: AgentContext):
        chat_id = context.metadata.get("chat_id")

        incoming_message = context.chat_history.last_user_message
        output_messages = self.respond(incoming_message, chat_id, context.client)
        output_messages.append(Block(text=f"Chat id: {chat_id}"))
        output_messages.append(Block(text=f"incoming_message: {incoming_message.message_id}"))
        output_messages.append(Block(text=f"incoming_message: {incoming_message.chat_id}"))
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
            output += "\n".join(
                [b.text if b.is_text() else f"({b.mime_type}: {b.id})" for b in blocks]
            )

        context.emit_funcs.append(sync_emit)
        self.run_agent(None, context)  # Maybe I override this
        return output
