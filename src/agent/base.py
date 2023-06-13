import uuid
from abc import ABC, abstractmethod
from typing import List

from langchain.agents import Tool, AgentExecutor
from langchain.memory.chat_memory import BaseChatMemory
from steamship import Block, Steamship
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import TelegramTransport, TelegramTransportConfig
from steamship.agents.schema import (
    AgentContext,
    Metadata,
    Action,
    FinishAction,
    Agent,
)
from steamship.agents.service.agent_service import AgentService
from steamship.data.tags.tag_constants import RoleTag
from steamship.invocable import post

from agent.utils import is_uuid, UUID_PATTERN

MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4.0"
TEMPERATURE = 0.7
VERBOSE = True


class LangChainAgent(Agent, ABC):
    tools: List[Tool] = None

    @abstractmethod
    def get_agent(self, client: Steamship, chat_id: str) -> AgentExecutor:
        raise NotImplementedError()

    @abstractmethod
    def get_memory(self, client: Steamship, chat_id: str) -> BaseChatMemory:
        raise NotImplementedError()

    @abstractmethod
    def get_tools(self, client: Steamship, chat_id: str) -> List[Tool]:
        raise NotImplementedError()

    def _agent_output_to_chat_messages(
            self, client: Steamship, chat_id: str, response_messages: List[str]
    ) -> List[Block]:
        """Transform the output of the Multi-Modal Agent into a list of ChatMessage objects.

        The response of a Multi-Modal Agent contains one or more:
        - parseable UUIDs, representing a block containing binary data, or:
        - Text

        This method inspects each string and creates a ChatMessage of the appropriate type.
        """
        ret = []
        for response in response_messages:
            if is_uuid(response):
                block = Block.get(client, _id=response)
                block.set_public_data(True)
                message = Block(**block.dict(), client=client)
                message.url = block.raw_data_url
                message.set_chat_role(RoleTag.AGENT)
            else:
                message = Block(
                    text=response,
                )
            ret.append(message)
        return ret

    def next_action(self, context: AgentContext) -> Action:
        chat_id = context.metadata.get("chat_id")

        incoming_message = context.chat_history.last_user_message

        if incoming_message.text == "/start":
            message = Block(text="New conversation started.")
            return FinishAction(output=[message])

        conversation = self.get_agent(
            client=context.client,
            chat_id=chat_id,
        )
        response = conversation.run(input=incoming_message.text)
        response = UUID_PATTERN.split(response)
        output_messages = self._agent_output_to_chat_messages(client=context.client,
                                                              chat_id=chat_id,
                                                              response_messages=response)
        output_messages.append(Block(text=f"Chat id: {chat_id}"))
        return FinishAction(output=output_messages)


class LangChainTelegramBot(AgentService):
    """Deployable Multimodal Agent that illustrates a character personality with voice.

    NOTE: To extend and deploy this agent, copy and paste the code into api.py.
    """
    config: TelegramTransportConfig

    def __init__(self, agent: LangChainAgent, **kwargs):
        super().__init__(**kwargs)

        self._agent = agent

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

    @post("prompt")
    def prompt(self, prompt: str) -> str:
        """Run an agent with the provided text as the input."""

        context = AgentContext.get_or_create(self.client, {"id": str(uuid.uuid4()),
                                                           "chat_id": "123"})
        context.chat_history.append_user_message(prompt)

        output = ""

        def sync_emit(blocks: List[Block], meta: Metadata):
            nonlocal output
            output += "\n".join(
                [b.text if b.is_text() else f"({b.mime_type}: {b.id})" for b in blocks]
            )

        context.emit_funcs.append(sync_emit)
        self.run_agent(self._agent, context)
        return output
