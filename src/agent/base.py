import logging
import re
import uuid
from abc import abstractmethod
from typing import List, Optional, Type

import requests
from langchain.agents import Tool, AgentExecutor
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import Field
from steamship import Block, Steamship
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.schema import (
    AgentContext,
    Metadata,
    Agent,
)
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import post, Config

from agent.telegram import ExtendedTelegramTransport
from agent.usage_tracking import UsageTracker
from agent.utils import is_uuid, UUID_PATTERN


class TelegramTransportConfig(Config):
    bot_token: str = Field(description="Telegram bot token, obtained via @BotFather")
    payment_provider_token: str = Field(description="Payment provider token, obtained via @BotFather")
    n_free_messages: int = Field(0, description="Number of free messages assigned to new users.")
    api_base: str = Field("https://api.telegram.org/bot", description="The root API for Telegram")


class LangChainTelegramBot(AgentService):
    """Deployable Multimodal Agent that illustrates a character personality with voice.

    NOTE: To extend and deploy this agent, copy and paste the code into api.py.
    """

    def send_invoice(self, chat_id):
        requests.post(
            f"{self.config.api_base}{self.config.bot_token}/sendInvoice",
            json={
                "chat_id": chat_id,
                "payload": "50",
                "currency": "USD",
                "title": "🍏 50 messages",
                "description": "Tap the button below and pay",
                "prices": [{
                    "label": "🍏 50 messages",
                    "amount": 599,
                }],
                "provider_token": self.config.payment_provider_token
            },
        )

    def set_payment_plan(self, pre_checkout_query):
        chat_id = str(pre_checkout_query["from"]["id"])
        payload = int(pre_checkout_query["invoice_payload"])
        self.usage.increase_message_limit(chat_id, payload)

    USED_MIXIN_CLASSES = [ExtendedTelegramTransport, SteamshipWidgetTransport]
    config: TelegramTransportConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_mixin(
            SteamshipWidgetTransport(client=self.client, agent_service=self, agent=None)
        )

        self.add_mixin(
            ExtendedTelegramTransport(
                client=self.client, config=self.config, agent_service=self, agent=None,
                set_payment_plan=self.set_payment_plan
            )
        )
        self.usage = UsageTracker(self.client, n_free_messages=self.config.n_free_messages)

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return TelegramTransportConfig

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

    def check_usage(self, chat_id: str, context: AgentContext) -> bool:
        if not self.usage.exists(chat_id):
            self.usage.add_user(chat_id)
        if self.usage.usage_exceeded(chat_id):
            self.send_messages(context, [
                Block(text="🔴 You've used up all your message credits"),
                Block(
                    text="Buy message credits to continue chatting."
                         ""
                         "Tap the button:"
                ),
            ])
            self.send_invoice(chat_id)
            return False
        return True

    def respond(
            self, incoming_message: Block, chat_id: str, context: AgentContext
    ) -> List[Block]:

        if incoming_message.text == "/balance":
            usage_entry = self.usage.get_usage(chat_id)
            return [Block(text=f"You have {usage_entry.message_limit - usage_entry.message_count} messages left. "
                               f"\n\nType /buy if you want to buy message credits.")]

        if not self.check_usage(chat_id, context):
            return []

        if incoming_message.text == "/new":
            self.get_memory(self.client, chat_id).chat_memory.clear()
            return [Block(text="New conversation started.")]

        conversation = self.get_agent(
            client=context.client,
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

        self.usage.increase_message_count(chat_id)
        return [
            Block.get(self.client, _id=response)
            if is_uuid(response)
            else Block(text=response)
            for response in response_messages
        ]

    def send_messages(self, context: AgentContext, output_messages: List[Block]):
        for func in context.emit_funcs:
            logging.info(f"Emitting via function: {func.__name__}")
            func(output_messages, context.metadata)

    def run_agent(self, agent: Agent, context: AgentContext):
        chat_id = context.metadata.get("chat_id")

        incoming_message = context.chat_history.last_user_message
        output_messages = self.respond(incoming_message, chat_id or incoming_message.chat_id, context)
        self.send_messages(context, output_messages)

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
