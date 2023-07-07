import logging
import re
import uuid
from abc import abstractmethod
from typing import List, Optional, Type

import requests
from langchain.agents import Tool, AgentExecutor
from langchain.memory.chat_memory import BaseChatMemory
from pydantic import Field
from steamship import Block
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.schema import (
    AgentContext,
    Metadata,
    Agent,
)
from steamship.agents.service.agent_service import AgentService
from steamship.cli.cli import cli
from steamship.invocable import post, Config
from steamship.utils.kv_store import KeyValueStore

from agent.telegram import ExtendedTelegramTransport
from agent.usage_tracking import UsageTracker
from agent.utils import is_uuid, UUID_PATTERN


class TelegramTransportConfig(Config):
    bot_token: Optional[str] = Field("", description="Your telegram bot token.\nLearn how to create one here: "
                                                     "https://github.com/steamship-packages/langchain-agent-production-starter/blob/main/docs/register-telegram-bot.md")
    payment_provider_token: Optional[str] = Field(
        "", description="Optional Payment provider token, obtained via @BotFather"
    )
    n_free_messages: Optional[int] = Field(
        -1, description="Number of free messages assigned to new users."
    )
    api_base: str = Field(
        "https://api.telegram.org/bot", description="The root API for Telegram"
    )


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
                "prices": [
                    {
                        "label": "🍏 50 messages",
                        "amount": 599,
                    }
                ],
                "provider_token": self.config.payment_provider_token,
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
        self.store = KeyValueStore(self.client, store_identifier="config")
        bot_token = self.store.get("bot_token")
        if bot_token:
            bot_token = bot_token.get("token")
        self.add_mixin(
            SteamshipWidgetTransport(client=self.client, agent_service=self, agent=None)
        )
        self.config.bot_token = bot_token
        self.add_mixin(
            ExtendedTelegramTransport(
                client=self.client,
                config=self.config,
                agent_service=self,
                agent=None,
                set_payment_plan=self.set_payment_plan,
            ),
            permit_overwrite_of_existing_methods=True,
        )

        self.usage = UsageTracker(
            self.client, n_free_messages=self.config.n_free_messages
        )

    @post("connect_telegram", public=True)
    def connect_telegram(self, bot_token: str):
        webhook_url = self.context.invocable_url + "telegram_respond"
        self.store.set("bot_token", {"token" : bot_token})

        api_root = f"https://api.telegram.org/bot{bot_token}"

        logging.info(
            f"Setting Telegram webhook URL: {webhook_url}. Post is to {api_root}/setWebhook"
        )

        response = requests.get(
            f"{api_root}/setWebhook",
            params={
                "url": webhook_url,
                "allowed_updates": ["message"],
                "drop_pending_updates": True,
            },
        )

        if not response.ok:
            return (
                f"Could not set webhook for bot. Webhook URL was {webhook_url}. Telegram response message: {response.text}"
            )
        else:
            return "OK"

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return TelegramTransportConfig

    @abstractmethod
    def get_agent(self, chat_id: str) -> AgentExecutor:
        raise NotImplementedError()

    @abstractmethod
    def get_memory(self, chat_id: str) -> BaseChatMemory:
        raise NotImplementedError()

    @abstractmethod
    def get_tools(self, chat_id: str) -> List[Tool]:
        raise NotImplementedError()

    def voice_tool(self) -> Optional[Tool]:
        return None

    def check_usage(self, chat_id: str, context: AgentContext) -> bool:
        if not self.usage.exists(chat_id):
            self.usage.add_user(chat_id)
        if self.usage.usage_exceeded(chat_id):
            self.send_messages(
                context,
                [
                    Block(text="🔴 You've used up all your message credits"),
                    Block(
                        text="Buy message credits to continue chatting."
                             "\n\n"
                             "Tap the button:"
                    ),
                ],
            )
            if self.config.payment_provider_token:
                self.send_invoice(chat_id)
            else:
                self.send_messages(
                    context,
                    [
                        Block(text="😭 Payments not set up for this bot."),
                    ],
                )
            return False
        return True

    def respond(
            self, incoming_message: Block, chat_id: str, context: AgentContext
    ) -> List[Block]:

        if incoming_message.text == "/balance":
            usage_entry = self.usage.get_usage(chat_id)
            return [
                Block(
                    text=f"You have {usage_entry.message_limit - usage_entry.message_count} messages left. "
                         f"\n\nType /buy if you want to buy message credits."
                )
            ]

        if not self.check_usage(chat_id, context):
            return []

        if incoming_message.text == "/new":
            self.get_memory(chat_id).chat_memory.clear()
            return [Block(text="New conversation started.")]

        conversation = self.get_agent(
            chat_id,
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
        output_messages = self.respond(
            incoming_message, chat_id or incoming_message.chat_id, context
        )
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


cli
