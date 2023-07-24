from typing import Callable

import requests
from steamship import Steamship
from steamship.agents.mixins.transports.telegram import (
    TelegramTransportConfig,
    TelegramTransport,
)
from steamship.agents.schema import Agent
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import InvocableResponse, post, Config, InvocationContext


class ExtendedTelegramTransport(TelegramTransport):
    api_root: str
    bot_token: str
    agent: Agent
    agent_service: AgentService
    set_payment_plan: Callable

    def __init__(
        self,
        set_payment_plan: Callable,
        client: Steamship,
        config: TelegramTransportConfig,
        agent_service: AgentService,
        agent: Agent,
    ):
        super().__init__(client, config, agent_service, agent)
        self.set_payment_plan = set_payment_plan

    def instance_init(self, config: Config, invocation_context: InvocationContext):
        if config.bot_token:
            super().instance_init(config=config, invocation_context=invocation_context)

    @post("telegram_respond", public=True, permit_overwrite_of_existing=True)
    def telegram_respond(self, **kwargs) -> InvocableResponse[str]:
        """Endpoint implementing the Telegram WebHook contract. This is a PUBLIC endpoint since Telegram cannot pass a Bearer token."""

        if "pre_checkout_query" in kwargs:
            pre_checkout_query = kwargs["pre_checkout_query"]
            self.set_payment_plan(pre_checkout_query)
            requests.post(
                f"{self.api_root}/answerPreCheckoutQuery",
                json={
                    "pre_checkout_query_id": pre_checkout_query["id"],
                    "ok": True,
                },
            )
            return InvocableResponse(string="OK")

        return super().telegram_respond(**kwargs)
