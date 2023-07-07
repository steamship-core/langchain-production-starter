import logging
from typing import Callable

import requests
from steamship import Steamship, SteamshipError
from steamship.agents.mixins.transports.telegram import (
    TelegramTransportConfig,
    TelegramTransport,
)
from steamship.agents.schema import Agent, AgentContext
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
        pass



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

        message = kwargs.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        try:
            incoming_message = self.parse_inbound(message)
            if incoming_message is not None and incoming_message.text is not None:
                context = AgentContext.get_or_create(
                    self.client, context_keys={"chat_id": chat_id}
                )
                context.chat_history.append_user_message(
                    text=incoming_message.text, tags=incoming_message.tags
                )
                context.emit_funcs = [self.build_emit_func(chat_id=chat_id)]

                response = self.agent_service.run_agent(self.agent, context)
                if response is not None:
                    self.send(response)
                else:
                    # Do nothing here; this could be a message we intentionally don't want to respond to (ex. an image or file upload)
                    pass
            else:
                # Do nothing here; this could be a message we intentionally don't want to respond to (ex. an image or file upload)
                pass
        except Exception as e:
            response = self.response_for_exception(e, chat_id=chat_id)

            if chat_id is not None:
                self.send([response])
        # Even if we do nothing, make sure we return ok
        return InvocableResponse(string="OK")
