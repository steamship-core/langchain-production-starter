import logging
import tempfile
from typing import Any, Dict, List, Optional, Callable

import requests
from steamship import Block, Steamship, SteamshipError
from steamship.agents.mixins.transports.telegram import TelegramTransportConfig
from steamship.agents.mixins.transports.transport import Transport
from steamship.agents.schema import Agent, AgentContext, EmitFunc, Metadata
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import Config, InvocableResponse, InvocationContext, post


class ExtendedTelegramTransport(Transport):
    api_root: str
    bot_token: str
    agent: Agent
    agent_service: AgentService
    set_payment_plan: Callable

    @post("telegram_respond", public=True)
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
            if incoming_message is not None:
                context = AgentContext.get_or_create(self.client, context_keys={"chat_id": chat_id})
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

    def __init__(
            self,
            client: Steamship,
            config: TelegramTransportConfig,
            agent_service: AgentService,
            agent: Agent,
            set_payment_plan: Callable
    ):
        super().__init__(client=client)
        self.api_root = f"{config.api_base}{config.bot_token}"
        self.bot_token = config.bot_token
        self.agent = agent
        self.agent_service = agent_service
        self.set_payment_plan = set_payment_plan

    def instance_init(self, config: Config, invocation_context: InvocationContext):
        webhook_url = invocation_context.invocable_url + "telegram_respond"

        logging.info(
            f"Setting Telegram webhook URL: {webhook_url}. Post is to {self.api_root}/setWebhook"
        )

        response = requests.get(
            f"{self.api_root}/setWebhook",
            params={
                "url": webhook_url,
                "allowed_updates": ["message"],
                "drop_pending_updates": True,
            },
        )

        if not response.ok:
            raise SteamshipError(
                f"Could not set webhook for bot. Webhook URL was {webhook_url}. Telegram response message: {response.text}"
            )

    @post("telegram_webhook_info")
    def telegram_webhook_info(self) -> dict:
        return requests.get(self.api_root + "/getWebhookInfo").json()

    @post("telegram_disconnect_webhook")
    def telegram_disconnect_webhook(self, *args, **kwargs):
        """Unsubscribe from Telegram updates."""
        requests.get(f"{self.api_root}/deleteWebhook")

    def _send(self, blocks: [Block], metadata: Metadata):
        """Send a response to the Telegram chat."""
        for block in blocks:
            chat_id = block.chat_id
            if block.is_text() or block.text:
                params = {"chat_id": int(chat_id), "text": block.text}
                requests.get(f"{self.api_root}/sendMessage", params=params)
            elif block.is_image() or block.is_audio() or block.is_video():
                if block.is_image():
                    suffix = "sendPhoto"
                    key = "photo"
                elif block.is_audio():
                    suffix = "sendAudio"
                    key = "audio"
                elif block.is_video():
                    suffix = "sendVideo"
                    key = "video"

                _bytes = block.raw()
                with tempfile.TemporaryFile(mode="r+b") as temp_file:
                    temp_file.write(_bytes)
                    temp_file.seek(0)
                    resp = requests.post(
                        url=f"{self.api_root}/{suffix}?chat_id={chat_id}",
                        files={key: temp_file},
                    )
                    if resp.status_code != 200:
                        logging.error(f"Error sending message: {resp.text} [{resp.status_code}]")
                        raise SteamshipError(
                            f"Message not sent to chat {chat_id} successfully: {resp.text}"
                        )
            else:
                logging.error(
                    f"Telegram transport unable to send a block of MimeType {block.mime_type}"
                )

    def _get_file(self, file_id: str) -> Dict[str, Any]:
        return requests.get(f"{self.api_root}/getFile", params={"file_id": file_id}).json()[
            "result"
        ]

    def _get_file_url(self, file_id: str) -> str:
        return f"https://api.telegram.org/file/bot{self.bot_token}/{self._get_file(file_id)['file_path']}"

    def _download_file(self, file_id: str):
        result = requests.get(self._get_file_url(file_id))
        if result.status_code != 200:
            raise Exception("Download file", result)

        return result.content

    def _parse_inbound(self, payload: dict, context: Optional[dict] = None) -> Optional[Block]:
        """Parses an inbound Telegram message."""

        chat = payload.get("chat")
        if chat is None:
            raise SteamshipError(f"No `chat` found in Telegram message: {payload}")

        chat_id = chat.get("id")
        if chat_id is None:
            raise SteamshipError(f"No 'chat_id' found in Telegram message: {payload}")

        if not isinstance(chat_id, int):
            raise SteamshipError(
                f"Bad 'chat_id' found in Telegram message: ({chat_id}). Should have been an int."
            )

        message_id = payload.get("message_id")
        if message_id is None:
            raise SteamshipError(f"No 'message_id' found in Telegram message: {payload}")

        if not isinstance(message_id, int):
            raise SteamshipError(
                f"Bad 'message_id' found in Telegram message: ({message_id}). Should have been an int"
            )

        if video_or_voice := (payload.get("voice") or payload.get("video_note")):
            file_id = video_or_voice.get("file_id")
            file_url = self._get_file_url(file_id)
            block = Block(
                text=payload.get("text"),
                url=file_url,
            )
            block.set_chat_id(str(chat_id))
            block.set_message_id(str(message_id))
            return block

        # Some incoming messages (like the group join message) don't have message text.
        # Rather than throw an error, we just don't return a Block.
        message_text = payload.get("text")
        if message_text is not None:
            result = Block(text=message_text)
            result.set_chat_id(str(chat_id))
            result.set_message_id(str(message_id))
            return result
        else:
            return None

    def build_emit_func(self, chat_id: str) -> EmitFunc:
        def new_emit_func(blocks: List[Block], metadata: Metadata):
            for block in blocks:
                block.set_chat_id(chat_id)
            return self.send(blocks, metadata)

        return new_emit_func
