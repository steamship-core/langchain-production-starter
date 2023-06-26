from typing import Optional

from pydantic import BaseModel
from steamship import Steamship
from steamship.utils.kv_store import KeyValueStore


class UsageEntry(BaseModel):
    message_count: int = 0
    message_limit: int = 0


class UsageTracker:
    kv_store: KeyValueStore

    def __init__(self, client: Steamship):
        self.kv_store = KeyValueStore(client, store_identifier="usage_tracking")

    def _get_usage(self, chat_id) -> UsageEntry:
        if not self.exists(chat_id):
            self._set_usage(chat_id, UsageEntry())
        return UsageEntry.parse_obj(self.kv_store.get(chat_id))

    def _set_usage(self, chat_id, usage: UsageEntry) -> None:
        self.kv_store.set(chat_id, usage.dict())

    def usage_exceeded(self, chat_id: str):
        usage_entry = self.kv_store.get(chat_id)
        return usage_entry["message_count"] >= usage_entry["message_limit"]

    def add_user(self, chat_id: str):
        if not self.kv_store.get(chat_id):
            self._set_usage(chat_id, UsageEntry())

    def exists(self, chat_id: str):
        return self.kv_store.get(chat_id) is not None

    def increase_message_count(self, chat_id: str, n_messages: Optional[int] = 1):
        usage_entry = self._get_usage(chat_id)
        usage_entry.message_count += n_messages
        self._set_usage(chat_id, usage_entry)
        return usage_entry.message_count

    def increase_message_limit(self, chat_id: str, n_messages: int):
        usage_entry = self._get_usage(chat_id)
        usage_entry.message_limit += n_messages
        self._set_usage(chat_id, usage_entry)
