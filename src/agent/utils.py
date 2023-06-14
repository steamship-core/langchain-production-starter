"""Define your LangChain chatbot."""
import re
import uuid

UUID_PATTERN = re.compile(
    r"([0-9A-Za-z]{8}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{12})"
)

MAX_FREE_MESSAGES = 10


def is_uuid(uuid_to_test: str, version: int = 4) -> bool:
    """Check a string to see if it is actually a UUID."""
    lowered = uuid_to_test.lower()
    try:
        return str(uuid.UUID(lowered, version=version)) == lowered
    except ValueError:
        return False
