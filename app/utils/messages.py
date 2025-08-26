from typing import List
from app.validators.api_request import Message

def enforce_alternating_roles(messages: List[Message]) -> List[Message]:
    """
    Enforce strict alternation between 'user' and 'assistant' roles
    in a messages array (like OpenAI's Chat API).

    Rules:
    - System messages allowed only at the start (preserved as-is).
    - After system(s), conversation must strictly alternate user/assistant.
    - If not, dummy turns are injected.
    - First non-system must be a user (otherwise inject dummy user).
    - Last message must be a user (otherwise append dummy user).
    """

    if not messages:
        return [Message(role="user", content="")]

    normalized: List[Message] = []
    n = len(messages)
    i = 0

    # Preserve system messages at start
    while i < n and messages[i].role == "system":
        normalized.append(messages[i])
        i += 1

    # Ensure first non-system is user
    if i >= n or messages[i].role != "user":
        normalized.append(Message(role="user", content=""))
        last_role = "user"
    else:
        normalized.append(messages[i])
        last_role = "user"
        i += 1

    # Process rest of conversation
    while i < n:
        m = messages[i]
        role = m.role

        if role == last_role:
            # Inject opposite dummy
            dummy_role = "assistant" if role == "user" else "user"
            normalized.append(Message(role=dummy_role, content=""))

        normalized.append(m)
        last_role = role
        i += 1

    # Ensure last message is user
    if normalized[-1].role != "user":
        normalized.append(Message(role="user", content=""))

    return normalized
