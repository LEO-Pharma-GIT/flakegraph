"""Put every instruction message where the engine can read it.

The OpenAI chat contract lets a caller place ``system`` and ``developer``
messages anywhere in a conversation, and coding harnesses do exactly that when
they resume a session: the saved history carries their instruction updates
between turns. The engine's chat template merges instruction messages only
while they lead the conversation and refuses one that comes later, so a
resumed session fails with "System message must be at the beginning" while a
fresh one works.

The gateway is what advertises the OpenAI contract, so it is the gateway that
reconciles the two: instruction messages move to the front in the order they
were sent, and everything else keeps its order. Nothing is reworded, and a
conversation whose instructions already lead is passed through untouched.

The same contract lets a tool call carry any string as its arguments, while the
engine parses them as JSON to render its template. A call whose stream was cut
off (an engine restart mid-answer) leaves the harness holding the prefix it had
received, and every later turn of that conversation is refused with
"Unterminated string". Such arguments are kept, wrapped in a JSON object that
says what they are, so the conversation can go on.
"""

from __future__ import annotations

import json
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

INSTRUCTION_ROLES = frozenset({"system", "developer"})
# The proxy names the route it is serving: chat completions carry the
# conversation in ``messages``, the Responses API carries it in ``input``
# (which may also be a bare string), and both admit instruction items anywhere.
CONVERSATION_FIELDS = {"completion": "messages", "acompletion": "messages", "aresponses": "input"}


def hoist_instructions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the messages with every instruction ahead of the conversation."""

    instructions = [m for m in messages if m.get("role") in INSTRUCTION_ROLES]
    if messages[: len(instructions)] == instructions:
        return messages
    return instructions + [m for m in messages if m.get("role") not in INSTRUCTION_ROLES]


def readable_arguments(arguments: object) -> object:
    """Return tool-call arguments the engine can parse, keeping a cut-off prefix."""

    if not isinstance(arguments, str) or not arguments:
        return arguments
    try:
        json.loads(arguments)
    except ValueError:
        return json.dumps({"partial_arguments": arguments})
    return arguments


def repair_tool_calls(messages: list[dict[str, Any]]) -> None:
    """Make every tool call's arguments parseable, on either route's shape."""

    for message in messages:
        if message.get("type") == "function_call":
            message["arguments"] = readable_arguments(message.get("arguments"))
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                function["arguments"] = readable_arguments(function.get("arguments"))


class MessageOrder(CustomLogger):
    """Repair a conversation before it reaches a model."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Repair the conversation on the conversational routes; leave other calls alone."""

        field = CONVERSATION_FIELDS.get(call_type)
        if field is None:
            return data
        conversation = data.get(field)
        if isinstance(conversation, list) and all(isinstance(item, dict) for item in conversation):
            data[field] = hoist_instructions(conversation)
            repair_tool_calls(data[field])
        return data


instance = MessageOrder()
