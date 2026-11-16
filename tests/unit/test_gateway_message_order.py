"""The gateway hook that lets a resumed session reach the engine."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from typing import Any

import pytest
import yaml
from helm import CHART, FULLNAME, one, render

_HOOK = CHART / "gateway" / "message_order.py"


@pytest.fixture
def hook(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the chart's hook module against a stand-in for litellm's base class."""

    # litellm is not a dependency of this project; the hook only subclasses
    # its CustomLogger, so an empty class with that name is enough to import it.
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = type("CustomLogger", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    monkeypatch.setitem(sys.modules, "litellm.integrations", integrations)
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger", custom_logger)
    spec = importlib.util.spec_from_file_location("message_order", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _message(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def test_instructions_sent_mid_conversation_move_to_the_front_in_order(
    hook: types.ModuleType,
) -> None:
    messages = [
        _message("system", "Be terse."),
        _message("user", "hi"),
        _message("assistant", "hello"),
        _message("developer", "Answer in French."),
        _message("tool", "{}"),
        _message("system", "Cite sources."),
        _message("user", "how are you"),
    ]

    hoisted = hook.hoist_instructions(messages)

    assert [m["role"] for m in hoisted] == [
        "system",
        "developer",
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert [m["content"] for m in hoisted[:3]] == [
        "Be terse.",
        "Answer in French.",
        "Cite sources.",
    ]


def test_a_conversation_whose_instructions_already_lead_is_passed_through(
    hook: types.ModuleType,
) -> None:
    messages = [_message("developer", "Be terse."), _message("user", "hi")]

    assert hook.hoist_instructions(messages) is messages


@pytest.mark.parametrize("call_type", ["completion", "acompletion"])
def test_the_hook_rewrites_chat_completions(hook: types.ModuleType, call_type: str) -> None:
    data = {"messages": [_message("user", "hi"), _message("system", "Be terse.")]}

    result = asyncio.run(hook.instance.async_pre_call_hook(None, None, data, call_type))

    assert [m["role"] for m in result["messages"]] == ["system", "user"]


def test_the_hook_rewrites_the_responses_route_too(hook: types.ModuleType) -> None:
    """Codex talks the Responses API, whose conversation travels in ``input``."""

    data = {
        "instructions": "You are terse.",
        "input": [
            _message("user", "Hello"),
            _message("assistant", "Hey!"),
            _message("developer", "Answer only in French."),
            {"type": "function_call_output", "call_id": "c1", "output": "{}"},
            _message("user", "How are you?"),
        ],
    }

    result = asyncio.run(hook.instance.async_pre_call_hook(None, None, data, "aresponses"))

    assert [item.get("role") or item["type"] for item in result["input"]] == [
        "developer",
        "user",
        "assistant",
        "function_call_output",
        "user",
    ]
    assert result["instructions"] == "You are terse."


def test_a_cut_off_tool_call_is_kept_readable_on_both_routes(hook: types.ModuleType) -> None:
    """An engine restart mid-answer leaves the harness with a prefix of the arguments."""

    cut_off = '{"cmd": "ls -la /very/long'
    responses = {
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": cut_off},
            {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": '{"a": 1}'},
            {"type": "function_call", "call_id": "c3", "name": "shell", "arguments": ""},
        ]
    }
    chat: dict[str, Any] = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "shell", "arguments": cut_off}}],
            }
        ]
    }

    asyncio.run(hook.instance.async_pre_call_hook(None, None, responses, "aresponses"))
    asyncio.run(hook.instance.async_pre_call_hook(None, None, chat, "acompletion"))

    wrapped = json.dumps({"partial_arguments": cut_off})
    assert [item["arguments"] for item in responses["input"]] == [wrapped, '{"a": 1}', ""]
    assert chat["messages"][0]["tool_calls"][0]["function"]["arguments"] == wrapped


def test_a_bare_string_input_is_left_alone(hook: types.ModuleType) -> None:
    data = {"input": "Hello"}

    assert asyncio.run(hook.instance.async_pre_call_hook(None, None, data, "aresponses")) == data


def test_the_hook_leaves_other_calls_alone(hook: types.ModuleType) -> None:
    data = {"input": "hi", "messages": [_message("user", "hi"), _message("system", "x")]}

    result = asyncio.run(hook.instance.async_pre_call_hook(None, None, data, "embeddings"))

    assert [m["role"] for m in result["messages"]] == ["user", "system"]


def test_the_chart_ships_the_hook_beside_the_proxy_config() -> None:
    """The proxy loads a callback module from its config file's directory."""

    config_map = one(render(), "ConfigMap", f"{FULLNAME}-litellm")
    assert config_map["data"]["message_order.py"] == _HOOK.read_text(encoding="utf-8")
    config = yaml.safe_load(config_map["data"]["config.yaml"])
    assert config["litellm_settings"]["callbacks"][0] == "message_order.instance"
