# -*- coding: utf-8 -*-
"""Message/tool translation between the Agent loop and the Responses endpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.codex_oauth_adapter import (  # noqa: E402
    CodexOAuthToolAdapter,
    _tool_declarations_to_responses,
    messages_to_responses_input,
)
from src.llm import codex_oauth  # noqa: E402


def _config(**overrides):
    base = {
        "codex_oauth_auth_file": "data/codex_oauth/auth.json",
        "codex_oauth_model": "gpt-5.6-sol",
        "codex_oauth_reasoning_effort": "medium",
        "agent_codex_oauth_model": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --- message translation ----------------------------------------------------


def test_system_messages_are_hoisted_into_instructions() -> None:
    items, instructions = messages_to_responses_input(
        [
            {"role": "system", "content": "你是股票助手"},
            {"role": "system", "content": "只用中文"},
            {"role": "user", "content": "NVDA 多少钱"},
        ]
    )

    # The endpoint rejects a null instructions field and treats it as the
    # standing directive, so system turns must not stay in the item list.
    assert instructions == "你是股票助手\n\n只用中文"
    assert [item["type"] for item in items] == ["message"]
    assert items[0]["role"] == "user"
    assert items[0]["content"] == [{"type": "input_text", "text": "NVDA 多少钱"}]


def test_assistant_tool_calls_become_function_call_items() -> None:
    items, _ = messages_to_responses_input(
        [
            {"role": "user", "content": "查一下"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "name": "get_realtime_quote", "arguments": {"stock_code": "NVDA"}}
                ],
            },
            {
                "role": "tool",
                "name": "get_realtime_quote",
                "tool_call_id": "call_1",
                "content": '{"price": 227.98}',
            },
        ]
    )

    assert [item["type"] for item in items] == ["message", "function_call", "function_call_output"]
    call = items[1]
    assert call["call_id"] == "call_1"
    assert call["name"] == "get_realtime_quote"
    # Arguments travel as a JSON string even though the loop holds a dict.
    assert json.loads(call["arguments"]) == {"stock_code": "NVDA"}
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"price": 227.98}',
    }


def test_assistant_text_and_tool_calls_both_survive() -> None:
    items, _ = messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "让我查一下",
                "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}],
            }
        ]
    )

    assert [item["type"] for item in items] == ["message", "function_call"]
    assert items[0]["content"] == [{"type": "output_text", "text": "让我查一下"}]


def test_block_style_content_is_flattened() -> None:
    items, _ = messages_to_responses_input(
        [{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}]
    )
    assert items[0]["content"][0]["text"] == "AB"


def test_tool_declarations_are_flattened_for_responses() -> None:
    converted = _tool_declarations_to_responses(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_quote",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {"type": "function"},  # malformed, must be dropped
        ]
    )

    # Responses wants the fields at the top level, not nested under "function".
    assert converted == [
        {
            "type": "function",
            "name": "get_quote",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


# --- adapter behaviour ------------------------------------------------------


def _patched_call(monkeypatch, *, text="", tool_calls=None, capture=None):
    def fake(credential, input_items, **kwargs):
        if capture is not None:
            capture["input"] = input_items
            capture["kwargs"] = kwargs
        return text, tool_calls or [], {"total_tokens": 42}, {}

    monkeypatch.setattr(codex_oauth, "generate_with_tools", fake)
    monkeypatch.setattr(codex_oauth, "load_credential", lambda path: {"access_token": "t"})
    monkeypatch.setattr(codex_oauth, "ensure_fresh_credential", lambda cred, path: cred)


def test_tool_calls_are_normalized_for_the_loop(monkeypatch) -> None:
    _patched_call(
        monkeypatch,
        tool_calls=[
            {"call_id": "c1", "name": "get_realtime_quote", "arguments": '{"stock_code": "NVDA"}'}
        ],
    )
    response = CodexOAuthToolAdapter(_config()).call_with_tools([], [])

    assert response.provider == "codex_oauth"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "c1"
    assert response.tool_calls[0].name == "get_realtime_quote"
    # The loop indexes arguments as a dict, so the JSON string must be parsed.
    assert response.tool_calls[0].arguments == {"stock_code": "NVDA"}
    assert response.usage == {"total_tokens": 42}


def test_unparseable_tool_arguments_degrade_to_empty_dict(monkeypatch) -> None:
    _patched_call(monkeypatch, tool_calls=[{"call_id": "c1", "name": "t", "arguments": "{not json"}])
    response = CodexOAuthToolAdapter(_config()).call_with_tools([], [])
    assert response.tool_calls[0].arguments == {}


def test_agent_model_overrides_the_generation_model(monkeypatch) -> None:
    capture: dict = {}
    _patched_call(monkeypatch, text="ok", capture=capture)

    adapter = CodexOAuthToolAdapter(_config(agent_codex_oauth_model="gpt-5.6-terra"))
    response = adapter.call_with_tools([], [])

    assert capture["kwargs"]["model"] == "gpt-5.6-terra"
    assert response.model == "gpt-5.6-terra"


def test_agent_model_falls_back_to_the_generation_model(monkeypatch) -> None:
    capture: dict = {}
    _patched_call(monkeypatch, text="ok", capture=capture)

    CodexOAuthToolAdapter(_config()).call_with_tools([], [])
    assert capture["kwargs"]["model"] == "gpt-5.6-sol"


def test_backend_errors_surface_as_an_error_response(monkeypatch) -> None:
    monkeypatch.setattr(codex_oauth, "load_credential", lambda path: {"access_token": "t"})
    monkeypatch.setattr(codex_oauth, "ensure_fresh_credential", lambda cred, path: cred)

    def boom(*args, **kwargs):
        raise codex_oauth.CodexOAuthError("login_required", "未找到凭证")

    monkeypatch.setattr(codex_oauth, "generate_with_tools", boom)
    response = CodexOAuthToolAdapter(_config()).call_with_tools([], [])

    # The loop keys off provider == "error" rather than exceptions.
    assert response.provider == "error"
    assert "未找到凭证" in (response.content or "")
    assert response.tool_calls == []


def test_empty_answer_is_not_an_error_when_tools_were_called(monkeypatch) -> None:
    _patched_call(monkeypatch, text="", tool_calls=[{"call_id": "c", "name": "t", "arguments": "{}"}])
    response = CodexOAuthToolAdapter(_config()).call_with_tools([], [])
    assert response.content is None
    assert response.tool_calls
