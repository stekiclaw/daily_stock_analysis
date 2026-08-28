# -*- coding: utf-8 -*-
"""LLM adapter that runs the Agent tool loop on a ChatGPT/Codex subscription.

``run_agent_loop`` talks to exactly one seam - ``call_with_tools(messages,
tools)`` returning an :class:`LLMResponse`. This adapter implements that seam
against the ChatGPT Responses endpoint, so the DSA-owned loop (tool execution,
parallel calls, timeouts, progress events, cancellation) is reused unchanged.

The two sides speak different shapes, and translating between them is all this
module does:

    OpenAI chat messages  <->  Responses input items
    role=system               instructions (hoisted out of the item list)
    role=user                 message item with input_text
    role=assistant            message item with output_text
    assistant.tool_calls      function_call items (call_id, name, arguments)
    role=tool                 function_call_output items (call_id, output)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.agent.llm_adapter import LLMResponse, ToolCall
from src.llm import codex_oauth

logger = logging.getLogger(__name__)

BACKEND_ID = "codex_oauth"


def _tool_declarations_to_responses(tools: Optional[List[dict]]) -> List[Dict[str, Any]]:
    """Flatten OpenAI ``{"function": {...}}`` declarations into Responses tools."""
    converted: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        converted.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", "") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def _stringify(content: Any) -> str:
    """Flatten a chat ``content`` field into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal-style blocks: keep the text parts, drop the rest.
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def messages_to_responses_input(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """Convert chat messages into ``(input_items, instructions)``.

    System messages are hoisted into ``instructions`` because the Responses
    endpoint rejects a null one and treats it as the standing directive.
    """
    items: List[Dict[str, Any]] = []
    instructions: List[str] = []

    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            instructions.append(_stringify(message.get("content")))
            continue

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": _stringify(message.get("content")),
                }
            )
            continue

        if role == "assistant":
            text = _stringify(message.get("content"))
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in message.get("tool_calls") or []:
                arguments = call.get("arguments")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or ""),
                        "name": str(call.get("name") or ""),
                        # The wire format is a JSON string even though the loop
                        # keeps arguments as a dict.
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments or {}, ensure_ascii=False),
                    }
                )
            continue

        # Anything else (user, and unknown roles) is sent as user input.
        items.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": _stringify(message.get("content"))}],
            }
        )

    return items, "\n\n".join(part for part in instructions if part)


def _parse_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for raw in raw_calls:
        raw_args = raw.get("arguments")
        if isinstance(raw_args, dict):
            arguments = raw_args
        else:
            try:
                arguments = json.loads(raw_args or "{}")
            except (TypeError, json.JSONDecodeError):
                logger.warning(
                    "codex_oauth returned unparseable tool arguments for %s", raw.get("name")
                )
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(
                id=str(raw.get("call_id") or raw.get("id") or ""),
                name=str(raw.get("name") or ""),
                arguments=arguments,
            )
        )
    return calls


class CodexOAuthToolAdapter:
    """``LLMToolAdapter``-shaped adapter backed by the Codex OAuth credential."""

    def __init__(self, config: Any) -> None:
        self._config = config

    # --- configuration ------------------------------------------------------

    @property
    def _auth_file(self) -> str:
        return str(getattr(self._config, "codex_oauth_auth_file", "") or "")

    @property
    def _model(self) -> str:
        # The Agent may run a different model than report generation, so its own
        # setting wins and only falls back to the generation model.
        return str(
            getattr(self._config, "agent_codex_oauth_model", "")
            or getattr(self._config, "codex_oauth_model", "")
            or codex_oauth.DEFAULT_MODEL
        )

    @property
    def _effort(self) -> str:
        return codex_oauth.normalize_effort(
            getattr(self._config, "codex_oauth_reasoning_effort", None)
        )

    @property
    def is_available(self) -> bool:
        try:
            codex_oauth.load_credential(self._auth_file)
        except codex_oauth.CodexOAuthError:
            return False
        return True

    @property
    def primary_provider(self) -> str:
        return BACKEND_ID

    # --- the single seam the agent loop uses --------------------------------

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[dict],
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Send one turn and return the normalized response."""
        del provider  # kept for interface compatibility, as LiteLLM does
        model = self._model
        input_items, instructions = messages_to_responses_input(messages)

        try:
            credential = codex_oauth.ensure_fresh_credential(
                codex_oauth.load_credential(self._auth_file), self._auth_file
            )
            text, raw_calls, usage, _diagnostics = codex_oauth.generate_with_tools(
                credential,
                input_items,
                tools=_tool_declarations_to_responses(tools),
                model=model,
                effort=self._effort,
                instructions=instructions,
                timeout_seconds=int(timeout) if timeout and timeout > 0 else 300,
            )
        except codex_oauth.CodexOAuthError as exc:
            logger.error("codex_oauth agent call failed: %s", exc.message)
            return LLMResponse(
                content=f"[codex_oauth] {exc.message}",
                provider="error",
                model=model,
            )

        return LLMResponse(
            content=text or None,
            tool_calls=_parse_tool_calls(raw_calls),
            usage=usage,
            provider=BACKEND_ID,
            model=model,
        )

    def call_text(self, messages: List[Dict[str, Any]], **kwargs: Any) -> LLMResponse:
        """Text-only turn: same path with no tools declared."""
        return self.call_with_tools(messages, [], timeout=kwargs.get("timeout"))
