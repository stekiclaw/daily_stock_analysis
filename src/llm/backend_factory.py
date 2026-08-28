# -*- coding: utf-8 -*-
"""Generation backend factory."""

from __future__ import annotations

from typing import Any, Optional

from src.llm.backend_registry import (
    CODEX_OAUTH_BACKEND_ID,
    LOCAL_CLI_GENERATION_BACKEND_IDS,
    LITELLM_BACKEND_ID,
)
from src.llm.generation_backend import GenerationBackend, GenerationError, GenerationErrorCode
from src.llm.litellm_backend import LiteLLMCallable, LiteLLMGenerationBackend
from src.llm.local_cli_backend import LocalCliGenerationBackend


def create_generation_backend(
    backend_id: str,
    *,
    config: Any,
    litellm_completion_callable: Optional[LiteLLMCallable] = None,
) -> GenerationBackend:
    """Create the configured generation backend."""

    normalized = (backend_id or "").strip().lower()
    if normalized == LITELLM_BACKEND_ID:
        if litellm_completion_callable is None:
            raise GenerationError(
                error_code=GenerationErrorCode.BACKEND_NOT_CONFIGURED,
                stage="configuration",
                retryable=False,
                fallbackable=False,
                backend=LITELLM_BACKEND_ID,
                provider=LITELLM_BACKEND_ID,
                details={"reason": "missing_litellm_completion_callable"},
            )
        return LiteLLMGenerationBackend(litellm_completion_callable)
    if normalized in LOCAL_CLI_GENERATION_BACKEND_IDS:
        return LocalCliGenerationBackend(config, preset_id=normalized)
    if normalized == CODEX_OAUTH_BACKEND_ID:
        # Imported lazily so the LiteLLM-only path does not pay for it.
        from src.llm.codex_oauth_backend import CodexOAuthGenerationBackend

        return CodexOAuthGenerationBackend(config)

    raise GenerationError(
        error_code=GenerationErrorCode.BACKEND_NOT_CONFIGURED,
        stage="configuration",
        retryable=False,
        fallbackable=False,
        backend=normalized or "unknown",
        provider=normalized or "unknown",
        details={"reason": "unknown_backend", "requested_backend": normalized},
    )
