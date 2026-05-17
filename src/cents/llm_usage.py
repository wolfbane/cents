"""Best-effort LLM usage recording.

Every call site that hits Anthropic's API funnels through `record_llm_usage`.
Failures (DB locked, table missing, etc.) are logged at debug level and
swallowed — the LLM call has already succeeded, and bookkeeping must never
break user-facing flows.
"""

from __future__ import annotations

import logging
from typing import Any

from cents.db import LLMUsageRepository
from cents.models import LLMUsage

logger = logging.getLogger(__name__)


def record_llm_usage(
    response: Any,
    agent: str,
    operation: str,
    context: str | None = None,
) -> None:
    """Persist a usage record from an Anthropic Message response.

    `response` is expected to expose `.model` and `.usage.{input_tokens,
    output_tokens, cache_read_input_tokens, cache_creation_input_tokens}`,
    matching the shape of `anthropic.types.Message`. Cache fields default to
    None for non-cached calls and are coerced to 0.

    Best-effort: catches all exceptions and logs them.
    """
    try:
        usage = response.usage
        row = LLMUsage(
            model=response.model or "",
            agent=agent,
            operation=operation,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
            context=context,
        )
        LLMUsageRepository().create(row)
    except Exception as e:  # noqa: BLE001 — bookkeeping must never raise
        logger.debug("record_llm_usage failed (agent=%s op=%s): %s", agent, operation, e)
