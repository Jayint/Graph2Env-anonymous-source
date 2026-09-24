from __future__ import annotations
import os
import random
import re
import sys
import time
from typing import Any, Callable, Optional

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    _RETRYABLE_EXC: tuple = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )
except Exception:  # pragma: no cover - openai missing/old: degrade to no transport retry
    APIStatusError = None  # type: ignore
    _RETRYABLE_EXC = ()

# Transient HTTP statuses worth retrying (timeouts / rate-limit / 5xx). 4xx
# (bad request, auth) are fatal and must surface, not be retried into a giveup.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_TRANSPORT_ATTEMPTS = int(os.getenv("LLM_MAX_TRANSPORT_RETRIES", "4"))
_MAX_ROUTE_404_ATTEMPTS = int(os.getenv("LLM_ROUTE_404_MAX_ATTEMPTS", "8"))
_TRANSPORT_BASE_DELAY = float(os.getenv("LLM_TRANSPORT_BASE_DELAY", "2.0"))
_TRANSPORT_MAX_DELAY = float(os.getenv("LLM_TRANSPORT_MAX_DELAY", "30.0"))
_ROUTE_404_MARKER = "not supported by any configured account in this group"

# Some OpenAI-compatible routes accept ``response_format=json_object`` only on
# a subset of their dynamic backends.  LinkAPI's free DeepSeek Flash route, for
# example, can select DFLASH speculative decoding, which rejects grammar-
# constrained decoding with HTTP 400.  Keep the capability decision in the
# shared request layer so the main ReAct Agent and its Diet reflector cannot
# silently diverge.
_PROMPT_JSON_MODELS = frozenset({"[free]deepseek-v4-flash"})


def json_response_format_kwargs(model: str) -> dict[str, Any]:
    """Return safe OpenAI request kwargs for the selected JSON contract.

    ``REACT_JSON_MODE`` is intentionally process-scoped so benchmark commands
    can opt out without modifying credentials or global Git/model settings:

    * ``auto`` (default): use server-enforced JSON except for known
      incompatible routes;
    * ``prompt``: omit ``response_format`` for every model and rely on the
      prompt plus host parsing/validation;
    * ``structured``: force server-enforced JSON for diagnostics/providers that
      support it.

    The prompt-only path does not weaken host authority: callers still require
    a single parsed JSON object before an action or compression is accepted.
    """
    mode = os.getenv("REACT_JSON_MODE", "auto").strip().lower()
    if mode not in {"auto", "prompt", "structured"}:
        raise ValueError(
            "REACT_JSON_MODE must be one of: auto, prompt, structured"
        )
    if mode == "prompt" or (
        mode == "auto" and str(model or "").strip().lower() in _PROMPT_JSON_MODELS
    ):
        return {}
    return {"response_format": {"type": "json_object"}}


def strip_reasoning_markup(text: str | None) -> str:
    """Remove ``<think>...</think>`` reasoning markup leaked into message content.

    Handles two cases produced by reasoning models (e.g. MiniMax M2.7 via
    OpenRouter):

    * **Complete blocks** – ``<think>...</think>`` anywhere in the text
      (including multiple occurrences, ``re.DOTALL``).
    * **Leading orphan fragment** – the opening ``<think>`` was placed in the
      separate ``reasoning`` field so only the tail (ending in ``</think>``)
      appears at the start of ``content``.  Strip from position 0 through the
      first ``</think>`` when no preceding ``<think>`` exists in that prefix.

    The function is idempotent; returns ``""`` for ``None`` or empty input;
    leaves clean text (no think markup) byte-identical.

    :param text: Raw content string, potentially containing think markup.
    :return: Cleaned string with all think markup removed and outer whitespace
        stripped.
    """
    if not text:
        return ""

    # Remove complete <think>...</think> blocks (non-greedy so each block is
    # matched independently; re.DOTALL).
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove a leading orphan fragment that ends at the first </think> when
    # there is no matching opening <think> before it.  Only strip when ALL of:
    #   1. No earlier <think> in the prefix (not a matched pair).
    #   2. The </think> tag is on its own line — the prefix ends with a newline
    #      or is entirely whitespace.  This prevents eating a real answer like
    #      "Answer with </think> rest" where </think> is mid-sentence.
    #   3. The prefix contains no Thought:/Action:/Final Answer: directive,
    #      preventing "Action: ls\n</think>\nmore" from losing the Action line.
    close_match = re.search(r"</think>", cleaned)
    if close_match:
        prefix = cleaned[: close_match.start()]
        _prefix_ends_at_line_start = not prefix or prefix[-1] == "\n" or not prefix.strip()
        _no_directive = not re.search(
            r"^\s*(Thought:|Action:|Final Answer:)", prefix, re.MULTILINE
        )
        if "<think>" not in prefix and _prefix_ends_at_line_start and _no_directive:
            cleaned = cleaned[close_match.end():]

    return cleaned.strip()


def response_text(response: Any) -> str:
    """Return the assistant message text from an OpenAI-compatible chat completion.

    Prefers the standard ``choices[0].message.content``. Reasoning models such as
    MiniMax via OpenRouter often return ``content=None``/empty and put the actual
    text in a separate ``reasoning`` field (either as a message attribute or under
    ``model_extra``); this helper falls back to that field so the parser still sees
    the action. Returns ``""`` if no usable text is found or the response shape is
    malformed.

    ``strip_reasoning_markup`` is applied to ``content`` before the non-empty
    check so that a content field consisting solely of a ``<think>`` fragment
    correctly falls through to the reasoning fallback.

    :param response: An OpenAI-compatible chat completion object (or any object
        that does not match the expected shape).
    :return: The non-empty assistant text, the reasoning fallback, or ``""``.
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError, KeyError):
        return ""

    content = getattr(message, "content", None)
    clean = strip_reasoning_markup(content)
    if clean:
        return clean

    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return strip_reasoning_markup(reasoning) or reasoning

    extra = getattr(message, "model_extra", None) or {}
    return extra.get("reasoning") or ""


_DEFAULT_RETRY_NUDGE = (
    "Your previous response was empty or invalid. "
    "Respond now in the exact required format."
)


def _is_retryable_transport_error(exc: BaseException) -> bool:
    """True for a transient transport failure worth retrying (timeout / connection
    drop / rate-limit / 5xx). 4xx (bad request, auth) and unknown errors are fatal."""
    if _RETRYABLE_EXC and isinstance(exc, _RETRYABLE_EXC):
        return True
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) in _RETRYABLE_STATUS
    return False


def _is_retryable_model_route_error(exc: BaseException) -> bool:
    """Return True only for the provider's transient account-group 404."""
    if getattr(exc, "status_code", None) != 404:
        return False
    fragments = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        fragments.append(str(body))
    return _ROUTE_404_MARKER in " ".join(fragments).lower()


def _sleep_backoff(attempt: int, base: float, cap: float) -> None:
    """Exponential backoff with 50-100% jitter (own function so tests can stub it)."""
    delay = min(cap, base * (2 ** attempt))
    time.sleep(delay * (0.5 + random.random() * 0.5))


def _create_with_backoff(
    client, model, messages, kwargs, *, attempts, base, cap, on_attempt=None,
    route_404_attempts=None,
):
    """Call ``chat.completions.create``, retrying transient transport failures with
    exponential backoff + jitter. Returns the response, or ``None`` when all
    transport attempts are exhausted. Re-raises non-retryable (fatal) errors
    immediately so real bugs (bad key, malformed request) surface."""
    attempts = max(1, attempts)
    route_404_attempts = max(
        1,
        _MAX_ROUTE_404_ATTEMPTS
        if route_404_attempts is None
        else route_404_attempts,
    )
    transport_failures = 0
    route_failures = 0
    while True:
        if on_attempt is not None:
            on_attempt()
        try:
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 - classify, then retry or re-raise
            if _is_retryable_model_route_error(exc):
                route_failures += 1
                if route_failures >= route_404_attempts:
                    return None
                print(
                    "[LLM Retry] transient account-group model route 404 "
                    f"attempt {route_failures}/{route_404_attempts}",
                    file=sys.stderr,
                    flush=True,
                )
                _sleep_backoff(route_failures - 1, base, cap)
                continue
            if _is_retryable_transport_error(exc):
                transport_failures += 1
                if transport_failures >= attempts:
                    return None
                _sleep_backoff(transport_failures - 1, base, cap)
                continue
            raise


def complete_with_retry(
    client: Any,
    model: str,
    messages: list[dict],
    accept: Optional[Callable[[str], Any]] = None,
    max_attempts: int = 3,
    retry_nudge: Optional[str] = None,
    max_transport_attempts: Optional[int] = None,
    on_exchange: Optional[Callable[[list[dict], str, Any, int], None]] = None,
    **kwargs: Any,
) -> tuple[str, dict, Any]:
    """Call ``client.chat.completions.create`` with retry on empty/unacceptable responses.

    Attempts up to *max_attempts* times.  A response is "good" iff its text
    (via :func:`response_text`) is non-empty AND ``accept(text)`` is truthy
    (or *accept* is ``None``).  On each failed attempt a corrective user
    message is appended to a **copy** of *messages* (the caller's list is
    never mutated).  Usage counters are accumulated across all attempts.

    After *max_attempts* the last (text, usage, response) is returned
    unconditionally.  Transient transport failures (timeout / connection drop /
    rate-limit / 5xx) are retried up to *max_transport_attempts* times with
    exponential backoff + jitter; if they are all exhausted the call returns an
    empty result (so the caller's existing empty-response handling runs instead of
    the run hanging ~30 min or crashing).  Fatal errors (4xx / bad request / auth)
    propagate unchanged.  If ``accept`` raises, that attempt is treated as
    not-good and retried.

    :param client: OpenAI-compatible client with ``chat.completions.create``.
    :param model: Model identifier forwarded to every ``create`` call.
    :param messages: Initial message list.  Never mutated.
    :param accept: Optional predicate; if provided, text must also satisfy it
        to be considered good.  Exceptions from ``accept`` are suppressed and
        treated as a not-good result.
    :param max_attempts: Maximum total ``create`` calls (default 3).
    :param retry_nudge: Content of the corrective user message appended on
        retries.  Defaults to a generic "respond in the required format" nudge.
    :param kwargs: Additional keyword arguments forwarded verbatim to every
        ``create`` call (e.g. ``temperature=0``).
    :return: ``(text, usage, response)`` where *usage* has keys
        ``input_tokens``, ``output_tokens``, ``total_tokens`` and ``api_calls``
        accumulated across all response and transport attempts, and *response* is
        the final raw completion object
        (suitable for passing to :func:`log_llm_exchange`).
    """
    nudge = retry_nudge if retry_nudge is not None else _DEFAULT_RETRY_NUDGE
    accumulated = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }

    def count_api_call() -> None:
        accumulated["api_calls"] += 1

    # Working copy of messages; grows with nudge entries on retries.
    current_messages = list(messages)
    last_text = ""
    last_response: Any = None
    t_attempts = (
        max_transport_attempts if max_transport_attempts is not None else _MAX_TRANSPORT_ATTEMPTS
    )

    for attempt in range(max_attempts):
        response = _create_with_backoff(
            client, model, current_messages, kwargs,
            attempts=t_attempts, base=_TRANSPORT_BASE_DELAY, cap=_TRANSPORT_MAX_DELAY,
            on_attempt=count_api_call,
        )
        if response is None:
            # Transport failed after all retries → return an empty result so the
            # caller's empty-response handling (giveup / skip the cycle) runs,
            # instead of the run hanging or crashing mid-loop.
            break
        text = response_text(response)

        # Accumulate usage (getattr-safe, mirrors maintainer.py pattern).
        usage_obj = getattr(response, "usage", None)
        accumulated["input_tokens"] += getattr(usage_obj, "prompt_tokens", 0) or 0
        accumulated["output_tokens"] += getattr(usage_obj, "completion_tokens", 0) or 0
        accumulated["total_tokens"] += getattr(usage_obj, "total_tokens", 0) or 0

        last_text = text
        last_response = response

        # Optional audit hook used by the ReAct experiment runner.  It sees the
        # exact per-attempt messages (including a retry nudge) and raw response,
        # without changing ordinary callers or their retry semantics.
        if on_exchange is not None:
            try:
                on_exchange(list(current_messages), text, response, attempt + 1)
            except Exception:
                pass

        try:
            accept_ok = accept is None or bool(accept(text))
        except Exception:
            accept_ok = False
        good = bool(text.strip()) and accept_ok
        if good:
            return text, accumulated, response

        if attempt < max_attempts - 1:
            # Append nudge to a copy so the caller list stays unmutated.
            current_messages = current_messages + [{"role": "user", "content": nudge}]

    return last_text, accumulated, last_response
