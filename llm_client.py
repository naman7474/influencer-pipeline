import json
import logging
import time
from typing import Any, Optional, Type

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Bump this when prompt structure changes so old vs new results are comparable
PROMPT_VERSION = "2.0"


def init_gemini(api_key: str):
    """Initialize Gemini client and return client."""
    client = genai.Client(api_key=api_key)
    return client


def call_gemini_json(
    client,
    prompt: str,
    *,
    max_retries: int = 3,
    expected_schema: Optional[Type[BaseModel]] = None,
    dimension: Optional[str] = None,
) -> dict:
    """
    Call Gemini, parse JSON, and (optionally) validate against a
    Pydantic schema.

    Returns:
      - a dict (schema-validated `model_dump()` when `expected_schema`
        is set, otherwise the raw JSON)
      - an `LLMFailure` dict on terminal failure (after `max_retries`
        parse/validation exhaustion). **Does not raise** on parse or
        validation failure so per-creator pipelines degrade gracefully.
        Network/SDK exceptions on the final attempt do still propagate.

    Backoff: exponential — 1s, 2s, 4s between attempts.
    """
    # Lazy import to avoid a circular dep with schemas package on test harnesses.
    from pipeline.schemas.intelligence import LLMFailure

    last_error: str = "unknown"
    current_prompt = prompt

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            text = response.text

            # ── Parse JSON ────────────────────────────────────────
            parsed: Any = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = _extract_json_object(text)

            if parsed is None:
                last_error = f"json_parse_failed: {text[:200]!r}"
                logger.warning(
                    f"Gemini parse failed (attempt {attempt + 1}/{max_retries + 1}) "
                    f"for dim={dimension}: {last_error}"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return LLMFailure(
                    dimension=dimension or "unknown",
                    error=last_error,
                    prompt_snippet=prompt[:300],
                )

            # ── Schema-validate ───────────────────────────────────
            if expected_schema is None:
                return parsed  # legacy path — caller does its own validation

            try:
                validated = expected_schema.model_validate(parsed)
                return validated.model_dump()
            except ValidationError as ve:
                last_error = f"schema_validation_failed: {ve.errors(include_url=False)[:3]}"
                logger.warning(
                    f"Gemini validation failed (attempt {attempt + 1}/{max_retries + 1}) "
                    f"for dim={dimension}: {last_error}"
                )
                if attempt < max_retries:
                    # Retry once with the validator error appended —
                    # Gemini often self-corrects when shown the issue.
                    current_prompt = (
                        prompt
                        + "\n\nYour previous response failed validation: "
                        + str(ve.errors(include_url=False)[:3])
                        + "\nReturn ONLY valid JSON matching the schema."
                    )
                    time.sleep(2 ** attempt)
                    continue
                return LLMFailure(
                    dimension=dimension or "unknown",
                    error=last_error,
                    prompt_snippet=prompt[:300],
                )

        except Exception as e:
            last_error = f"sdk_error: {type(e).__name__}: {e}"
            logger.warning(
                f"Gemini SDK error (attempt {attempt + 1}/{max_retries + 1}) "
                f"for dim={dimension}: {last_error}"
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            # Terminal SDK failure — surface as LLMFailure too so the
            # per-creator pipeline keeps going and the CPI degrades
            # gracefully rather than aborting the whole run.
            return LLMFailure(
                dimension=dimension or "unknown",
                error=last_error,
                prompt_snippet=prompt[:300],
            )

    # Unreachable — retained as belt-and-braces.
    return LLMFailure(
        dimension=dimension or "unknown",
        error=last_error or "max_retries_exceeded",
        prompt_snippet=prompt[:300],
    )


def _extract_json_object(text: str) -> dict | None:
    """
    Extract the outermost JSON object from text using brace depth tracking.
    More robust than simple first-{-to-last-} which breaks with explanatory text.
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Legacy helpers, retained for backward-compat until every caller
# migrates to schema-validated call_gemini_json. ──────────────────

def validate_llm_response(
    response: dict, required_keys: list[str]
) -> dict:
    """
    Ensure critical top-level keys exist in LLM response.
    Fills missing keys with empty dicts to prevent downstream KeyErrors.

    DEPRECATED: prefer schema-based validation via `expected_schema`
    on `call_gemini_json`. Kept so legacy callers don't break mid-migration.
    """
    for key in required_keys:
        if key not in response or response[key] is None:
            response[key] = {}
            logger.warning(f"LLM response missing key '{key}', defaulting to {{}}")
    return response


def normalize_language_mix(lang_mix: dict) -> dict:
    """DEPRECATED: schema validators now handle this. Retained for parity."""
    if not lang_mix:
        return {}
    values = list(lang_mix.values())
    if not values:
        return lang_mix
    if any(v > 1.0 for v in values if isinstance(v, (int, float))):
        return {k: round(v / 100, 3) for k, v in lang_mix.items()}
    return lang_mix
