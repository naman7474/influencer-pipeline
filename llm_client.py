import json
import logging
import re

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Bump this when prompt structure changes so old vs new results are comparable
PROMPT_VERSION = "2.0"


def init_gemini(api_key: str):
    """Initialize Gemini client and return client."""
    client = genai.Client(api_key=api_key)
    return client


def call_gemini_json(client, prompt: str, max_retries: int = 2) -> dict:
    """
    Call Gemini and parse JSON response.

    Uses response_mime_type="application/json" for structured output.
    Falls back to regex JSON extraction if direct parsing fails.
    """
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            text = response.text

            # Try direct parse first
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Try extracting the outermost JSON object with brace matching
            parsed = _extract_json_object(text)
            if parsed is not None:
                return parsed

            if attempt == max_retries:
                raise ValueError(
                    f"Could not parse JSON from Gemini response: {text[:200]}"
                )

        except Exception as e:
            logger.warning(
                f"Gemini call attempt {attempt + 1} failed: {e}"
            )
            if attempt == max_retries:
                raise
            continue


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


def validate_llm_response(
    response: dict, required_keys: list[str]
) -> dict:
    """
    Ensure critical top-level keys exist in LLM response.
    Fills missing keys with empty dicts to prevent downstream KeyErrors.
    """
    for key in required_keys:
        if key not in response or response[key] is None:
            response[key] = {}
            logger.warning(f"LLM response missing key '{key}', defaulting to {{}}")
    return response


def normalize_language_mix(lang_mix: dict) -> dict:
    """
    Normalize language_mix_percentages to decimal format (0.0-1.0).
    LLM sometimes returns percentages (60) instead of decimals (0.6).
    """
    if not lang_mix:
        return {}
    values = list(lang_mix.values())
    if not values:
        return lang_mix
    # If any value > 1.0, assume percentages and convert
    if any(v > 1.0 for v in values if isinstance(v, (int, float))):
        return {k: round(v / 100, 3) for k, v in lang_mix.items()}
    return lang_mix
