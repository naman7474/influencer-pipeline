from pipeline.llm_client import (
    call_gemini_json,
    PROMPT_VERSION,
)
from pipeline.schemas.intelligence import (
    TranscriptIntelligencePayload,
    is_llm_failure,
)

TRANSCRIPT_ANALYSIS_PROMPT = """You are an influencer content analyst specializing in video content quality. Analyze the following reel transcripts from a single Instagram creator.

CREATOR HANDLE: {handle}
DATA QUALITY NOTE: {data_quality_note}

TRANSCRIPTS:
{transcripts_block}

Respond with a JSON object:

{{
    "data_quality": "sufficient" or "insufficient" — set to "insufficient" if fewer than 2 reels have actual speech (not just music),
    "speaking_language": {{
        "primary_spoken_language": "string — the actual language spoken by the creator (ignore background music lyrics)",
        "languages_spoken": ["languages the CREATOR actually speaks, not background music"],
        "accent_notes": "any notable accent or dialect observations about the creator's speech",
        "caption_vs_spoken_mismatch": true/false — does the creator write captions in a different language than they speak?
    }},
    "hook_analysis": {{
        "hooks": [
            {{
                "post_id": "string",
                "hook_text": "first 3 seconds transcript",
                "hook_type": "question/statement/shock/story/direct_address/music_only",
                "hook_quality_score": 0.0-1.0,
                "reasoning": "why this hook works or doesn't"
            }}
        ],
        "avg_hook_quality": 0.0-1.0,
        "dominant_hook_style": "string"
    }},
    "brand_mention_analysis": [
        {{
            "post_id": "string",
            "brand_mentioned": "string or null — set to null if no brand is mentioned",
            "mention_naturalness": null if no brand mentioned, else 0.0-1.0,
            "feels_scripted": null if no brand mentioned, else true/false,
            "integration_style": "organic_conversation/dedicated_segment/passing_mention/hard_sell/none"
        }}
    ],
    "content_depth": {{
        "avg_word_count_per_reel": 0,
        "vocabulary_complexity": "simple/moderate/advanced",
        "educational_density": 0.0-1.0,
        "storytelling_score": 0.0-1.0,
        "filler_word_frequency": "low/medium/high"
    }},
    "regional_signals": {{
        "cultural_references": ["any regional/cultural references detected in creator's SPEECH"],
        "local_places_mentioned": ["any specific cities, neighborhoods, landmarks"],
        "regional_language_phrases": ["any regional language words/phrases the creator uses"],
        "estimated_region": "best guess of creator's region based on their speech patterns (not background music)"
    }}
}}

=== IMPORTANT RULES ===

1. MUSIC vs SPEECH: If a reel's transcript is clearly song lyrics or background music (marked as music_only, or contains lyrics in a language the creator doesn't otherwise speak), DO NOT use it for language detection, content depth analysis, or regional signals. Only analyze the creator's actual speech.

2. BRAND MENTIONS: When no brand is mentioned in a reel, set brand_mentioned, mention_naturalness, and feels_scripted to null, and integration_style to "none". Do NOT set mention_naturalness to 0.0 — that means "completely unnatural", which is different from "no mention".

3. DATA QUALITY: If most/all transcripts are music-only with no creator speech, set data_quality to "insufficient" and provide best-effort analysis based on available data. Clearly note in accent_notes and estimated_region that the data is limited.

4. HOOK QUALITY CALIBRATION:
   0.9-1.0 = Exceptional hook that creates immediate curiosity or value proposition
   0.7-0.8 = Good hook with clear viewer benefit or relatable scenario
   0.5-0.6 = Average hook, trending audio with visual sync
   0.3-0.4 = Weak hook, generic or repetitive across reels
   0.1-0.2 = No real hook, just background music with no verbal element
"""


def analyze_transcripts(
    client, handle: str, transcripts: list[dict]
) -> dict:
    """
    Run Gemini analysis on Whisper transcripts.

    Args:
        client: Initialized Gemini client
        handle: Creator handle
        transcripts: List of transcript dicts from transcribe_reels()
                     (should already have music-only transcripts filtered)

    Returns:
        Structured transcript intelligence
    """
    speech_transcripts = [
        t for t in transcripts
        if t.get("transcript_text") and not t.get("is_likely_music", False)
    ]
    music_transcripts = [
        t for t in transcripts
        if t.get("is_likely_music", False)
    ]

    if not speech_transcripts and not music_transcripts:
        return {"data_quality": "insufficient", "_prompt_version": PROMPT_VERSION}

    data_quality_note = (
        f"{len(speech_transcripts)} reels with speech, "
        f"{len(music_transcripts)} with music-only (excluded from analysis)."
    )
    if not speech_transcripts:
        data_quality_note += (
            " WARNING: No creator speech detected — all reels use background "
            "music/trending audio. Analysis will be very limited."
        )

    transcripts_block = ""
    for t in transcripts:
        if not t.get("transcript_text"):
            continue
        # IG path keys transcripts on `post_id`; YT path on `video_id`. Accept either.
        item_id = t.get("post_id") or t.get("video_id") or "?"
        is_music = t.get("is_likely_music", False)
        label = " [MUSIC-ONLY — do not use for language/content analysis]" if is_music else ""
        transcripts_block += f"""
--- Reel {item_id}{label} (length: {t.get('reel_length_seconds', '?')}s, detected language: {t.get('detected_language', '?')}, confidence: {t.get('avg_confidence', '?')}) ---
Hook (first 3s): {t.get('hook_text', 'N/A')}
Full transcript: {t['transcript_text'][:1000]}
Caption for context: {t.get('caption', '')[:200]}
"""

    if not transcripts_block.strip():
        return {"data_quality": "insufficient", "_prompt_version": PROMPT_VERSION}

    prompt = TRANSCRIPT_ANALYSIS_PROMPT.format(
        handle=handle,
        data_quality_note=data_quality_note,
        transcripts_block=transcripts_block,
    )

    result = call_gemini_json(
        client, prompt,
        expected_schema=TranscriptIntelligencePayload,
        dimension="transcripts",
    )
    if is_llm_failure(result):
        result["_prompt_version"] = PROMPT_VERSION
        return result
    result["_prompt_version"] = PROMPT_VERSION
    return result
