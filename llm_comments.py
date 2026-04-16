from pipeline.llm_client import (
    call_gemini_json,
    validate_llm_response,
    PROMPT_VERSION,
)

COMMENT_ANALYSIS_PROMPT = """You are an audience intelligence analyst. Analyze the following Instagram comments to infer audience characteristics for creator @{handle}.

COMMENTS ({total_count} total from {post_count} posts):
{comments_block}

COMMENT TIMING DATA (UTC hour distribution — useful for timezone/geography inference):
{hour_distribution}

Respond with a JSON object:

{{
    "audience_language_distribution": {{
        "languages": {{"English": 0.5, "Hindi": 0.3, "Telugu": 0.1, "emoji_only": 0.1}},
        "primary_audience_language": "string",
        "multilingual_audience": true/false
    }},
    "audience_geography_inference": {{
        "regions": [
            {{
                "region": "string — be as specific as possible: state/city level if evidence supports it, otherwise country level",
                "confidence": 0.0-1.0,
                "signals": ["what clues pointed to this region"]
            }}
        ],
        "domestic_vs_international_split": {{
            "domestic_percentage": 0.0-1.0,
            "primary_country": "string"
        }}
    }},
    "audience_authenticity": {{
        "authenticity_score": 0.0-1.0,
        "emoji_only_percentage": 0.0-1.0,
        "generic_comment_percentage": 0.0-1.0,
        "substantive_comment_percentage": 0.0-1.0,
        "suspicious_patterns": ["any bot-like or fake engagement patterns detected"],
        "reasoning": "brief explanation of authenticity assessment"
    }},
    "audience_sentiment": {{
        "overall_sentiment": "positive/neutral/mixed/negative",
        "sentiment_score": 0.0-1.0,
        "common_positive_themes": ["what fans love"],
        "common_negative_themes": ["any complaints or criticism"],
        "trust_indicators": ["signs that audience trusts/values the creator"]
    }},
    "audience_demographics_inference": {{
        "estimated_age_group": "teen/young_adult/adult/mixed",
        "estimated_gender_skew": "female/male/balanced",
        "interest_signals": ["topics/interests commenters mention"],
        "reasoning": "what clues pointed to these demographics"
    }},
    "engagement_quality": {{
        "quality_score": 0.0-1.0,
        "conversation_depth": "shallow/moderate/deep",
        "fan_loyalty_indicators": ["signs of repeat/loyal fans"],
        "community_feel": "strong/moderate/weak"
    }}
}}

=== CRITICAL CALIBRATION GUIDES ===

AUTHENTICITY SCORING — this measures how MEANINGFUL the engagement is, not just whether commenters are real humans:
  0.1-0.3 = Almost all comments are single-word triggers ("Link", "Snd"), emoji-only, or generic ("nice", "wow"). Even if commenters are real people, the engagement is purely transactional with no genuine interaction.
  0.4-0.5 = Mix of transactional comments and some genuine reactions, but limited discussion. Comments show interest but not deep engagement.
  0.6-0.7 = Meaningful engagement present — users ask questions, share opinions, reference specific content. Some transactional comments but also real conversation.
  0.8-0.9 = Strong genuine engagement — detailed questions, personal stories, product experience sharing, follow-up discussions. Community actively participates.
  0.95-1.0 = Exceptional community — deep discussions, users help each other, thoughtful feedback, creator-audience dialogue.

IMPORTANT: An audience of real humans all typing "Link" to trigger DM automation scores 0.2-0.3 on authenticity, NOT 0.8+. "Real but transactional" is low-authenticity engagement.

GEOGRAPHY INFERENCE — use ALL available signals:
  - Comment language and script (Hindi/Devanagari = India, Tamil script = Tamil Nadu)
  - Commenter handles (regional names, regional language usernames)
  - UTC hour distribution: IST peak (UTC 13-18) = Indian audience, EST peak (UTC 12-17 summer) = US East Coast
  - Cultural references in comments (brand names, local slang, honorifics like "Di", "Bhai")
  - If evidence supports a SPECIFIC state/city, say so (e.g., "Maharashtra" not just "India")

ENGAGEMENT QUALITY — be discriminating:
  "deep" = Users write multi-sentence comments, ask follow-up questions, share personal experiences
  "moderate" = Mix of short and medium comments, some genuine questions or reactions
  "shallow" = Dominated by single-word comments, CTA triggers, emoji-only responses

LANGUAGE MIX: Values MUST be decimals summing to 1.0. Use 0.6 NOT 60.
"""


def analyze_comments(
    client,
    handle: str,
    comment_texts: list[str],
    comment_timestamps: list[str] = None,
    commenter_handles: list[str] = None,
    comment_hour_distribution: dict = None,
    num_posts_with_comments: int = 0,
) -> dict:
    """
    Run Gemini audience intelligence analysis on comments.

    Args:
        client: Initialized Gemini client
        handle: Creator handle
        comment_texts: List of comment text strings
        comment_timestamps: Optional list of ISO timestamps
        commenter_handles: Optional list of commenter usernames
        comment_hour_distribution: UTC hour -> percentage distribution
        num_posts_with_comments: Number of posts these comments came from
    """
    comments_block = ""
    for i, text in enumerate(comment_texts[:50], 1):
        parts = []
        # Add commenter handle if available
        if commenter_handles and i <= len(commenter_handles):
            parts.append(f"@{commenter_handles[i - 1]}")
        parts.append(text)
        if comment_timestamps and i <= len(comment_timestamps):
            parts.append(f"[{comment_timestamps[i - 1]}]")
        comments_block += f"{i}. {' | '.join(parts)}\n"

    # Format hour distribution for the prompt
    hour_str = "Not available"
    if comment_hour_distribution:
        sorted_hours = sorted(
            comment_hour_distribution.items(),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        top_hours = sorted_hours[:5]
        hour_str = ", ".join(
            f"UTC {h}:00 = {float(v)*100:.0f}%" for h, v in top_hours
        )

    prompt = COMMENT_ANALYSIS_PROMPT.format(
        handle=handle,
        total_count=len(comment_texts),
        post_count=num_posts_with_comments or "unknown",
        comments_block=comments_block,
        hour_distribution=hour_str,
    )

    result = call_gemini_json(client, prompt)
    result = validate_llm_response(
        result,
        [
            "audience_language_distribution",
            "audience_geography_inference",
            "audience_authenticity",
            "audience_sentiment",
            "audience_demographics_inference",
            "engagement_quality",
        ],
    )
    result["_prompt_version"] = PROMPT_VERSION
    return result
