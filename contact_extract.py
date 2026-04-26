"""Bio-text contact extractors shared across IG and YT scrapers.

Creators commonly put their business email and phone in the profile bio
rather than the dedicated contact fields exposed by Instagram or YouTube.
The IG `contact_email` field requires the creator to enable Business
Account contact info; YouTube has no equivalent at all (about-page emails
are gated behind a CAPTCHA reveal). So both pipelines need to scan the
bio as a fallback.

Both helpers return the FIRST match (or None). They're intentionally
conservative — false-positive emails downstream would mean we'd email
the wrong contact during outreach.
"""

from __future__ import annotations

import re

# Email regex — RFC-5322-light. Restricts the TLD to letters only so we
# don't snag things like `name@1.2.3` (an IP-form, treated as invalid).
# Allows the common `+`, `.`, `_`, `-` in the local-part.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Phone regex — international and Indian-domestic shapes.
# Matches:
#   +91 98765 43210         (intl with space)
#   +1 (555) 123-4567       (intl with parens)
#   9876543210              (10-digit Indian mobile, leading 6/7/8/9)
#   555-123-4567            (US 7-3-3 dash)
# Avoids 4–6 digit standalone numbers (years, follower counts).
_PHONE_RE = re.compile(
    r"""
    (?:
        \+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}    # +CC NNN NNN NNNN
        |
        \(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}                             # (NNN) NNN NNNN
        |
        \b[6789]\d{9}\b                                                # IN 10-digit mobile
        |
        \b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b                                # NNN-NNN-NNNN
    )
    """,
    re.VERBOSE,
)


def extract_email_from_text(text: str | None) -> str | None:
    """Return the first plausible email address found in `text`, or None.

    Prefers business-style local-parts (`business@`, `contact@`, `hello@`,
    `info@`) when multiple emails are present — those are the
    sponsorship-intent addresses creators advertise.
    """
    if not text:
        return None

    matches = _EMAIL_RE.findall(text)
    if not matches:
        return None

    # Prefer business-intent local-parts so we don't grab a personal
    # @gmail when a `business@` is also listed.
    business_prefixes = (
        "business",
        "contact",
        "hello",
        "info",
        "press",
        "partnerships",
        "sponsorships",
        "collab",
        "collabs",
        "booking",
        "media",
        "pr",
    )
    for match in matches:
        local = match.split("@", 1)[0].lower()
        if any(local == p or local.startswith(p + "+") for p in business_prefixes):
            return match
    return matches[0]


def extract_phone_from_text(text: str | None) -> str | None:
    """Return the first plausible phone number found in `text`, or None."""
    if not text:
        return None
    match = _PHONE_RE.search(text)
    return match.group(0).strip() if match else None
