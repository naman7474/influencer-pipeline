"""Send Instagram DMs via the Apify cookie-based actor.

Drains `instagram_dm_send_apify` jobs from background_jobs. Each job carries:

    payload = {
        "account_id": "<brand_instagram_accounts.id>",
        "recipient_id": "<IG-scoped user id (PSID) or @handle>",
        "body": "<message text>",
        "provisional_message_id": "<apify-pending-... placeholder written by web>",
    }

Pipeline:
  1. Look up the brand_instagram_accounts row → decrypt cookies blob.
  2. Resolve the recipient handle (the actor wants @handles, not PSIDs;
     if `recipient_id` is numeric we attempt a Graph public lookup but
     accept any failure and fall back to the raw value).
  3. Call apify/am_production~instagram-direct-messages-dms-automation
     with INSTAGRAM_COOKIES + influencers + messages.
  4. On success: update outreach_messages.provider_message_id from
     `provisional_message_id` to whatever Apify returned (if anything;
     this actor is fire-and-forget so we may keep the provisional id).
  5. On failure: re-raise so the api.py dispatcher marks the job failed.
     The brand's UI will surface the error and prompt cookie refresh.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pipeline.apify_client import ApifyClient, make_default_client
from pipeline.ig_token_crypto import decode_ig_cookies

logger = logging.getLogger(__name__)

# The Apify actor ID. Override via env if you swap providers later.
_DEFAULT_ACTOR = "am_production/instagram-direct-messages-dms-automation"


def _actor_id() -> str:
    return os.environ.get("APIFY_IG_DM_SEND_ACTOR", _DEFAULT_ACTOR)


def _resolve_recipient_handle(raw: str) -> str:
    """The actor expects an @handle. Web layer passes whatever it has —
    sometimes that's a PSID (numeric id), sometimes already a handle.
    Strip leading @ if present; numeric strings are returned as-is and
    the actor's own handle-resolution will surface a clear error if the
    PSID can't be resolved on its side."""
    s = raw.strip().lstrip("@")
    return s


def _is_likely_handle(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._]{1,30}", s)) and not s.isdigit()


def handle_instagram_dm_send_apify(db, job: dict) -> None:
    payload = job.get("payload") or {}
    account_id = payload.get("account_id")
    recipient_id = payload.get("recipient_id")
    body = payload.get("body")
    provisional_id = payload.get("provisional_message_id")

    if not account_id or not recipient_id or not body:
        raise ValueError(
            "instagram_dm_send_apify job missing account_id/recipient_id/body"
        )

    # 1. Load account, decrypt cookies.
    res = (
        db.table("brand_instagram_accounts")
        .select("id, brand_id, access_token, personal_token_kind, ig_username")
        .eq("id", account_id)
        .single()
        .execute()
    )
    account = res.data
    if not account:
        raise RuntimeError(f"brand_instagram_accounts row not found: {account_id}")
    if account.get("personal_token_kind") != "apify_actor":
        raise RuntimeError(
            f"account {account_id} not in apify_actor mode "
            f"(got {account.get('personal_token_kind')})"
        )

    cookies = decode_ig_cookies(account["access_token"])

    # 2. Resolve recipient.
    recipient = _resolve_recipient_handle(recipient_id)
    if not _is_likely_handle(recipient):
        # Numeric PSID — we don't have a way to map back to @handle without
        # another Apify call or Meta Graph access. Surface clearly so the
        # UI can prompt the user to retry once a handle is available.
        raise RuntimeError(
            f"recipient_id {recipient!r} is not an Instagram @handle; "
            f"the cookie-mode actor requires handles. Resolve upstream."
        )

    # 3. Call Apify.
    client: ApifyClient = make_default_client()
    actor_input = {
        "INSTAGRAM_COOKIES": cookies,
        "influencers": [recipient],
        "messages": [body],
    }

    logger.info(
        "ig dm send: account=%s @%s -> @%s (%d chars)",
        account_id,
        account.get("ig_username"),
        recipient,
        len(body),
    )

    try:
        items = client.trigger_and_wait(_actor_id(), actor_input, timeout=600)
    except Exception:
        # Re-raise; api.py marks the job failed and surfaces error in UI.
        logger.exception("apify ig dm send failed: account=%s -> @%s", account_id, recipient)
        raise

    # 4. Best-effort: update the outreach_messages row with anything the
    # actor returned. Actor doesn't surface a stable per-message id, so
    # we mainly use this to confirm send succeeded and clear the
    # `apify-pending-` placeholder by setting status='sent'.
    apify_msg_id = _extract_message_id(items) or provisional_id
    if provisional_id:
        try:
            db.table("outreach_messages").update(
                {"provider_message_id": apify_msg_id, "status": "sent"}
            ).eq("provider_message_id", provisional_id).execute()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "outreach_messages update failed for provisional_id=%s: %s",
                provisional_id,
                e,
            )

    logger.info(
        "ig dm send ok: account=%s -> @%s (apify_items=%d)",
        account_id,
        recipient,
        len(items or []),
    )


def _extract_message_id(items: list[dict]) -> str | None:
    """The actor's output schema isn't documented; try the obvious shapes."""
    if not items:
        return None
    first = items[0] if isinstance(items, list) else items
    if not isinstance(first, dict):
        return None
    for k in ("messageId", "message_id", "id"):
        v = first.get(k)
        if isinstance(v, str) and v:
            return v
    return None
