"""HTTP client to the Next.js brand-match batch endpoint.

Discovery is Python; the brand-match scoring engine
(`web/src/lib/matching/engine.ts`) is TypeScript with a complex set of
sub-scores, calibration files, and weights that we deliberately do not
port to Python. Instead the discovery service POSTs the creator IDs to
the Next.js endpoint after stage `matching` so the TS engine reads
caption_intelligence + audience_intelligence (which we just wrote) and
upserts `creator_brand_matches` for this brand.

Authenticates via a shared secret header (`X-Discovery-Service-Secret`)
that both sides read from env. NOT a user-auth flow — service-to-service.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


class BrandMatchClient:
    def __init__(
        self,
        base_url: str | None = None,
        secret: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (
            base_url
            or os.environ.get("DISCOVERY_WEB_BASE_URL", "")
        ).rstrip("/")
        self.secret = secret or os.environ.get("DISCOVERY_SERVICE_SECRET", "")
        self.timeout = timeout
        if not self.base_url:
            logger.warning(
                "BrandMatchClient: DISCOVERY_WEB_BASE_URL not set; "
                "brand-match scoring will be skipped"
            )
        if not self.secret:
            logger.warning(
                "BrandMatchClient: DISCOVERY_SERVICE_SECRET not set; "
                "brand-match scoring will be skipped"
            )

    def compute_batch(
        self,
        creator_ids: Iterable[str],
        brand_id: str,
    ) -> dict:
        """POST /api/matching/compute-batch.

        Returns the parsed JSON response: `{computed, failed, errors}`.
        On HTTP / network error returns a dict with `error` populated;
        the caller is expected to log + bump `discovery_requests.status`
        to 'failed' if scoring is required for the request to be useful.
        """
        ids = [c for c in creator_ids if c]
        if not ids:
            return {"computed": 0, "failed": 0, "errors": []}
        if not self.base_url or not self.secret:
            return {
                "computed": 0,
                "failed": len(ids),
                "errors": ["brand_match_client unconfigured"],
            }

        url = f"{self.base_url}/api/matching/compute-batch"
        headers = {
            "X-Discovery-Service-Secret": self.secret,
            "Content-Type": "application/json",
        }
        payload = {"creator_ids": ids, "brand_id": brand_id}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            logger.error(
                f"compute-batch HTTP {e.response.status_code}: {body}"
            )
            return {
                "computed": 0,
                "failed": len(ids),
                "errors": [f"http {e.response.status_code}: {body}"],
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"compute-batch network error: {e}")
            return {
                "computed": 0,
                "failed": len(ids),
                "errors": [str(e)],
            }
