"""Apify Web Scraper API client.

Public methods:
- ``start_run(actor_id, payload, webhooks=None)`` — start an actor run
  and return immediately with ``{id, defaultDatasetId, status}``. Used
  by the webhook-driven IG FSM (Phase 2+).
- ``fetch_dataset_items(dataset_id)`` — pull a finished run's records.
- ``get_run_status(run_id)`` — one-shot status check (no polling loop).
  Used by the stale-job recovery sweep.
- ``trigger_and_wait(actor_id, payload)`` — legacy sync flow. Starts a
  run, polls until finished, returns dataset items. Kept for non-FSM
  callers (e.g. brand scrape, single-post lookups).
- ``scrape_and_wait(actor_id, payload, extra_params=None)`` — same
  contract; ``extra_params`` is shallow-merged into the actor input.

The client speaks raw HTTP (no ``apify-client`` SDK dependency).
"""

import json
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Run statuses that mean "still working" — anything else is terminal.
_RUNNING_STATUSES = {"READY", "RUNNING"}
_SUCCESS_STATUSES = {"SUCCEEDED"}
_FAILURE_STATUSES = {"FAILED", "ABORTED", "TIMED-OUT"}


class ApifyClient:
    BASE_URL = "https://api.apify.com/v2"

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    # ── public API ───────────────────────────────────────────────────────

    def start_run(
        self,
        actor_id: str,
        payload: list[dict] | dict,
        webhooks: list[dict] | None = None,
    ) -> dict:
        """Start an actor run and return immediately.

        ``webhooks`` is a list of webhook specs in Apify's ad-hoc inline
        format: ``[{"eventTypes": [...], "requestUrl": "...",
        "headersTemplate": "...", "payloadTemplate": "..."}]``. Apify
        accepts inline webhooks as a base64-encoded JSON array on the
        ``webhooks`` query param of the run-create endpoint.

        Returns the run resource dict — at minimum ``{id, defaultDatasetId,
        status}``. The caller is expected to either receive an Apify
        webhook callback or poll ``get_run_status`` later.
        """
        import base64

        actor_path = actor_id.replace("/", "~")
        url = f"{self.BASE_URL}/acts/{actor_path}/runs"
        params: dict[str, str] = {}
        if webhooks:
            params["webhooks"] = base64.b64encode(
                json.dumps(webhooks).encode("utf-8")
            ).decode("ascii")
        response = self._request_with_retry(
            "POST", url, json_body=payload, params=params or None
        )
        data = response.json()
        return data.get("data") or data

    def fetch_dataset_items(self, dataset_id: str) -> list[dict]:
        """Pull a finished run's dataset items as a clean JSON array."""
        return self._fetch_dataset_items(dataset_id)

    def get_run_status(self, run_id: str) -> dict:
        """One-shot status check for ``run_id``. No polling.

        Returns ``{id, status, defaultDatasetId, statusMessage, ...}``.
        Used by the stale-job recovery sweep when a webhook is missed.
        """
        response = self._request_with_retry(
            "GET", f"{self.BASE_URL}/actor-runs/{run_id}"
        )
        data = response.json()
        return data.get("data") or data

    def trigger_and_wait(
        self,
        actor_id: str,
        payload: list[dict] | dict,
        timeout: int = 900,
        interval: int = 10,
    ) -> list[dict]:
        """Run an actor synchronously and return its dataset items.

        ``payload`` is the actor's ``input`` object. Most Apify Instagram
        actors accept either a dict (e.g. ``{"usernames": [...]}``) or
        a list (legacy actors). Pass through whatever the caller hands us.
        """
        run = self._start_run(actor_id, payload)
        run_id = run["id"]
        dataset_id = run["defaultDatasetId"]
        logger.info(
            f"Apify run {run_id} started for actor {actor_id} "
            f"(dataset {dataset_id})"
        )

        self._poll_run(run_id, timeout=timeout, interval=interval)
        return self._fetch_dataset_items(dataset_id)

    def scrape_and_wait(
        self,
        actor_id: str,
        payload: list[dict] | dict,
        extra_params: Optional[dict] = None,
        timeout: int = 900,
        interval: int = 10,
    ) -> list[dict]:
        """Signature-compatible alias for ``trigger_and_wait``.

        BrightData splits sync vs async scrape endpoints; Apify only
        has one actor-run flow. ``extra_params`` is shallow-merged into
        ``payload`` when payload is a dict, otherwise ignored — callers
        that need more nuanced merging should build the payload upstream.
        """
        if extra_params and isinstance(payload, dict):
            payload = {**payload, **extra_params}
        return self.trigger_and_wait(
            actor_id, payload, timeout=timeout, interval=interval
        )

    # ── internals ────────────────────────────────────────────────────────

    def _start_run(self, actor_id: str, payload) -> dict:
        # Apify actor IDs in URLs use ``~`` instead of ``/`` to avoid
        # path-segment collisions: ``apify/instagram-scraper`` →
        # ``apify~instagram-scraper``.
        actor_path = actor_id.replace("/", "~")
        url = f"{self.BASE_URL}/acts/{actor_path}/runs"
        response = self._request_with_retry(
            "POST", url, json_body=payload
        )
        body = response.json()
        return body.get("data") or body

    def _poll_run(self, run_id: str, timeout: int, interval: int) -> None:
        start = time.time()
        consecutive_net_errs = 0
        while (time.time() - start) < timeout:
            try:
                response = requests.get(
                    f"{self.BASE_URL}/actor-runs/{run_id}",
                    headers=self.headers,
                    timeout=30,
                )
                response.raise_for_status()
                run = (response.json().get("data") or response.json())
                consecutive_net_errs = 0
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                consecutive_net_errs += 1
                logger.warning(
                    "Polling Apify run %s: network error (%s); retrying "
                    "in %ds (consecutive=%d)",
                    run_id, type(e).__name__, interval, consecutive_net_errs,
                )
                time.sleep(interval)
                continue

            status = run.get("status")
            if status in _SUCCESS_STATUSES:
                elapsed = int(time.time() - start)
                logger.info(f"Apify run {run_id} succeeded after {elapsed}s")
                return
            if status in _FAILURE_STATUSES:
                raise Exception(
                    f"Apify run {run_id} ended with status {status}: "
                    f"{run.get('statusMessage') or run.get('exitCode')}"
                )

            time.sleep(interval)

        raise TimeoutError(
            f"Apify run {run_id} did not complete within {timeout}s"
        )

    def _fetch_dataset_items(self, dataset_id: str) -> list[dict]:
        # ``clean=true`` strips Apify-internal metadata (#error, #debug)
        # so the items match the shape the actor's documentation shows.
        response = self._request_with_retry(
            "GET",
            f"{self.BASE_URL}/datasets/{dataset_id}/items",
            params={"format": "json", "clean": "true"},
        )
        items = response.json()
        if not isinstance(items, list):
            # Some actors return NDJSON when called without ``format=json``;
            # we explicitly ask for JSON, so this is defensive only.
            return [items] if items else []
        return items

    def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        **kwargs,
    ) -> requests.Response:
        if "json_body" in kwargs:
            kwargs["json"] = kwargs.pop("json_body")

        for attempt in range(max_retries + 1):
            try:
                response = requests.request(
                    method, url, headers=self.headers, timeout=60, **kwargs,
                )
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    if attempt < max_retries:
                        wait = 2**attempt * 2
                        logger.warning(
                            f"Apify {response.status_code} on {method} {url}, "
                            f"retrying in {wait}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait)
                        continue
                if not response.ok:
                    logger.error(
                        f"Apify request failed ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                    response.raise_for_status()
                return response

            except requests.ConnectionError as e:
                if attempt < max_retries:
                    wait = 2**attempt * 2
                    logger.warning(
                        f"Connection error on {method} {url}, "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                    continue
                raise

        raise Exception(f"Max retries exceeded for {method} {url}")


def make_default_client() -> ApifyClient:
    """Construct an ApifyClient from the ``APIFY_TOKEN`` env var.

    Convenience for the dispatcher path so individual scrapers don't
    each have to plumb the token through their signatures.
    """
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise RuntimeError(
            "APIFY_TOKEN is not set; Apify is the only scraping provider"
        )
    return ApifyClient(api_token=token)
