import requests
import time
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Transient HTTP status codes worth retrying
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class BrightdataClient:
    BASE_URL = "https://api.brightdata.com/datasets/v3"

    def __init__(self, api_token: str, webhook_url: Optional[str] = None):
        self.api_token = api_token
        self.webhook_url = webhook_url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    def trigger(
        self, dataset_id: str, payload: list[dict], format: str = "json"
    ) -> str:
        """
        Trigger an async scraping job. Returns snapshot_id.
        """
        params = {
            "dataset_id": dataset_id,
            "format": format,
        }

        if self.webhook_url:
            params["webhook"] = self.webhook_url
            params["uncompressed_webhook"] = "true"

        response = self._request_with_retry(
            "POST",
            f"{self.BASE_URL}/trigger",
            params=params,
            json_body=payload,
        )
        result = response.json()
        snapshot_id = result["snapshot_id"]
        logger.info(f"Triggered job {snapshot_id} for dataset {dataset_id}")
        return snapshot_id

    def scrape(
        self,
        dataset_id: str,
        payload: list[dict],
        extra_params: dict = None,
    ) -> dict:
        """
        Synchronous scrape via /scrape endpoint.

        Uses {"input": [...]} format. Supports extra query params
        like type=discover_new, discover_by=url, etc.

        Returns either:
          {"snapshot_id": "..."} — needs poll + download
          {"data": [...]}       — inline NDJSON response, ready to use
        """
        params = {
            "dataset_id": dataset_id,
            "notify": "false",
            "include_errors": "true",
        }
        if extra_params:
            params.update(extra_params)

        body = json.dumps({"input": payload})

        response = self._request_with_retry(
            "POST",
            f"{self.BASE_URL}/scrape",
            params=params,
            data=body,
        )

        # Try standard JSON first ({"snapshot_id": "..."})
        text = response.text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "snapshot_id" in result:
                logger.info(
                    f"Scrape job {result['snapshot_id']} for dataset {dataset_id}"
                )
                return result
            # Single JSON object that is actual data
            return {"data": [result] if isinstance(result, dict) else result}
        except json.JSONDecodeError:
            pass

        # NDJSON: one JSON object per line
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed NDJSON line: {line[:100]}")
        logger.info(
            f"Scrape returned {len(records)} records inline for dataset {dataset_id}"
        )
        return {"data": records}

    def poll_snapshot(
        self, snapshot_id: str, timeout: int = 900, interval: int = 10
    ) -> str:
        """Poll until snapshot is ready. Returns status."""
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            response = requests.get(
                f"{self.BASE_URL}/progress/{snapshot_id}",
                headers=self.headers,
            )
            response.raise_for_status()
            status_data = response.json()
            status = status_data.get("status")

            if status == "ready":
                elapsed = int(time.time() - start_time)
                logger.info(f"Snapshot {snapshot_id} ready after {elapsed}s")
                return "ready"
            elif status == "failed":
                raise Exception(f"Snapshot {snapshot_id} failed: {status_data}")

            time.sleep(interval)

        raise TimeoutError(
            f"Snapshot {snapshot_id} did not complete within {timeout}s"
        )

    def download_snapshot(self, snapshot_id: str) -> list[dict]:
        """Download completed snapshot data as JSON."""
        response = self._request_with_retry(
            "GET",
            f"{self.BASE_URL}/snapshot/{snapshot_id}",
            params={"format": "json"},
        )
        return response.json()

    def trigger_and_wait(
        self, dataset_id: str, payload: list[dict]
    ) -> list[dict]:
        """
        Async: trigger + poll + download.
        """
        snapshot_id = self.trigger(dataset_id, payload)
        self.poll_snapshot(snapshot_id)
        return self.download_snapshot(snapshot_id)

    def scrape_and_wait(
        self,
        dataset_id: str,
        payload: list[dict],
        extra_params: dict = None,
    ) -> list[dict]:
        """
        Sync-style: scrape + poll + download.
        Uses /scrape endpoint with {"input": [...]} format.

        If the /scrape call returns data inline (NDJSON), returns it directly.
        If it returns a snapshot_id, polls and downloads.
        """
        result = self.scrape(dataset_id, payload, extra_params)

        if "data" in result:
            return result["data"]

        snapshot_id = result["snapshot_id"]
        self.poll_snapshot(snapshot_id)
        return self.download_snapshot(snapshot_id)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        **kwargs,
    ) -> requests.Response:
        """
        HTTP request with exponential backoff for transient failures.
        Retries on 429, 5xx, and connection errors.
        """
        # Move json_body to the correct requests kwarg
        if "json_body" in kwargs:
            kwargs["json"] = kwargs.pop("json_body")

        for attempt in range(max_retries + 1):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=self.headers,
                    **kwargs,
                )
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    if attempt < max_retries:
                        wait = 2**attempt * 2  # 2, 4, 8 seconds
                        logger.warning(
                            f"Brightdata {response.status_code} on {method} {url}, "
                            f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait)
                        continue
                if not response.ok:
                    logger.error(
                        f"Brightdata request failed ({response.status_code}): "
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

        # Should never reach here, but just in case
        raise Exception(f"Max retries exceeded for {method} {url}")
