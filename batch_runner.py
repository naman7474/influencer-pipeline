import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.pipeline import build_creator_intelligence_profile, clean_cip_for_export
from pipeline.db import store_full_cip
from pipeline.calibration import load_er_benchmarks

logger = logging.getLogger(__name__)


def run_batch(
    profile_urls: list[str],
    max_workers: int = 3,
    output_dir: str = "./cip_output",
    db=None,
) -> list[dict]:
    """
    Process multiple creators in parallel.

    Recommended: max_workers=3 to stay within Brightdata rate limits.
    For 1,000 creators at 3 parallel workers:
    - ~5 min per creator (scraping wait time)
    - ~28 hours total
    - ~$130 total cost (Brightdata + Whisper + Gemini)
    """
    os.makedirs(output_dir, exist_ok=True)

    brightdata_token = os.environ["BRIGHTDATA_API_TOKEN"]
    gemini_key = os.environ["GEMINI_API_KEY"]
    openai_key = os.environ["OPENAI_API_KEY"]

    results = []

    # Pre-load percentile-calibrated ER benchmarks once per batch.
    # Falls back to DEFAULT_ER_BENCHMARKS when no live calibration
    # exists yet; loader caches internally so parallel workers share.
    er_benchmarks = load_er_benchmarks(db) if db else None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(
                build_creator_intelligence_profile,
                url,
                brightdata_token,
                gemini_key,
                openai_key,
                er_benchmarks=er_benchmarks,
            ): url
            for url in profile_urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                cip = future.result()
                handle = cip.get("profile", {}).get("handle", "unknown")

                # Store to DB first (before cleaning internal fields)
                if db and not cip.get("error"):
                    try:
                        store_full_cip(db, cip)
                    except Exception as e:
                        logger.error(f"DB store failed for @{handle}: {e}")

                # Export clean copy (preserves original for DB)
                cip_export = clean_cip_for_export(cip)
                filepath = os.path.join(output_dir, f"{handle}.json")
                with open(filepath, "w") as f:
                    json.dump(cip_export, f, indent=2, default=str)

                results.append(cip_export)
                cpi = cip.get("scores", {}).get("cpi", "N/A")
                logger.info(f"Completed @{handle} — CPI: {cpi}")

            except Exception as e:
                logger.error(f"Failed {url}: {e}")
                results.append({"profile_url": url, "error": str(e)})

    return results
