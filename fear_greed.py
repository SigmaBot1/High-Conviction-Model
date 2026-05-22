"""
CNN Fear & Greed Index fetcher with 24-hour file cache.

Endpoint: https://production.dataviz.cnn.io/index/fearandgreed/graphdata
Field:    fear_and_greed.score  (float, 0–100; lower = more fear)

Score interpretation:
  0–25   Extreme Fear
  25–35  Fear
  35–55  Neutral
  55–75  Greed
  75–100 Extreme Greed

The model uses < 35 as the fear threshold.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CNN_URL  = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_CACHE_TTL = 86_400  # 24 hours in seconds
_CACHE_FILE = "fear_greed.json"


def fetch_fear_greed_score(cache_dir: str = "data_cache") -> Optional[float]:
    """
    Return the current CNN Fear & Greed Index score (0–100).

    Caches the result for 24 hours in <cache_dir>/fear_greed.json.
    Returns None if the fetch fails or the data is unavailable — callers
    should treat None as "condition unavailable", not as a failure.
    """
    cache_path = Path(cache_dir) / _CACHE_FILE

    # Serve from cache if fresh
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if time.time() - cached.get("ts", 0) < _CACHE_TTL:
                return float(cached["score"])
        except Exception:
            pass

    # Fetch live — CNN requires browser-like headers to avoid 418
    try:
        resp = requests.get(
            _CNN_URL,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
                "Origin": "https://www.cnn.com",
            },
        )
        resp.raise_for_status()
        data  = resp.json()
        score = float(data["fear_and_greed"]["score"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"score": score, "ts": time.time()}))
        logger.debug(f"CNN Fear & Greed score fetched: {score:.1f}")
        return score
    except Exception as e:
        logger.debug(f"CNN Fear & Greed fetch failed: {e}")
        return None
