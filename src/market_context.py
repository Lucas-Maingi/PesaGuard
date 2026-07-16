"""Live Kenya market context from Mizani's published data feed.

Mizani (https://github.com/Lucas-Maingi/mizani) is the data-platform side
of this ecosystem: a medallion ETL pipeline over Central Bank of Kenya,
World Bank, and GSMA data that publishes a machine-readable snapshot on
every scheduled run. PesaGuard consumes it to show analysts the market
they are protecting.

Design rules:
  * never raises — the console must work fully offline, so any network,
    schema, or parsing problem returns None and the panel degrades to a
    caption instead of an error
  * every value in the feed carries its own `as_of` date (the FX history
    file lags; mobile-money statistics are near-current) and the UI must
    display it rather than imply freshness
"""

import requests

MIZANI_FEED_URL = "https://lucas-maingi.github.io/mizani/data/latest.json"
FEED_SCHEMA_VERSION = 1


def fetch_market_context(url: str = MIZANI_FEED_URL, timeout: float = 6.0) -> dict | None:
    """Fetch and validate the Mizani feed. Returns None on any failure."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        feed = resp.json()
        if feed.get("schema_version") != FEED_SCHEMA_VERSION:
            return None
        # touch every field the panel renders so a partial feed fails here,
        # not mid-render
        float(feed["fx"]["kes_per_usd"]["mean"])
        str(feed["fx"]["kes_per_usd"]["as_of"])
        mm = feed["kenya_mobile_money"]
        float(mm["registered_accounts_millions"])
        int(mm["active_agents"])
        float(mm["agent_cico_value_ksh_billions"])
        str(mm["as_of_month"])
        return feed
    except Exception:
        return None


def summarize(feed: dict) -> dict:
    """Flatten the feed into the fields the sidebar panel displays."""
    fx = feed["fx"]["kes_per_usd"]
    mm = feed["kenya_mobile_money"]
    trend = [p["value"] for p in mm.get("cico_value_ksh_billions_last_12m", [])]
    return {
        "kes_per_usd": fx["mean"],
        "fx_as_of": fx["as_of"],
        "accounts_millions": mm["registered_accounts_millions"],
        "active_agents": mm["active_agents"],
        "cico_value_ksh_billions": mm["agent_cico_value_ksh_billions"],
        "mm_as_of": mm["as_of_month"],
        "cico_trend": trend,
        "generated_at": feed.get("generated_at_utc", ""),
    }
