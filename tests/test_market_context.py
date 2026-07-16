"""The Mizani market-context feed must never break the console.

fetch_market_context returns a validated dict or None — it must not raise
for any network condition, bad payload, or schema drift, because the
dashboard renders a fallback caption when it gets None.
"""

from unittest.mock import MagicMock, patch

import requests

from src.market_context import fetch_market_context, summarize

VALID_FEED = {
    "schema_version": 1,
    "generated_at_utc": "2026-07-16T12:00:00+00:00",
    "fx": {
        "kes_per_usd": {"mean": 157.32, "buy": 156.95, "sell": 157.69, "as_of": "2024-01-03"},
        "regional_units_per_kes": {},
    },
    "kenya_mobile_money": {
        "as_of_month": "2026-05-01",
        "registered_accounts_millions": 94.09,
        "active_agents": 564330,
        "agent_cico_volume_million": 214.33,
        "agent_cico_value_ksh_billions": 681.45,
        "cico_value_ksh_billions_last_12m": [
            {"month": "2026-04-01", "value": 680.99},
            {"month": "2026-05-01", "value": 681.45},
        ],
    },
    "account_ownership": {"KEN": {"pct_adults_15plus": 90.1, "as_of_year": 2024}},
}


def _response(json_body, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


class TestFetchMarketContext:
    def test_valid_feed_is_returned(self):
        with patch("src.market_context.requests.get", return_value=_response(VALID_FEED)):
            feed = fetch_market_context()
        assert feed is not None
        assert feed["fx"]["kes_per_usd"]["mean"] == 157.32

    def test_network_failure_returns_none_not_raise(self):
        with patch(
            "src.market_context.requests.get",
            side_effect=requests.ConnectionError("offline"),
        ):
            assert fetch_market_context() is None

    def test_timeout_returns_none(self):
        with patch(
            "src.market_context.requests.get",
            side_effect=requests.Timeout("slow"),
        ):
            assert fetch_market_context() is None

    def test_unknown_schema_version_is_rejected(self):
        feed = dict(VALID_FEED, schema_version=2)
        with patch("src.market_context.requests.get", return_value=_response(feed)):
            assert fetch_market_context() is None

    def test_partial_feed_is_rejected_before_render(self):
        feed = {"schema_version": 1, "fx": {"kes_per_usd": {"mean": 157.32}}}
        with patch("src.market_context.requests.get", return_value=_response(feed)):
            assert fetch_market_context() is None

    def test_non_json_body_returns_none(self):
        resp = _response(None)
        resp.json.side_effect = ValueError("not json")
        with patch("src.market_context.requests.get", return_value=resp):
            assert fetch_market_context() is None


class TestSummarize:
    def test_flattens_the_fields_the_panel_renders(self):
        s = summarize(VALID_FEED)
        assert s["kes_per_usd"] == 157.32
        assert s["fx_as_of"] == "2024-01-03"  # staleness stays visible
        assert s["accounts_millions"] == 94.09
        assert s["active_agents"] == 564330
        assert s["mm_as_of"] == "2026-05-01"
        assert s["cico_trend"] == [680.99, 681.45]
