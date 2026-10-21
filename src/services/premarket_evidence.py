"""Conservative publication-time and provenance checks for premarket context.

Search metadata is not a substitute for the linked article itself.  Date-only
metadata from an earlier date is usable with a partial-verification label;
same-day date-only metadata cannot establish that it existed by run cutoff.
"""

from __future__ import annotations

import ipaddress
from datetime import datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

_SOURCE_HOST_SUFFIXES = (
    "sec.gov", "reuters.com", "apnews.com", "longbridge.com",
    "nasdaq.com", "nyse.com", "finance.yahoo.com",
)


class _PublicationMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.published: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        key = (values.get("property") or values.get("name") or "").lower()
        if key in {"article:published_time", "datepublished", "publish_date", "publication_date"}:
            self.published = values.get("content") or self.published


@lru_cache(maxsize=256)
def _source_publication(url: str) -> str | None:
    """Bounded fetch of trusted public-source HTML metadata, no redirects."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme != "https" or port not in (None, 443) or not any(
        host == suffix or host.endswith("." + suffix) for suffix in _SOURCE_HOST_SUFFIXES
    ):
        return None
    try:
        ipaddress.ip_address(host)
        return None
    except ValueError:
        pass
    try:
        import requests
        response = requests.get(
            url, timeout=3, allow_redirects=False, stream=True,
            headers={"User-Agent": "DailyStockAnalysis/1.0 (publication verification)"},
        )
        if response.status_code != 200 or "html" not in response.headers.get("Content-Type", "").lower():
            response.close()
            return None
        chunks: list[bytes] = []
        used = 0
        for chunk in response.iter_content(chunk_size=16384):
            chunks.append(chunk)
            used += len(chunk)
            if used >= 262144:
                break
        response.close()
        html = b"".join(chunks).decode("utf-8", errors="replace")
        parser = _PublicationMetaParser()
        parser.feed(html)
        return parser.published
    except Exception:
        return None


def audit_news_result(item: Any, cutoff: datetime, *, verify_source: bool = False) -> dict[str, Any]:
    raw_date = str(getattr(item, "published_date", None) or "").strip()
    url = str(getattr(item, "url", None) or "").strip()
    source = str(getattr(item, "source", None) or "").strip()
    audit = {
        "title": str(getattr(item, "title", None) or "").strip(),
        "url": url,
        "source": source,
        "published_date_raw": raw_date,
        "retrieved_at": getattr(item, "retrieved_at", None),
        "status": "unverified",
        "reason": "",
    }
    try:
        parsed_url = urlparse(url)
        valid_url = parsed_url.scheme in {"http", "https"} and bool(parsed_url.hostname)
    except ValueError:
        valid_url = False
    if not valid_url or not source:
        audit["reason"] = "来源URL或来源名称缺失"
        return audit
    retrieved = audit["retrieved_at"]
    if retrieved:
        try:
            retrieved_time = datetime.fromisoformat(str(retrieved).replace("Z", "+00:00"))
        except ValueError:
            retrieved_time = None
        if retrieved_time is not None and retrieved_time.tzinfo is not None:
            if retrieved_time.astimezone(timezone.utc) > cutoff.astimezone(timezone.utc):
                audit["reason"] = "检索获取时间晚于本轮截止"
                return audit
    if verify_source:
        source_date = _source_publication(url)
        if source_date:
            try:
                source_time = datetime.fromisoformat(source_date.replace("Z", "+00:00"))
            except ValueError:
                source_time = None
            if source_time is not None and source_time.tzinfo is not None:
                if source_time.astimezone(timezone.utc) > cutoff.astimezone(timezone.utc):
                    audit["reason"] = "来源页面显示发布时间晚于本轮盘前截止"
                    return audit
                audit["status"] = "source_verified"
                audit["published_at_utc"] = source_time.astimezone(timezone.utc).isoformat()
                audit["reason"] = "来源页面发布时间已核；原文具体事实仍待复核"
                audit["timestamp_corrected"] = raw_date != source_date
                return audit
    if not raw_date:
        audit["reason"] = "发布时间缺失"
        return audit
    normalized = raw_date.replace("Z", "+00:00")
    try:
        published_at = datetime.fromisoformat(normalized)
    except ValueError:
        audit["reason"] = "发布时间格式不可核验"
        return audit
    if published_at.tzinfo is None:
        if len(raw_date) == 10:
            # Only an earlier civil date is unambiguously before a UTC cutoff.
            if published_at.date() < cutoff.astimezone(timezone.utc).date():
                audit["status"] = "partial"
                audit["reason"] = "仅有早于运行日的发布日期，缺具体时间和时区"
                return audit
            audit["reason"] = "运行日仅有日期，无法证明早于盘前截止时刻"
            return audit
        audit["reason"] = "发布时间缺少时区"
        return audit
    if published_at.astimezone(timezone.utc) > cutoff.astimezone(timezone.utc):
        audit["reason"] = "发布时间晚于本轮盘前截止"
        return audit
    audit["status"] = "metadata_verified"
    audit["published_at_utc"] = published_at.astimezone(timezone.utc).isoformat()
    audit["reason"] = "仅搜索提供方的时间元数据可核；来源页面未核到发布时间"
    return audit


def audit_financial_context(context: Any, cutoff: datetime) -> dict[str, Any]:
    report = context.get("financial_report") if isinstance(context, dict) else None
    if not isinstance(report, dict) or not report:
        return {"status": "unavailable", "reason": "无财务数据"}
    publication = report.get("published_at") or report.get("announcement_at")
    source_url = report.get("source_url") or report.get("filing_url")
    if not publication or not source_url:
        return {
            "status": "unverified",
            "period": report.get("report_date"),
            "reason": "财报期间不能代替披露时间；缺少披露时刻或原始来源链接",
        }
    evidence = type("FinancialEvidence", (), {
        "title": "财务披露", "published_date": publication,
        "source": "财务原始来源", "url": source_url, "retrieved_at": None,
    })()
    checked = audit_news_result(evidence, cutoff, verify_source=True)
    return {
        "status": checked["status"],
        "period": report.get("report_date"),
        "currency": report.get("currency"),
        "publication": publication,
        "source_url": source_url,
        "reason": checked["reason"],
    }


def summarize_evidence(news: list[dict[str, Any]], financial: dict[str, Any]) -> str:
    if any(item.get("status") == "unverified" for item in news) or financial.get("status") == "unverified":
        return "部分核实；未核实的具体新闻或财务断言不纳入结论"
    if news and all(item.get("status") == "source_verified" for item in news) and financial.get("status") in {"source_verified", "unavailable"}:
        return "来源页面发布时间已核；具体事实仍以原文复核为准"
    if news or financial.get("status") in {"metadata_verified", "source_verified"}:
        return "部分核实；日期或原文证据仍有缺口"
    return "未取得可核实的盘前新闻或财务出处"
