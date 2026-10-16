"""Opt-in literature discovery with bounded responses and explicit provenance."""

from __future__ import annotations

import json
import re
import time
from urllib.parse import urlsplit
from xml.etree import ElementTree

from ..schemas import contracts as C
from .base import SearchResult

ARXIV_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
MAX_RESPONSE_BYTES = 512_000


def search_backend(settings) -> str:
    configured = settings.env_get("ALGO_SEARCH_BACKEND")
    if configured:
        return configured.strip().lower()
    # Keep existing private/enterprise search integrations working.
    if settings.env_get("ALGO_SEARCH_URL"):
        return "http"
    return str((settings.workflow.get("search") or {}).get("backend", "disabled")).lower()


def _public_url(value: str) -> bool:
    url = urlsplit(value)
    return url.scheme in {"http", "https"} and bool(url.hostname) and not (url.username or url.password)


def parse_arxiv(content: bytes, limit: int = 8) -> list[dict]:
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("search response exceeds size limit")
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("XML declarations are not supported")
    root = ElementTree.fromstring(content)
    if root.tag != ATOM + "feed":
        raise ValueError("expected an Atom feed")
    papers = []
    for entry in root.findall(ATOM + "entry")[:limit]:
        uri = (entry.findtext(ATOM + "id") or "").strip()
        if "api/errors" in uri:
            raise ValueError("arXiv rejected the query")
        parsed = urlsplit(uri)
        if parsed.hostname not in {"arxiv.org", "export.arxiv.org"} or not parsed.path.startswith("/abs/"):
            continue
        papers.append({
            "url": "https://arxiv.org" + parsed.path,
            "title": " ".join((entry.findtext(ATOM + "title") or "").split())[:500],
            "summary": " ".join((entry.findtext(ATOM + "summary") or "").split())[:3500],
            "published": entry.findtext(ATOM + "published") or "",
            "updated": entry.findtext(ATOM + "updated") or "",
            "authors": [(a.findtext(ATOM + "name") or "")[:150]
                        for a in entry.findall(ATOM + "author")[:12]],
            "evidence_level": "abstract_only_not_reproduced",
        })
    return papers


def _fetch(url, params, timeout):
    import httpx
    chunks, size = [], 0
    with httpx.stream("GET", url, params=params, timeout=timeout, follow_redirects=False,
                      headers={"User-Agent": "algo-agent/literature-research",
                               "Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("search response exceeds size limit")
            chunks.append(chunk)
    return b"".join(chunks)


def search_sources(settings, query: str, budget: C.Budget) -> SearchResult:
    backend = search_backend(settings)
    if backend == "disabled":
        return SearchResult(note="live search disabled; configure an authorised search backend")
    if backend not in {"http", "arxiv"}:
        return SearchResult(note=f"unsupported search backend: {backend}")
    query = query.strip()[:2000]
    timeout = max(1, min(20, budget.max_wall_clock_s))
    try:
        if backend == "arxiv":
            # Share the API's >=3 second spacing across workers of this platform.
            import fcntl
            lock_path = settings.state_dir / "arxiv.rate"
            with lock_path.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                lock.seek(0)
                last = float(lock.read() or "0")
                time.sleep(max(0, 3 - (time.time() - last)))
                lock.seek(0)
                lock.truncate()
                lock.write(str(time.time()))
                lock.flush()
                # A literal keyword query prevents accidental arXiv query operators.
                words = re.findall(r"[\w-]+", query)[:20]
                if not words:
                    raise ValueError("search needs technical keywords")
                cfg = settings.workflow.get("search") or {}
                sort = cfg.get("sort_by", "relevance")
                if sort not in {"relevance", "lastUpdatedDate", "submittedDate"}:
                    raise ValueError("invalid arXiv sort_by")
                content = _fetch(ARXIV_URL, {
                    "search_query": " AND ".join('all:"' + word + '"' for word in words),
                    "start": 0, "max_results": 8, "sortBy": sort, "sortOrder": "descending"}, timeout)
            documents = parse_arxiv(content)
        else:
            url = settings.env_get("ALGO_SEARCH_URL") or ""
            if not _public_url(url):
                raise ValueError("ALGO_SEARCH_URL must be a configured HTTP endpoint without userinfo")
            items = json.loads(_fetch(url, {"q": query}, timeout)).get("results", [])
            documents = [{"url": str(it.get("url", "")), "title": str(it.get("title", ""))[:500],
                          "summary": str(it.get("summary", it.get("snippet", "")))[:3500],
                          "evidence_level": "search_metadata_not_reproduced"}
                         for it in items[:8] if isinstance(it, dict) and _public_url(str(it.get("url", "")))]
        refs = [C.EvidenceRef(
            evidence_id=f"src_{i}", uri=d["url"], locator=d["title"],
            source_type="paper" if backend == "arxiv" else "web",
            version=d.get("updated", ""), upload_policy="open", trust="medium")
                for i, d in enumerate(documents)]
        return SearchResult(refs=refs, documents=documents, used_search_service=True,
                            note=f"{backend}: {len(refs)} results; metadata/abstracts only, not reproduction")
    except Exception as exc:
        # Error bodies/URLs from third-party services may contain credentials.
        return SearchResult(note=f"{backend} search failed ({type(exc).__name__}); no evidence obtained",
                            used_search_service=True)
