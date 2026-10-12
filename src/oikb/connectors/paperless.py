"""Paperless-ngx connector - sync document text from a Paperless-ngx instance.

Auth via PAPERLESS_TOKEN env var (a Paperless API token). The base URL comes
from the source string or the PAPERLESS_URL env var.

Source syntax:
    paperless:https://paperless.example.com
    paperless:https://paperless.example.com?tags=invoice,contract&document_types=Bill
    paperless:                              # base URL from PAPERLESS_URL
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx

from oikb.connectors import BaseConnector, ManifestEntry, SourceFileUnavailable

_ACCEPT = "application/json; version=10"
_NO_CORRESPONDENT = "_no_correspondent"
_UNSAFE_CHARS = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]+')


class PaperlessConnector(BaseConnector):
    """Sync Paperless-ngx documents as .txt files.

    Args:
        base_url:       Paperless base URL, or PAPERLESS_URL env var.
        token:          API token, or PAPERLESS_TOKEN env var.
        tags:           Only documents carrying at least one of these tag names.
        document_types: Only documents of one of these document type names.
        page_size:      Page size for list requests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        tags: list[str] | None = None,
        document_types: list[str] | None = None,
        page_size: int = 100,
    ):
        self._base_url = (base_url or os.environ.get("PAPERLESS_URL", "")).rstrip("/")
        token = token or os.environ.get("PAPERLESS_TOKEN", "")
        if not self._base_url:
            raise ValueError("Paperless URL required. Use paperless:<url> or set PAPERLESS_URL.")
        if not token:
            raise ValueError("Paperless token required. Set PAPERLESS_TOKEN env var.")
        base = urlsplit(self._base_url)
        if base.scheme not in ("http", "https") or not base.netloc:
            raise ValueError(f"Paperless URL must be http(s)://host, got {self._base_url!r}.")
        self._origin = (base.scheme, base.netloc)
        if not 1 <= page_size <= 1000:
            raise ValueError(f"Paperless page_size must be between 1 and 1000, got {page_size}.")
        self._tags = {t.casefold() for t in tags or []}
        self._document_types = {t.casefold() for t in document_types or []}
        self._page_size = page_size
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Token {token}",
                "Accept": _ACCEPT,
                "User-Agent": "oikb/paperless",
            },
            timeout=60.0,
        )

    def build_manifest(self) -> list[ManifestEntry]:
        correspondents = self._names("/api/correspondents/")
        document_types = self._names("/api/document_types/")
        tags = self._names("/api/tags/")

        entries: list[ManifestEntry] = []
        for doc in self._paginate("/api/documents/"):
            tag_names = [tags[t] for t in doc.get("tags") or [] if t in tags]
            doc_type = document_types.get(doc.get("document_type"))
            if self._tags and not self._tags & {t.casefold() for t in tag_names}:
                continue
            if self._document_types and (doc_type or "").casefold() not in self._document_types:
                continue

            correspondent = correspondents.get(doc.get("correspondent"))
            created = str(doc.get("created") or "")[:10]
            year = created[:4] if re.fullmatch(r"\d{4}", created[:4]) else "undated"
            text = doc.get("content") or ""
            content = text.encode()
            fingerprint = json.dumps(
                {
                    "content": text,
                    "tags": sorted(tag_names),
                    "document_type": doc_type,
                    "correspondent": correspondent,
                    "created": created,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            metadata: dict[str, Any] = {
                "doc_date": created or None,
                "added": doc.get("added"),
                "modified": doc.get("modified"),
                "correspondent": correspondent,
                "document_type": doc_type,
                "tags": ", ".join(tag_names) or None,
                "paperless_id": doc["id"],
                "paperless_url": f"{self._base_url}/documents/{doc['id']}/details",
            }
            entries.append(
                ManifestEntry(
                    filename=f"{doc['id']}_{_safe_name(doc.get('title') or '', 'untitled')}.txt",
                    path=f"{_safe_name(correspondent or '', _NO_CORRESPONDENT)}/{year}",
                    checksum=hashlib.sha256(fingerprint.encode()).hexdigest(),
                    size=len(content),
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
        entries.sort(key=lambda e: e.display_path)
        return entries

    def read_file(self, path: str, filename: str) -> bytes:
        doc_id = filename.split("_", 1)[0]
        resp = self._http.get(f"/api/documents/{doc_id}/")
        if resp.status_code == 404:
            raise SourceFileUnavailable(f"Paperless document {doc_id} no longer exists")
        resp.raise_for_status()
        return (resp.json().get("content") or "").encode()

    def close(self) -> None:
        self._http.close()

    def _names(self, endpoint: str) -> dict[int, str]:
        return {item["id"]: item["name"] for item in self._paginate(endpoint)}

    def _paginate(self, endpoint: str):
        url: str | None = endpoint
        params: dict[str, Any] | None = {"page_size": self._page_size}
        while url:
            resp = self._http.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            yield from payload.get("results", [])
            url = payload.get("next")
            if url and urlsplit(url)[:2] != self._origin:
                raise ValueError(f"Paperless pagination left {self._base_url}: {url}")
            params = None


def parse_paperless_source(source: str) -> dict[str, Any]:
    """Parse a paperless:<url>?tags=..&document_types=..&page_size=.. source string."""
    rest = source.removeprefix("paperless:")
    parts = urlsplit(rest)
    query = parse_qs(parts.query)

    def _list(key: str) -> list[str] | None:
        values = [v.strip() for raw in query.get(key, []) for v in raw.split(",") if v.strip()]
        return values or None

    parsed: dict[str, Any] = {
        "base_url": urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) or None,
        "tags": _list("tags"),
        "document_types": _list("document_types"),
    }
    if query.get("page_size"):
        parsed["page_size"] = int(query["page_size"][0])
    return parsed


def _safe_name(value: str, fallback: str) -> str:
    cleaned = " ".join(_UNSAFE_CHARS.sub("-", value).split()).strip(" .-")
    return cleaned[:100] or fallback
