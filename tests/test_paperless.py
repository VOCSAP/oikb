from __future__ import annotations

import httpx
import pytest
import respx

from oikb.cli import _resolve_connector
from oikb.connectors import SourceFileUnavailable
from oikb.connectors.paperless import PaperlessConnector, parse_paperless_source

BASE = "https://paperless.example.test"


def _page(results: list[dict], next_url: str | None = None) -> dict:
    return {"count": len(results), "next": next_url, "previous": None, "results": results}


def _mock_lookups(router: respx.MockRouter) -> None:
    router.get(f"{BASE}/api/correspondents/").respond(200, json=_page([{"id": 1, "name": "ACME / Corp"}]))
    router.get(f"{BASE}/api/document_types/").respond(200, json=_page([{"id": 5, "name": "Contract"}]))
    router.get(f"{BASE}/api/tags/").respond(200, json=_page([{"id": 7, "name": "work"}, {"id": 8, "name": "tax"}]))


DOC_1 = {
    "id": 12, "title": "Service agreement: 2026", "content": "Travel costs are refunded.",
    "created": "2026-03-14", "added": "2026-04-02T10:21:00Z", "modified": "2026-04-03T08:00:00Z",
    "correspondent": 1, "document_type": 5, "tags": [7, 8],
}
DOC_2 = {
    "id": 13, "title": "", "content": "Scanned note", "created": "2025-12-01",
    "added": "2025-12-02T00:00:00Z", "modified": "2025-12-02T00:00:00Z",
    "correspondent": None, "document_type": None, "tags": [],
}


@pytest.fixture
def connector():
    c = PaperlessConnector(base_url=BASE, token="secret")
    yield c
    c.close()


@respx.mock(assert_all_called=False)
def test_manifest_follows_pages_and_carries_metadata(connector, respx_mock):
    _mock_lookups(respx_mock)
    second = f"{BASE}/api/documents/?page=2&page_size=100"
    respx_mock.get(second).respond(200, json=_page([DOC_2]))
    first = respx_mock.get(f"{BASE}/api/documents/", params={"page_size": "100"}).respond(200, json=_page([DOC_1], second))

    manifest = connector.build_manifest()

    assert [e.display_path for e in manifest] == [
        "ACME - Corp/2026/12_Service agreement- 2026.txt",
        "_no_correspondent/2025/13_untitled.txt",
    ]
    assert manifest[0].metadata == {
        "doc_date": "2026-03-14", "added": "2026-04-02T10:21:00Z", "modified": "2026-04-03T08:00:00Z",
        "correspondent": "ACME / Corp", "document_type": "Contract", "tags": "work, tax",
        "paperless_id": 12, "paperless_url": f"{BASE}/documents/12/details",
    }
    assert manifest[1].metadata == {
        "doc_date": "2025-12-01", "added": "2025-12-02T00:00:00Z", "modified": "2025-12-02T00:00:00Z",
        "paperless_id": 13, "paperless_url": f"{BASE}/documents/13/details",
    }
    assert manifest[0].size == len(DOC_1["content"].encode())
    request = first.calls.last.request
    assert request.headers["Authorization"] == "Token secret"
    assert request.headers["Accept"] == "application/json; version=10"


@respx.mock(assert_all_called=False)
def test_tag_and_document_type_filters(respx_mock):
    _mock_lookups(respx_mock)
    respx_mock.get(f"{BASE}/api/documents/").respond(200, json=_page([DOC_1, DOC_2]))
    for kwargs in ({"tags": ["TAX"]}, {"document_types": ["contract"]}):
        c = PaperlessConnector(base_url=BASE, token="secret", **kwargs)
        assert [e.metadata["paperless_id"] for e in c.build_manifest()] == [12]
        c.close()


@respx.mock(assert_all_called=False)
def test_checksum_changes_with_tags_but_not_with_tag_order(connector, respx_mock):
    _mock_lookups(respx_mock)
    docs = respx_mock.get(f"{BASE}/api/documents/")

    def checksum(tags: list[int]) -> str:
        docs.respond(200, json=_page([{**DOC_1, "tags": tags}]))
        return connector.build_manifest()[0].checksum

    assert checksum([7]) != checksum([7, 8])
    assert checksum([7, 8]) == checksum([8, 7])


@pytest.mark.parametrize("next_url", ["https://elsewhere.example/api/documents/?page=2", "http://paperless.example.test/api/documents/?page=2"])
@respx.mock(assert_all_called=False)
def test_pagination_refuses_to_leave_the_configured_origin(connector, respx_mock, next_url):
    _mock_lookups(respx_mock)
    respx_mock.get(f"{BASE}/api/documents/").respond(200, json=_page([DOC_1], next_url))
    foreign = respx_mock.get(next_url).respond(200, json=_page([]))
    with pytest.raises(ValueError, match="pagination"):
        connector.build_manifest()
    assert not foreign.called


@respx.mock(assert_all_called=False)
def test_document_without_a_valid_created_year_goes_to_undated(connector, respx_mock):
    _mock_lookups(respx_mock)
    respx_mock.get(f"{BASE}/api/documents/").respond(200, json=_page([{**DOC_1, "created": None}, {**DOC_2, "created": "n/a"}]))
    assert {e.path.split("/")[1] for e in connector.build_manifest()} == {"undated"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_url": "ftp://paperless.example.test"}, "http"),
        ({"base_url": "paperless.example.test"}, "http"),
        ({"page_size": 0}, "page_size"),
        ({"page_size": 1001}, "page_size"),
    ],
)
def test_invalid_url_or_page_size_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PaperlessConnector(**{"base_url": BASE, "token": "secret", **kwargs})


@respx.mock
def test_read_file_returns_current_content(connector, respx_mock):
    respx_mock.get(f"{BASE}/api/documents/12/").respond(200, json=DOC_1)
    assert connector.read_file("ACME - Corp/2026", "12_Service agreement- 2026.txt") == b"Travel costs are refunded."


@respx.mock
def test_deleted_document_is_unavailable(connector, respx_mock):
    respx_mock.get(f"{BASE}/api/documents/12/").respond(404, json={"detail": "No Document matches the given query."})
    with pytest.raises(SourceFileUnavailable):
        connector.read_file("ACME - Corp/2026", "12_x.txt")


@respx.mock
def test_auth_error_stops_the_scan(connector, respx_mock):
    respx_mock.get(f"{BASE}/api/correspondents/").respond(401, json={"detail": "Invalid token."})
    with pytest.raises(httpx.HTTPStatusError):
        connector.build_manifest()


def test_missing_token_or_url(monkeypatch):
    monkeypatch.delenv("PAPERLESS_TOKEN", raising=False)
    monkeypatch.delenv("PAPERLESS_URL", raising=False)
    with pytest.raises(ValueError, match="token"):
        PaperlessConnector(base_url=BASE)
    with pytest.raises(ValueError, match="URL"):
        PaperlessConnector(token="secret")


def test_source_parsing_and_cli_registration(monkeypatch):
    assert parse_paperless_source("paperless:https://p.example/sub?tags=a,b&document_types=Bill&page_size=25") == {
        "base_url": "https://p.example/sub", "tags": ["a", "b"], "document_types": ["Bill"], "page_size": 25,
    }
    assert parse_paperless_source("paperless:")["base_url"] is None
    monkeypatch.setenv("PAPERLESS_URL", BASE)
    monkeypatch.setenv("PAPERLESS_TOKEN", "secret")
    connector = _resolve_connector("paperless:")
    assert isinstance(connector, PaperlessConnector)
    connector.close()
