import json
import mimetypes
import threading
import time
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import respx
from click.testing import CliRunner

from oikb.client import OikbClient
from oikb.connectors import BaseConnector, ManifestEntry
from oikb.kb_sync import group_entries_by_kb, run_entries_sync
from oikb.sync import SyncCancelled, SyncResult


class Source(BaseConnector):
    def __init__(self, content, fail=False):
        self.content = content
        self.fail = fail
        self.closed = False

    def build_manifest(self):
        if self.fail:
            raise RuntimeError("scan failed")
        return [ManifestEntry(name, "", str(len(data)), len(data)) for name, data in self.content.items()]

    def read_file(self, path, filename):
        assert path == ""  # Destination prefixes must not leak into source reads.
        return self.content[filename]

    def close(self):
        self.closed = True


def test_combined_manifest_filters_routing_and_auth():
    sources = [Source({"readme.txt": b"first", "skip.bin": b"ignored"}), Source({"readme.txt": b"second"})]
    resolver = Mock(side_effect=sources)
    client = Mock()
    client.sync_diff.side_effect = lambda kb, manifest: {"added": manifest}
    entries = [
        {"source": "confluence:ENG", "kb-id": "kb", "filter": {"include": ["*.txt"]}, "auth": {"token": "one"}},
        {"source": "confluence:HR", "kb-id": "kb", "auth": {"token": "two"}},
    ]
    result = run_entries_sync(client, entries, resolve_connector=resolver, quiet=True)
    assert result.added == 2
    assert [e["path"] for e in client.sync_diff.call_args.args[1]] == ["ENG", "HR"]
    assert [call.kwargs["file_content"] for call in client.upload_file.call_args_list] == [b"first", b"second"]
    assert [call.kwargs["auth"] for call in resolver.call_args_list] == [{"token": "one"}, {"token": "two"}]
    client.sync_diff.assert_called_once()
    client.sync_cleanup.assert_not_called()
    assert all(source.closed for source in sources)


@pytest.mark.parametrize("failure", ["scan", "resolve", "duplicate", "filter"])
def test_failure_prevents_any_kb_mutation_and_closes_sources(failure):
    first = Source({"a.txt": b"a"})
    second = Source({"a.txt" if failure == "duplicate" else "b.txt": b"b"}, fail=failure == "scan")
    resolver = Mock(side_effect=[first, RuntimeError("resolve failed") if failure == "resolve" else second])
    entries = [{"source": "one", "kb-id": "kb"}, {"source": "two", "kb-id": "kb"}]
    if failure == "filter":
        entries[1]["filter"] = {"max-size": "bad"}
    client = Mock()
    with pytest.raises((ValueError, RuntimeError)):
        run_entries_sync(client, entries, resolve_connector=resolver, quiet=True)
    assert not client.mock_calls
    assert first.closed
    if failure not in {"resolve", "filter"}:
        assert second.closed


def test_single_source_keeps_paths_and_target_path_is_optional():
    for extra, expected in [({}, ""), ({"target-path": "docs/api"}, "docs/api")]:
        client = Mock()
        client.sync_diff.return_value = {}
        source = Source({"a.txt": b"a"})
        run_entries_sync(client, [{"source": "confluence:ENG", "kb-id": "kb", **extra}], resolve_connector=lambda *a, **kw: source, quiet=True)
        assert client.sync_diff.call_args.args[1][0]["path"] == expected
        assert source.closed


@pytest.mark.parametrize("prefix", ["/absolute", "../parent", "a/../b", "a\\b", "a//b"])
def test_invalid_target_path_fails_before_scan(prefix):
    resolver = Mock()
    with pytest.raises(ValueError, match="target-path"):
        run_entries_sync(Mock(), [{"source": "one", "kb-id": "kb", "target-path": prefix}], resolve_connector=resolver)
    resolver.assert_not_called()


def test_group_validation_and_cancellation():
    with pytest.raises(ValueError, match="source and kb-id"):
        group_entries_by_kb([{"source": "one"}])
    with pytest.raises(ValueError, match="same url and token"):
        group_entries_by_kb([{"source": "one", "kb-id": "kb", "url": "a"}, {"source": "two", "kb-id": "kb", "url": "b"}])
    client, resolver = Mock(), Mock()
    with pytest.raises(SyncCancelled):
        run_entries_sync(client, [{"source": "one", "kb-id": "kb"}], resolve_connector=resolver, cancel_requested=lambda: True)
    assert not client.mock_calls and not resolver.mock_calls


def test_cli_name_selects_entire_kb_group_and_closes_client(monkeypatch):
    import oikb.cli as cli
    import oikb.kb_sync as kb_sync
    entries = [{"name": "one", "source": "one", "kb-id": "kb"}, {"source": "two", "kb-id": "kb"}, {"source": "other", "kb-id": "other"}]
    monkeypatch.setattr(cli, "_load_oikb_yaml", lambda: entries)
    client = Mock()
    monkeypatch.setattr(cli, "_make_client", lambda *a: client)
    sync = Mock(return_value=SyncResult())
    monkeypatch.setattr(kb_sync, "run_entries_sync", sync)
    result = CliRunner().invoke(cli.cli, ["sync", "--name", "one", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert sync.call_args.args[1] == entries[:2]
    assert sync.call_args.kwargs["dry_run"] is True
    client.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["webhook", "alias", "kb"])
async def test_daemon_triggers_include_sibling_sources(monkeypatch, trigger):
    import oikb.cli as cli
    import oikb.daemon as daemon
    import oikb.kb_sync as kb_sync
    entries = [{"name": "one", "source": "one", "kb-id": "kb"}, {"source": "two", "kb-id": "kb"}]
    monkeypatch.setattr(daemon, "_entries", entries)
    monkeypatch.setattr(daemon, "_sync_locks", {})
    monkeypatch.setattr(daemon, "_scheduler_state", {})
    monkeypatch.setattr(daemon, "_shutdown_event", None)
    monkeypatch.setattr(daemon, "_history", None)
    monkeypatch.setattr(daemon, "_send_notification", AsyncMock())
    monkeypatch.setattr(daemon, "record_sync", Mock())
    client = Mock()
    monkeypatch.setattr(cli, "_make_client", lambda **kw: client)
    sync = Mock(return_value=SyncResult(added=2))
    monkeypatch.setattr(kb_sync, "run_entries_sync", sync)
    if trigger == "webhook":
        await daemon._run_entry(entries[0])  # The same callback registered with webhooks.
        assert all(daemon._scheduler_state[s]["files_added"] == 2 for s in ("one", "two"))
    else:
        result = await daemon.trigger_sync("one" if trigger == "alias" else "kb", dry_run=True)
        assert result["result"]["added"] == 2
        assert daemon._scheduler_state == {}
    assert sync.call_args.args[1] == entries
    client.close.assert_called_once()


def test_upload_error_includes_server_detail():
    client = Mock()
    client.sync_diff.side_effect = lambda kb, manifest: {"added": manifest}
    response = httpx.Response(400, json={"detail": "extraction failed"}, request=httpx.Request("POST", "https://webui.example/files"))
    client.upload_file.side_effect = httpx.HTTPStatusError("Bad Request", request=response.request, response=response)
    result = run_entries_sync(client, [{"source": "one", "kb-id": "kb"}], resolve_connector=lambda *a, **kw: Source({"a.txt": b"a"}), quiet=True)
    assert "extraction failed" in result.errors[0]


class MetadataSource(Source):
    def build_manifest(self):
        return [
            ManifestEntry("a.txt", "", "h", 1, metadata={"doc_date": "2026-03-14", "knowledge_id": "spoofed"}),
            ManifestEntry("b.txt", "", "h", 1),
        ]


@respx.mock
def test_manifest_metadata_reaches_upload_without_overriding_reserved_keys():
    respx.post("https://webui.example/api/v1/knowledge/kb/sync/diff").mock(
        side_effect=lambda request: httpx.Response(200, json={"added": json.loads(request.content)["manifest"]})
    )
    uploads = respx.post("https://webui.example/api/v1/files/").mock(return_value=httpx.Response(200, json={"id": "f"}))
    source = MetadataSource({"a.txt": b"a", "b.txt": b"b"})
    run_entries_sync(OikbClient("https://webui.example", "token"), [{"source": "one", "kb-id": "kb"}], resolve_connector=lambda *a, **kw: source, quiet=True)
    sent = {}
    for call in uploads.calls:
        body = call.request.content.decode()
        name = "a.txt" if 'filename="a.txt"' in body else "b.txt"
        sent[name] = json.loads(body.split('name="metadata"\r\n\r\n', 1)[1].split("\r\n--", 1)[0])
    assert sent["a.txt"] == {"doc_date": "2026-03-14", "knowledge_id": "kb", "file_hash": "h"}
    assert sent["b.txt"] == {"knowledge_id": "kb", "file_hash": "h"}


def test_manifest_metadata_is_not_sent_for_diffing():
    assert "metadata" not in ManifestEntry("a.txt", "", "h", 1, metadata={"k": "v"}).to_dict()


@respx.mock
def test_upload_waits_for_processing_with_long_read_timeout():
    route = respx.post("https://webui.example/api/v1/files/").mock(return_value=httpx.Response(200, json={"id": "f"}))
    OikbClient("https://webui.example", "token").upload_file(b"a", "a.txt", "kb", "hash")
    request = route.calls.last.request
    assert request.url.params["process_in_background"] == "false"
    assert request.extensions["timeout"]["read"] == 900.0


def _slow_upload_client(seconds):
    client = Mock()
    client.sync_diff.side_effect = lambda kb, manifest: {"added": manifest}
    client.upload_file.side_effect = lambda **kw: time.sleep(seconds) or {"id": "f"}
    client.get_file.return_value = {"id": "f", "data": {"status": "completed"}}
    return client


def _timed_sync(client, cancel_requested=None, files=None, **entry):
    started = time.monotonic()
    result = run_entries_sync(
        client, [{"source": "one", "kb-id": "kb", **entry}],
        resolve_connector=lambda *a, **kw: Source(files or {"a.txt": b"a"}), quiet=True,
        cancel_requested=cancel_requested,
    )
    return result, time.monotonic() - started


def test_upload_duty_idles_in_proportion_to_upload_time():
    result, elapsed = _timed_sync(_slow_upload_client(0.3), **{"upload-duty": 0.5})
    assert result.added == 1
    assert 0.55 <= elapsed < 1.2


def test_upload_duty_applies_to_the_whole_kb_across_workers():
    # Two workers at duty 0.5: each idles 0.2 * (2 / 0.5 - 1) = 0.6 s after its 0.2 s upload.
    result, elapsed = _timed_sync(
        _slow_upload_client(0.2), files={"a.txt": b"a", "b.txt": b"b"},
        concurrency=2, **{"upload-duty": 0.5},
    )
    assert result.added == 2
    assert 0.75 <= elapsed < 1.5


def test_default_upload_duty_never_idles_with_concurrency():
    result, elapsed = _timed_sync(
        _slow_upload_client(0.05), files={f"{i}.txt": b"x" for i in range(8)},
        concurrency=2,
    )
    assert result.added == 8
    assert elapsed < 0.3


def test_read_timeout_still_idles_before_reporting_the_error():
    client = _slow_upload_client(0)

    def timeout(**kw):
        time.sleep(0.3)
        raise httpx.ReadTimeout("read timed out")

    client.upload_file.side_effect = timeout
    result, elapsed = _timed_sync(client, **{"upload-duty": 0.5})
    client.upload_file.assert_called_once()
    assert result.added == 0 and "read timed out" in result.errors[0]
    assert elapsed >= 0.55


def test_full_upload_duty_does_not_idle():
    result, elapsed = _timed_sync(_slow_upload_client(0.3), **{"upload-duty": 1.0})
    assert result.added == 1
    assert elapsed < 0.5


def test_cancel_interrupts_upload_duty_idle():
    client = _slow_upload_client(0.1)
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    started = time.monotonic()
    with pytest.raises(SyncCancelled):
        _timed_sync(client, cancel_requested=stop.is_set, **{"upload-duty": 0.01})
    assert time.monotonic() - started < 0.3 + 0.5


@pytest.mark.parametrize("duty", [0, -0.5, 1.5, "0.5", True])
def test_invalid_upload_duty_rejected_before_any_kb_call(duty):
    client, resolver = Mock(), Mock()
    with pytest.raises(ValueError, match="upload-duty"):
        run_entries_sync(client, [{"source": "one", "kb-id": "kb", "upload-duty": duty}], resolve_connector=resolver)
    assert not client.mock_calls and not resolver.mock_calls


def test_failed_processing_counts_as_pass_error():
    client = _slow_upload_client(0)
    client.get_file.return_value = {"id": "f", "data": {"status": "failed", "error": "embedding timeout"}}
    result, _ = _timed_sync(client)
    client.get_file.assert_called_once_with("f")
    client.delete_file.assert_called_once_with("f")
    assert result.added == 0
    assert len(result.errors) == 1 and "embedding timeout" in result.errors[0]


def test_failed_file_is_retried_once_per_pass():
    client = _slow_upload_client(0)
    client.get_file.return_value = {"id": "f", "data": {"status": "failed"}}
    for expected_uploads in (1, 2):
        result, _ = _timed_sync(client)
        assert client.upload_file.call_count == expected_uploads
        assert len(result.errors) == 1


def test_orphan_delete_failure_is_reported_in_the_error():
    client = _slow_upload_client(0)
    client.get_file.return_value = {"id": "f", "data": {"status": "failed", "error": "bad pdf"}}
    client.delete_file.side_effect = RuntimeError("boom")
    result, _ = _timed_sync(client)
    assert "bad pdf" in result.errors[0] and "orphan file f not deleted: boom" in result.errors[0]
    client.upload_file.assert_called_once()


@respx.mock
def test_delete_file_calls_files_endpoint():
    route = respx.delete("https://webui.example/api/v1/files/f").mock(return_value=httpx.Response(200, json={}))
    OikbClient("https://webui.example", "token").delete_file("f")
    assert route.called


def test_failed_status_check_does_not_reupload():
    client = _slow_upload_client(0)
    response = httpx.Response(503, request=httpx.Request("GET", "https://webui.example/files/f"))
    client.get_file.side_effect = httpx.HTTPStatusError("Unavailable", request=response.request, response=response)
    result, _ = _timed_sync(client)
    client.upload_file.assert_called_once()
    assert result.added == 0 and "status check failed" in result.errors[0]


@respx.mock
def test_upload_sends_markdown_as_text_without_system_mime_types(monkeypatch):
    monkeypatch.setattr(mimetypes, "guess_type", lambda *a, **kw: (None, None))
    route = respx.post("https://webui.example/api/v1/files/").mock(return_value=httpx.Response(200, json={"id": "f"}))
    OikbClient("https://webui.example", "token").upload_file(b"| a |\n| - |\n", "doc.md", "kb", "hash")
    assert b"Content-Type: text/markdown" in route.calls.last.request.content
