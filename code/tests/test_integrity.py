import json

import pytest

from rec.data import download
from rec.io import file_hash
from rec.verification import verify_evidence


def test_corrupt_existing_archive_is_rejected_without_silent_reuse(tmp_path):
    (tmp_path / "data_original.zip").write_bytes(b"truncated archive")
    with pytest.raises(ValueError, match="pinned archive"):
        download(tmp_path)


def test_modified_published_artifact_fails_verification(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    artifact = evidence / "results.json"
    artifact.write_text('{"score": 0.1}', encoding="utf-8")
    (evidence / "checksums.json").write_text(
        json.dumps({"results.json": file_hash(artifact)}), encoding="utf-8"
    )
    artifact.write_text('{"score": 0.9}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_evidence(tmp_path)
