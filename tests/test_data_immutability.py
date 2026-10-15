"""Frozen data releases: checksums are recorded and never mutated by re-export."""

from __future__ import annotations

import pytest

from algo_agent.paths import TaskPaths, read_json, sha256_file
from algo_agent.reuse import export_reproduction


def _release_manifests(settings, task_id, repo):
    spec = repo.get_spec(task_id)
    return sorted((TaskPaths(settings, spec.task_slug).data / "releases").glob("*/manifest.json"))


def test_demo_produces_frozen_releases(settings, demo_report, demo_repo):
    manifests = _release_manifests(settings, "demo_task", demo_repo)
    assert manifests, "the demo must write at least one data release"
    for path in manifests:
        manifest = read_json(path)
        assert isinstance(manifest.get("seed"), int)
        assert manifest["checksums"], "a release must freeze its artifact checksums"


def test_recorded_checksums_match_the_release_artifacts(settings, demo_report, demo_repo):
    for path in _release_manifests(settings, "demo_task", demo_repo):
        manifest = read_json(path)
        for name, want in manifest["checksums"].items():
            got = sha256_file(path.parent / name)
            assert got == want, f"{path.parent.name}/{name} drifted from its frozen checksum"


def test_reproduction_export_does_not_mutate_releases(settings, demo_report, demo_repo):
    manifests = _release_manifests(settings, "demo_task", demo_repo)
    before = {p: p.read_bytes() for p in manifests}
    export_reproduction(settings, demo_repo, "demo_task")
    after = {p: p.read_bytes() for p in manifests}
    assert before == after


def test_same_seed_reproduces_the_same_release_fingerprint(settings, demo_report, demo_repo):
    """A re-run with the frozen seed must reproduce byte-identical artifacts."""
    manifests = _release_manifests(settings, "demo_task", demo_repo)
    for path in manifests:
        manifest = read_json(path)
        recomputed = {name: sha256_file(path.parent / name) for name in manifest["checksums"]}
        assert recomputed == manifest["checksums"]
