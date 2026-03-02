from __future__ import annotations

import argparse
import json
from pathlib import Path

from cli_bench.__main__ import run_compare
from cli_bench.config import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    SuiteMetadata,
    TimingResult,
)
from cli_bench.results import save_suite


def _timing(times_ms: list[float]) -> TimingResult:
    return TimingResult(
        mean_ms=sum(times_ms) / len(times_ms),
        stddev_ms=0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        runs=len(times_ms),
        times_ms=times_ms,
    )


def _result(command: str, times_ms: list[float]) -> BenchmarkResult:
    return BenchmarkResult(
        command=command,
        command_args=command.split(),
        category="startup",
        warm_cache=_timing(times_ms),
    )


def _suite(path: Path, sha: str, results: list[BenchmarkResult]) -> None:
    suite = BenchmarkSuite(
        metadata=SuiteMetadata(
            timestamp="2026-01-01T00:00:00+00:00",
            git_sha=sha,
            git_branch="main",
            python_version="3.12.0",
            platform="Linux/x86_64",
            target_version="1.0.0",
            hyperfine_version="1.20.0",
        ),
        results=results,
        config=BenchmarkConfig(runs=5),
    )
    save_suite(suite, path)


def test_compare_writes_summary_and_digest_files(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    head = tmp_path / "head.json"
    summary_md = tmp_path / "compare.md"
    digest_json = tmp_path / "digest.json"

    _suite(
        baseline,
        "base123",
        [
            _result("tool --slow", [100.0, 101.0, 99.0, 100.5, 99.5]),
            _result("tool --fast", [100.0, 101.0, 99.0, 100.5, 99.5]),
        ],
    )
    _suite(
        head,
        "head123",
        [
            _result("tool --slow", [130.0, 131.0, 129.0, 132.0, 128.0]),
            _result("tool --fast", [80.0, 81.0, 79.0, 82.0, 78.0]),
        ],
    )

    args = argparse.Namespace(
        baseline=baseline,
        head=head,
        threshold=15.0,
        fail_on_regression=False,
        summary_md=summary_md,
        digest_json=digest_json,
    )

    exit_code = run_compare(args)
    assert exit_code == 0
    assert summary_md.exists()
    assert digest_json.exists()

    summary_text = summary_md.read_text()
    assert "CLI Benchmark Results" in summary_text
    assert "tool --slow" in summary_text

    digest = json.loads(digest_json.read_text())
    assert digest["baseline_sha"] == "base123"
    assert digest["head_sha"] == "head123"
    assert digest["threshold_percent"] == 15.0
    assert len(digest["significant_regressions"]) == 1
    assert len(digest["significant_improvements"]) == 1
    assert digest["significant_regressions"][0]["command"] == "tool --slow"
    assert digest["significant_improvements"][0]["command"] == "tool --fast"


def test_compare_fails_when_requested_for_regressions(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    head = tmp_path / "head.json"

    _suite(
        baseline,
        "base123",
        [_result("tool --slow", [100.0, 101.0, 99.0, 100.5, 99.5])],
    )
    _suite(
        head,
        "head123",
        [_result("tool --slow", [130.0, 131.0, 129.0, 132.0, 128.0])],
    )

    args = argparse.Namespace(
        baseline=baseline,
        head=head,
        threshold=15.0,
        fail_on_regression=True,
        summary_md=None,
        digest_json=None,
    )

    assert run_compare(args) == 1
