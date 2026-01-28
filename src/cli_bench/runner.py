"""benchmark runner using hyperfine for timing."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import BenchmarkCommand, BenchmarkConfig, BenchmarkResult, ProjectConfig, TimingResult


def check_hyperfine() -> str | None:
    """check if hyperfine is installed and return version.

    Returns:
        Version string if installed, None otherwise.
    """
    try:
        result = subprocess.run(
            ["hyperfine", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip().split()[-1]
        return None
    except FileNotFoundError:
        return None


def get_hyperfine_install_instructions() -> str:
    """get platform-specific installation instructions for hyperfine."""
    system = platform.system()
    if system == "Darwin":
        return "brew install hyperfine"
    elif system == "Linux":
        return (
            "# Ubuntu/Debian:\n"
            "wget https://github.com/sharkdp/hyperfine/releases/download/v1.18.0/hyperfine_1.18.0_amd64.deb\n"
            "sudo dpkg -i hyperfine_1.18.0_amd64.deb\n\n"
            "# Or with cargo:\n"
            "cargo install hyperfine"
        )
    else:
        return "See https://github.com/sharkdp/hyperfine#installation"


def get_cache_clear_command(project: ProjectConfig) -> str:
    """get command to clear caches between cold runs.

    This clears Python's __pycache__ directories for the target project.
    """
    project_root = project.project_root

    # clear __pycache__ in the project directory
    pycache_clear = (
        f"find {project_root} -type d -name __pycache__ "
        f"-exec rm -rf {{}} + 2>/dev/null || true"
    )

    system = platform.system()
    if system == "Darwin":
        return f"{pycache_clear}; sync"
    elif system == "Linux":
        return (
            f"{pycache_clear}; sync; "
            f"{{ echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; }}"
        )
    else:
        return pycache_clear


class BenchmarkRunner:
    """runs benchmarks using hyperfine."""

    def __init__(self, config: BenchmarkConfig, project: ProjectConfig):
        self.config = config
        self.project = project
        self._hyperfine_version = check_hyperfine()

    @property
    def hyperfine_available(self) -> bool:
        return self._hyperfine_version is not None

    @property
    def hyperfine_version(self) -> str:
        return self._hyperfine_version or "not installed"

    def run_benchmark(self, cmd: BenchmarkCommand) -> BenchmarkResult:
        """run both cold and warm benchmarks for a command.

        Args:
            cmd: The command to benchmark.

        Returns:
            BenchmarkResult with timing data.
        """
        result = BenchmarkResult(
            command=cmd.name,
            command_args=cmd.args,
            category=cmd.category,
        )

        # check if server is required but not configured
        if cmd.requires_server and not self.config.server_url:
            api_var = self.project.api_url_env_var or "API_URL"
            result.error = f"requires server (use --server-url or set {api_var})"
            return result

        try:
            if not self.config.skip_cold:
                result.cold_start = self._run_cold_start(cmd)

            if not self.config.skip_warm:
                result.warm_cache = self._run_warm_cache(cmd)

        except subprocess.TimeoutExpired:
            result.error = f"timeout after {self.config.timeout_seconds}s"
        except subprocess.CalledProcessError as e:
            result.error = f"command failed with exit code {e.returncode}"
        except Exception as e:
            result.error = str(e)

        return result

    def _run_cold_start(self, cmd: BenchmarkCommand) -> TimingResult:
        """run cold start benchmark (clearing caches between runs)."""
        return self._run_hyperfine(
            cmd,
            warmup=0,
            prepare=get_cache_clear_command(self.project),
        )

    def _run_warm_cache(self, cmd: BenchmarkCommand) -> TimingResult:
        """run warm cache benchmark (with warmup runs)."""
        return self._run_hyperfine(
            cmd,
            warmup=self.config.warmup_runs,
            prepare=None,
        )

    def _run_hyperfine(
        self,
        cmd: BenchmarkCommand,
        warmup: int,
        prepare: str | None,
    ) -> TimingResult:
        """run hyperfine and parse results.

        Args:
            cmd: Command to benchmark.
            warmup: Number of warmup runs.
            prepare: Command to run before each benchmark run.

        Returns:
            TimingResult with statistics.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_output = Path(f.name)

        try:
            hyperfine_args = [
                "hyperfine",
                "--warmup",
                str(warmup),
                "--runs",
                str(self.config.runs),
                "--export-json",
                str(json_output),
                "--time-unit",
                "millisecond",
            ]

            if prepare:
                hyperfine_args.extend(["--prepare", prepare])

            # build the command to run in the project context
            project_root = self.project.project_root
            # use uv run to execute in the project's venv
            full_command = f"uv run --directory {project_root} {' '.join(cmd.args)}"
            hyperfine_args.append(full_command)

            # set up environment for API commands
            env = None
            if cmd.requires_server and self.config.server_url:
                env = os.environ.copy()
                api_var = self.project.api_url_env_var or "API_URL"
                env[api_var] = self.config.server_url

            if self.config.verbose:
                print(f"  Running: {' '.join(hyperfine_args)}", file=sys.stderr)

            subprocess.run(
                hyperfine_args,
                capture_output=not self.config.verbose,
                timeout=self.config.timeout_seconds * (self.config.runs + warmup + 5),
                check=True,
                env=env,
            )

            # parse results
            with open(json_output) as f:
                data = json.load(f)

            result = data["results"][0]
            stddev = result.get("stddev")
            raw_times_ms = [t * 1000 for t in result["times"]]
            return TimingResult(
                mean_ms=result["mean"] * 1000,
                stddev_ms=stddev * 1000 if stddev is not None else 0.0,
                min_ms=result["min"] * 1000,
                max_ms=result["max"] * 1000,
                runs=len(raw_times_ms),
                times_ms=raw_times_ms,
            )

        finally:
            json_output.unlink(missing_ok=True)
