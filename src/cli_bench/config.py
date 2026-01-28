"""configuration and dataclasses for CLI benchmarks."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]


@dataclass
class BenchmarkCommand:
    """definition of a command to benchmark."""

    name: str
    args: list[str]
    category: str = "default"
    description: str = ""
    requires_server: bool = False
    timeout: int = 30


@dataclass
class ProjectConfig:
    """configuration loaded from bench.toml."""

    # project info
    name: str = "cli"
    project_root: Path = field(default_factory=Path.cwd)
    import_path: str = ""  # e.g., "prefect.cli" for profiling

    # commands to benchmark
    commands: list[BenchmarkCommand] = field(default_factory=list)

    # version detection
    version_module: str = ""  # e.g., "prefect" to get prefect.__version__
    version_command: list[str] | None = None  # e.g., ["prefect", "--version"]

    # optional: env vars to set for API commands
    api_url_env_var: str = ""  # e.g., "PREFECT_API_URL"

    # typer app path for memory profiling (if using typer)
    typer_app_path: str = ""  # e.g., "prefect.cli:app"


@dataclass
class BenchmarkConfig:
    """configuration for benchmark runs."""

    runs: int = 10
    warmup_runs: int = 3
    timeout_seconds: int = 30
    regression_threshold_percent: float = 10.0
    output_dir: Path = field(default_factory=lambda: Path(".benchmarks"))
    json_output: bool = False
    output_file: Path | None = None
    skip_memory: bool = False
    skip_cold: bool = False
    skip_warm: bool = False
    categories: list[str] | None = None
    server_url: str | None = None
    compare_baseline: Path | None = None
    verbose: bool = False


@dataclass
class TimingResult:
    """result from a hyperfine timing run."""

    mean_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    runs: int
    times_ms: list[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """result for a single benchmark command."""

    command: str
    command_args: list[str]
    category: str
    cold_start: TimingResult | None = None
    warm_cache: TimingResult | None = None
    peak_memory_mb: float | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class SuiteMetadata:
    """metadata for a benchmark suite run."""

    timestamp: str
    git_sha: str
    git_branch: str
    python_version: str
    platform: str
    target_version: str
    hyperfine_version: str


@dataclass
class BenchmarkSuite:
    """complete results from a benchmark suite run."""

    metadata: SuiteMetadata
    results: list[BenchmarkResult]
    config: BenchmarkConfig


def load_project_config(config_path: Path | None = None) -> ProjectConfig:
    """load project configuration from bench.toml.

    Searches for bench.toml in:
    1. Explicit path if provided
    2. Current directory
    3. Parent directories up to git root

    Returns default config if no file found.
    """
    if config_path and config_path.exists():
        return _parse_config(config_path)

    # search for bench.toml
    search_path = Path.cwd()
    while search_path != search_path.parent:
        config_file = search_path / "bench.toml"
        if config_file.exists():
            return _parse_config(config_file)
        # stop at git root
        if (search_path / ".git").exists():
            break
        search_path = search_path.parent

    # return default config
    return ProjectConfig()


def _parse_config(path: Path) -> ProjectConfig:
    """parse a bench.toml file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    config = ProjectConfig()
    config.project_root = path.parent

    # project section
    project = data.get("project", {})
    config.name = project.get("name", "cli")
    config.import_path = project.get("import_path", "")
    config.version_module = project.get("version_module", "")
    config.version_command = project.get("version_command")
    config.api_url_env_var = project.get("api_url_env_var", "")
    config.typer_app_path = project.get("typer_app_path", "")

    # commands section
    commands_data = data.get("commands", [])
    config.commands = [_parse_command(c) for c in commands_data]

    return config


def _parse_command(data: dict[str, Any]) -> BenchmarkCommand:
    """parse a command definition from toml."""
    return BenchmarkCommand(
        name=data["name"],
        args=data["args"],
        category=data.get("category", "default"),
        description=data.get("description", ""),
        requires_server=data.get("requires_server", False),
        timeout=data.get("timeout", 30),
    )


def get_commands(
    project: ProjectConfig,
    categories: list[str] | None = None,
    include_api: bool = True,
) -> list[BenchmarkCommand]:
    """get benchmark commands filtered by category.

    Args:
        project: Project configuration with command definitions.
        categories: List of categories to include. If None, include all.
        include_api: Whether to include API commands (requires server).

    Returns:
        Filtered list of benchmark commands.
    """
    commands = project.commands

    if categories:
        commands = [c for c in commands if c.category in categories]

    if not include_api:
        commands = [c for c in commands if not c.requires_server]

    return commands
