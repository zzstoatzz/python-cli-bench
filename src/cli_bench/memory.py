"""memory profiling using tracemalloc."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from .config import BenchmarkCommand, BenchmarkConfig, ProjectConfig


class MemoryProfiler:
    """profile memory usage of CLI commands using tracemalloc."""

    def __init__(self, config: BenchmarkConfig, project: ProjectConfig):
        self.config = config
        self.project = project

    def measure_command(self, cmd: BenchmarkCommand) -> float | None:
        """measure peak memory usage of a command.

        Uses tracemalloc in a subprocess to measure memory used by
        importing and invoking the CLI.

        Args:
            cmd: The command to measure.

        Returns:
            Peak memory in MB, or None if measurement failed.
        """
        script = self._build_measurement_script(cmd)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = Path(f.name)

        try:
            env = None
            if cmd.requires_server and self.config.server_url:
                env = os.environ.copy()
                api_var = self.project.api_url_env_var or "API_URL"
                env[api_var] = self.config.server_url

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=env,
            )

            if result.returncode != 0:
                if self.config.verbose:
                    print(
                        f"  Memory profiling failed: {result.stderr}", file=sys.stderr
                    )
                return None

            try:
                data = json.loads(result.stdout.strip())
                if data.get("exit_code", 0) != 0:
                    if self.config.verbose:
                        print(
                            f"  CLI command failed with exit code {data.get('exit_code')}",
                            file=sys.stderr,
                        )
                    return None
                return data["peak_mb"]
            except (json.JSONDecodeError, KeyError):
                if self.config.verbose:
                    print(
                        f"  Failed to parse memory output: {result.stdout}",
                        file=sys.stderr,
                    )
                return None

        except subprocess.TimeoutExpired:
            if self.config.verbose:
                print("  Memory profiling timed out", file=sys.stderr)
            return None
        finally:
            script_path.unlink(missing_ok=True)

    def _build_measurement_script(self, cmd: BenchmarkCommand) -> str:
        """build a Python script to measure memory usage.

        Uses CliRunner (typer) if typer_app_path is configured,
        otherwise falls back to subprocess.
        """
        cli_args = cmd.args[1:] if len(cmd.args) > 1 else []

        if self.project.typer_app_path:
            # use typer's CliRunner for in-process measurement
            module_path, app_name = self.project.typer_app_path.rsplit(":", 1)
            return textwrap.dedent(f"""
                import json
                import tracemalloc
                import sys

                tracemalloc.start()

                try:
                    from {module_path} import {app_name}
                    from typer.testing import CliRunner

                    runner = CliRunner()
                    result = runner.invoke({app_name}, {cli_args!r}, catch_exceptions=True)

                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    print(json.dumps({{
                        "peak_mb": peak / 1024 / 1024,
                        "current_mb": current / 1024 / 1024,
                        "exit_code": result.exit_code,
                    }}))

                except Exception as e:
                    tracemalloc.stop()
                    print(json.dumps({{"error": str(e)}}))
                    sys.exit(1)
            """).strip()
        else:
            # fallback to subprocess measurement (less accurate for CLI overhead)
            return textwrap.dedent(f"""
                import json
                import subprocess
                import tracemalloc
                import sys

                tracemalloc.start()

                try:
                    result = subprocess.run(
                        {cmd.args!r},
                        capture_output=True,
                        timeout=30,
                    )

                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    print(json.dumps({{
                        "peak_mb": peak / 1024 / 1024,
                        "current_mb": current / 1024 / 1024,
                        "exit_code": result.returncode,
                    }}))

                except Exception as e:
                    tracemalloc.stop()
                    print(json.dumps({{"error": str(e)}}))
                    sys.exit(1)
            """).strip()


def measure_import_memory(import_path: str) -> float:
    """measure memory used by just importing a module.

    This is useful for understanding baseline memory overhead.

    Args:
        import_path: The module path to import (e.g., "prefect.cli").
    """
    script = textwrap.dedent(f"""
        import json
        import tracemalloc

        tracemalloc.start()

        import {import_path}

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(json.dumps({{
            "peak_mb": peak / 1024 / 1024,
            "current_mb": current / 1024 / 1024,
        }}))
    """).strip()

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        data = json.loads(result.stdout.strip())
        return data["peak_mb"]
    return 0.0
