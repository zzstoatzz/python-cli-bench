#!/usr/bin/env python
"""
CLI benchmark harness for Python projects.

Subcommands:
  cli-bench run              # hyperfine wall-time benchmarks
  cli-bench profile          # pyinstrument call tree
  cli-bench imports          # import time breakdown
  cli-bench plot FILE        # visualize results
  cli-bench init             # create a bench.toml template

Examples:
  cli-bench run --output baseline.json
  cli-bench run --compare baseline.json
  cli-bench plot results.json --compare baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import (
    BenchmarkConfig,
    BenchmarkSuite,
    get_commands,
    load_project_config,
    ProjectConfig,
)
from .memory import MemoryProfiler
from .reporter import (
    console,
    print_comparison,
    print_error,
    print_info,
    print_progress,
    print_results,
    print_warning,
)
from .results import (
    compare_suites,
    create_metadata,
    load_suite,
    save_suite,
    suite_to_dict,
)
from .runner import (
    BenchmarkRunner,
    check_hyperfine,
    get_hyperfine_install_instructions,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Python CLI performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        "-C",
        type=Path,
        help="path to bench.toml config file",
    )
    parser.add_argument(
        "--project-root",
        "-P",
        type=Path,
        help="project root directory (where commands run from)",
    )
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="run hyperfine benchmarks")
    _add_run_args(run_parser)

    # profile subcommand
    profile_parser = subparsers.add_parser(
        "profile", help="profile CLI import with pyinstrument"
    )
    profile_parser.add_argument(
        "--import-path",
        help="module path to profile (overrides bench.toml)",
    )

    # imports subcommand
    imports_parser = subparsers.add_parser(
        "imports", help="analyze import time breakdown"
    )
    imports_parser.add_argument(
        "--top", "-n", type=int, default=25, help="number of modules to show"
    )
    imports_parser.add_argument(
        "--import-path",
        help="module path to analyze (overrides bench.toml)",
    )

    # plot subcommand
    plot_parser = subparsers.add_parser("plot", help="visualize benchmark results")
    plot_parser.add_argument("results", type=Path, help="benchmark results JSON")
    plot_parser.add_argument(
        "--compare", "-c", type=Path, help="baseline JSON to compare against"
    )
    plot_parser.add_argument(
        "--metric",
        choices=["all", "warm", "cold", "memory"],
        default="all",
        help="which metric to plot",
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="compare two benchmark result files"
    )
    compare_parser.add_argument("baseline", type=Path, help="baseline results JSON")
    compare_parser.add_argument("head", type=Path, help="head results JSON")
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="regression threshold %% (default: 10)",
    )
    compare_parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any command exceeds the regression threshold",
    )

    # init subcommand
    subparsers.add_parser("init", help="create a bench.toml template")

    return parser


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """add arguments for the run subcommand."""
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="runs per command (default: 5)",
    )
    parser.add_argument(
        "--category",
        default="all",
        help="command category (default: all)",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="skip memory profiling",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="save results to JSON file",
    )
    parser.add_argument(
        "--compare",
        "-c",
        type=Path,
        help="compare against baseline JSON (uses Welch's t-test)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show terminal visualization",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output raw JSON to stdout",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose output",
    )
    parser.add_argument(
        "--server-url",
        help="API URL for API commands (uses api_url_env_var from bench.toml)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="regression threshold %% (default: 10)",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        nargs="?",
        const=4,
        default=None,
        help="run benchmarks in parallel (default: 4 workers)",
    )


# =============================================================================
# Profile mode: pyinstrument call tree
# =============================================================================


def run_profile(project: ProjectConfig, import_path: str | None = None) -> int:
    """profile CLI import with pyinstrument to see where time is spent."""
    path = import_path or project.import_path
    if not path:
        print_error("no import_path specified")
        console.print("set import_path in bench.toml or use --import-path")
        return 1

    console.print(f"\n[bold cyan]Profiling import of {path}...[/bold cyan]\n")

    script = f"""
from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

import {path}

profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=False))
"""

    result = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(project.project_root),
            "--with",
            "pyinstrument",
            "python",
            "-c",
            script,
        ],
        capture_output=False,
    )
    return result.returncode


# =============================================================================
# Import mode: python -X importtime analysis
# =============================================================================


def run_imports(
    project: ProjectConfig, top_n: int = 25, import_path: str | None = None
) -> int:
    """analyze import times and show slowest modules."""
    path = import_path or project.import_path
    if not path:
        print_error("no import_path specified")
        console.print("set import_path in bench.toml or use --import-path")
        return 1

    console.print(f"\n[bold cyan]Analyzing import times for {path}...[/bold cyan]\n")

    result = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(project.project_root),
            "python",
            "-X",
            "importtime",
            "-c",
            f"import {path}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print_error(f"Import failed: {result.stderr}")
        return 1

    # parse importtime output
    imports = []
    for line in result.stderr.strip().split("\n"):
        match = re.match(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.+)", line)
        if match:
            self_time = int(match.group(1))
            cumulative = int(match.group(2))
            module = match.group(3).strip()
            imports.append((cumulative, self_time, module))

    imports.sort(reverse=True)

    from rich.table import Table

    table = Table(title=f"Top {top_n} Slowest Imports", show_header=True)
    table.add_column("Module", style="cyan")
    table.add_column("Cumulative (ms)", justify="right")
    table.add_column("Self (ms)", justify="right")
    table.add_column("", width=30)

    max_cum = imports[0][0] if imports else 1

    for cumulative, self_time, module in imports[:top_n]:
        bar_len = int((cumulative / max_cum) * 30)
        bar = "|" * bar_len

        if cumulative > 100000:
            bar = f"[red]{bar}[/red]"
        elif cumulative > 50000:
            bar = f"[yellow]{bar}[/yellow]"
        else:
            bar = f"[green]{bar}[/green]"

        table.add_row(
            module,
            f"{cumulative / 1000:.1f}",
            f"{self_time / 1000:.1f}",
            bar,
        )

    console.print(table)

    total_ms = imports[0][0] / 1000 if imports else 0
    console.print(f"\n[bold]Total import time:[/bold] {total_ms:.0f}ms")

    # project module breakdown (filter by project name)
    project_name = project.name.lower().replace("-", "_")
    console.print(f"\n[bold]{project.name} module breakdown:[/bold]")
    project_imports = [
        (c, s, m) for c, s, m in imports if project_name in m.lower()
    ][:10]
    for cumulative, _, module in project_imports:
        console.print(f"  {module}: [yellow]{cumulative / 1000:.0f}ms[/yellow]")

    return 0


# =============================================================================
# Init mode: create bench.toml template
# =============================================================================


def run_init() -> int:
    """create a bench.toml template in the current directory."""
    config_path = Path.cwd() / "bench.toml"
    if config_path.exists():
        print_error("bench.toml already exists")
        return 1

    template = '''\
# cli-bench configuration
# see https://github.com/zzstoatzz/python-cli-bench

[project]
name = "mycli"                           # project name for display
import_path = "mycli.cli"                # module to profile/analyze
version_module = "mycli"                 # module with __version__ attribute
# version_command = ["mycli", "--version"]  # alternative: run command for version
# typer_app_path = "mycli.cli:app"       # for memory profiling with typer
# api_url_env_var = "MYCLI_API_URL"      # env var for API commands

# benchmark commands
[[commands]]
name = "mycli --help"
args = ["mycli", "--help"]
category = "startup"
description = "baseline import cost"

[[commands]]
name = "mycli --version"
args = ["mycli", "--version"]
category = "startup"
description = "minimal output"

# [[commands]]
# name = "mycli some-command"
# args = ["mycli", "some-command", "--flag"]
# category = "local"
# description = "local command without network"

# [[commands]]
# name = "mycli api-command"
# args = ["mycli", "api-command"]
# category = "api"
# requires_server = true
# description = "command that needs a server"
'''

    config_path.write_text(template)
    print_info(f"Created {config_path}")
    console.print("\nEdit bench.toml to configure your CLI benchmarks, then run:")
    console.print("  cli-bench run")
    return 0


# =============================================================================
# Benchmark mode: hyperfine wall-time measurement
# =============================================================================


def _run_single_benchmark(cmd, config, project):
    """run a single benchmark - used for parallel execution."""
    runner = BenchmarkRunner(config, project)
    return runner.run_benchmark(cmd)


def run_benchmarks(args: argparse.Namespace, project: ProjectConfig) -> int:
    """run hyperfine benchmarks with statistical analysis."""
    # check hyperfine
    hyperfine_version = check_hyperfine()
    if hyperfine_version is None:
        print_error("hyperfine is required for benchmarks")
        console.print(f"\nInstall: {get_hyperfine_install_instructions()}")
        console.print(
            "\nOr use 'cli-bench profile' or 'cli-bench imports' "
            "for dependency-free analysis"
        )
        return 1

    # check for commands
    if not project.commands:
        print_error("no commands defined")
        console.print("run 'cli-bench init' to create a bench.toml template")
        return 1

    # get server URL from env if not specified
    server_url = args.server_url
    if not server_url and project.api_url_env_var:
        server_url = os.environ.get(project.api_url_env_var)

    # build config
    config = BenchmarkConfig(
        runs=args.runs,
        warmup_runs=3,
        timeout_seconds=30,
        regression_threshold_percent=args.threshold,
        json_output=args.json,
        output_file=args.output,
        skip_memory=args.skip_memory,
        skip_cold=True,
        skip_warm=False,
        categories=[args.category] if args.category != "all" else None,
        server_url=server_url,
        compare_baseline=args.compare,
        verbose=args.verbose,
    )

    # get commands
    commands = get_commands(
        project,
        categories=config.categories,
        include_api=config.server_url is not None,
    )

    if not commands:
        print_error("No commands to benchmark")
        return 1

    # filter API commands if no server
    api_cmds = [c for c in commands if c.requires_server]
    if api_cmds and not config.server_url:
        print_warning(f"Skipping {len(api_cmds)} API commands (no --server-url)")
        commands = [c for c in commands if not c.requires_server]

    if not commands:
        print_error("No commands left after filtering")
        return 1

    import time

    print_info(f"Benchmarking {len(commands)} commands ({config.runs} runs each)")
    console.print()

    # run benchmarks
    runner = BenchmarkRunner(config, project)
    memory_profiler = MemoryProfiler(config, project) if not config.skip_memory else None

    start_time = time.time()

    if args.parallel:
        print_warning(f"Parallel mode ({args.parallel} workers) - results may be less accurate due to CPU contention")
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = [None] * len(commands)
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_idx = {
                executor.submit(_run_single_benchmark, cmd, config, project): i
                for i, cmd in enumerate(commands)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                result = future.result()
                print_progress(result.command, completed, len(commands))

                if memory_profiler and result.success:
                    result.peak_memory_mb = memory_profiler.measure_command(commands[idx])

                results[idx] = result
    else:
        results = []
        for i, cmd in enumerate(commands, 1):
            print_progress(cmd.name, i, len(commands))
            result = runner.run_benchmark(cmd)

            if memory_profiler and result.success:
                result.peak_memory_mb = memory_profiler.measure_command(cmd)

            results.append(result)

    elapsed = time.time() - start_time

    # create suite
    metadata = create_metadata(hyperfine_version, project)
    suite = BenchmarkSuite(metadata=metadata, results=results, config=config)

    # output
    console.print()
    console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]")
    console.print()

    if config.compare_baseline:
        try:
            baseline = load_suite(config.compare_baseline)
            comparisons = compare_suites(
                suite, baseline, config.regression_threshold_percent
            )
            print_comparison(suite, comparisons, str(config.compare_baseline), project)
        except FileNotFoundError:
            print_error(f"Baseline not found: {config.compare_baseline}")
            print_results(suite, project)
    else:
        print_results(suite, project)

    # save output
    output_path = config.output_file
    if config.json_output:
        if output_path:
            save_suite(suite, output_path)
            print_info(f"Saved to {output_path}")
        else:
            print(json.dumps(suite_to_dict(suite), indent=2))
    elif output_path:
        save_suite(suite, output_path)
        print_info(f"Saved to {output_path}")

    # plot
    if args.plot:
        from .plot import run_plot

        temp_path = output_path
        if not temp_path:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(suite_to_dict(suite), f)
                temp_path = Path(f.name)
        run_plot(temp_path, config.compare_baseline)

    # return code
    errors = sum(1 for r in results if r.error)
    return 1 if errors > 0 else 0


# =============================================================================
# Compare mode: regression detection for CI
# =============================================================================


def run_compare(args: argparse.Namespace) -> int:
    """compare two benchmark result files and optionally fail on regression."""
    from .results import compare_suites, load_suite

    baseline = load_suite(args.baseline)
    head = load_suite(args.head)
    comparisons = compare_suites(head, baseline, args.threshold)

    regressions = [c for c in comparisons if c.is_regression]

    # build markdown summary
    lines = [
        "## CLI Benchmark Results",
        "",
        f"Threshold: {args.threshold}% regression (p < 0.05)",
        "",
        "| Command | Base (ms) | Head (ms) | Delta | Sig? |",
        "|---|---:|---:|---:|:---:|",
    ]

    for comp in comparisons:
        b = comp.baseline
        c = comp.current
        c_warm = c.warm_cache

        if b is None:
            c_mean = c_warm.mean_ms if c_warm else None
            lines.append(
                f"| {c.command} | (new) | {c_mean:.0f} | - | - |"
                if c_mean is not None
                else f"| {c.command} | (new) | - | - | - |"
            )
            continue

        b_warm = b.warm_cache
        b_mean = b_warm.mean_ms if b_warm else None
        c_mean = c_warm.mean_ms if c_warm else None

        if b_mean is None or c_mean is None:
            lines.append(f"| {c.command} | - | - | - | - |")
            continue

        delta_pct = comp.warm_diff_percent
        sig = "yes" if comp.warm_significant else "no"
        is_reg = comp.is_regression
        flag = " :red_circle:" if is_reg else ""
        lines.append(
            f"| {c.command} | {b_mean:.0f} | {c_mean:.0f}"
            f" | {delta_pct:+.1f}% | {sig}{flag} |"
        )

    if regressions:
        lines.append("")
        lines.append(f"**{len(regressions)} significant regression(s) detected:**")
        for comp in regressions:
            lines.append(
                f"- `{comp.command}`: {comp.warm_diff_percent:+.1f}%"
                f" (p={comp.warm_p_value:.4f})"
            )

    text = "\n".join(lines)

    # write to GITHUB_STEP_SUMMARY if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(text + "\n")

    print(text)

    if regressions and args.fail_on_regression:
        print_error(
            f"CLI startup regression: {len(regressions)} command(s)"
            f" exceeded {args.threshold}% threshold"
        )
        return 1

    return 0


# =============================================================================
# Main entry point
# =============================================================================


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    # load project config
    project = load_project_config(
        config_path=args.config if hasattr(args, "config") else None,
        project_root=getattr(args, "project_root", None),
    )

    # default to 'run' if no subcommand
    if args.command is None:
        # re-parse with 'run' as default
        args = parser.parse_args(["run"] + sys.argv[1:])

    if args.command == "init":
        return run_init()

    if args.command == "profile":
        return run_profile(project, getattr(args, "import_path", None))

    if args.command == "imports":
        return run_imports(project, args.top, getattr(args, "import_path", None))

    if args.command == "plot":
        from .plot import run_plot

        return run_plot(args.results, args.compare, args.metric)

    if args.command == "compare":
        return run_compare(args)

    if args.command == "run":
        return run_benchmarks(args, project)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
