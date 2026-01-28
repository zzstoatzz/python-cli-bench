# python-cli-bench

a benchmark harness for python CLIs using [hyperfine](https://github.com/sharkdp/hyperfine).

## features

- wall-time benchmarks with hyperfine (warm/cold cache)
- import time analysis with `python -X importtime`
- call tree profiling with pyinstrument
- memory profiling with tracemalloc
- Welch's t-test for statistically significant regression detection
- terminal visualization with plotext
- JSON export for CI integration

## install

```bash
# with uv
uv tool install python-cli-bench

# with pip
pip install python-cli-bench

# hyperfine is required for benchmarks
brew install hyperfine  # macOS
```

## quickstart

```bash
# create a bench.toml in your project
cd your-project
cli-bench init

# edit bench.toml to configure your CLI commands
# then run benchmarks
cli-bench run

# save results for comparison
cli-bench run -o baseline.json

# compare against baseline (with Welch's t-test)
cli-bench run -c baseline.json
```

## configuration

create a `bench.toml` in your project root:

```toml
[project]
name = "mycli"
import_path = "mycli.cli"          # for profile/imports analysis
version_module = "mycli"           # to detect version
typer_app_path = "mycli.cli:app"   # for memory profiling (if using typer)

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

[[commands]]
name = "mycli some-command"
args = ["mycli", "some-command"]
category = "local"
description = "local command without network"
```

## usage

```bash
# run benchmarks
cli-bench run                    # all commands
cli-bench run --category startup # specific category
cli-bench run -o results.json    # save results
cli-bench run -c baseline.json   # compare against baseline

# analysis tools
cli-bench profile                # pyinstrument call tree
cli-bench imports                # import time breakdown
cli-bench imports --top 50       # show more modules

# visualization
cli-bench plot results.json
cli-bench plot results.json -c baseline.json
```

## statistical comparison

when comparing against a baseline, the tool uses Welch's t-test to determine if differences are statistically significant (p < 0.05). results marked with `?` are not statistically significant and should be interpreted with caution.

## license

MIT
