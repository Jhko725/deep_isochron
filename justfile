# The project's commands, one recipe each; `uv run just --list` prints them. CI calls
# these recipes too, so each command list is written once (README, docs/auto.md).

set shell := ["bash", "-euo", "pipefail", "-c"]

# Show the recipes
default:
    @just --list

# Check the system tools, sync the environment, build AUTO, install the git hooks
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    for tool in git gfortran make; do
        command -v "$tool" > /dev/null || missing+=("$tool")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "just setup: missing ${missing[*]}. Install them with one of:" >&2
        echo "  Debian, Ubuntu or WSL2:  sudo apt-get update && sudo apt-get install -y git gfortran make" >&2
        echo "  a cluster:               module load gcc   (the module that provides gfortran)" >&2
        echo "  conda:                   conda install -c conda-forge gfortran make git" >&2
        exit 1
    fi
    uv sync
    {{just_executable()}} install-auto
    uv run prek install

# Build AUTO-07p's Fortran tree at the commit pinned in uv.lock (docs/auto.md)
install-auto:
    scripts/install_auto.sh

# `check` ends with a grep because ty honors ignore comments, so a clean run does not
# prove there are none (ADR-0004). CI passes ty="uvx ty@latest" to check against the
# newest ty; locally the pinned dev-group copy runs.

# Lint, format, spelling, Markdown and docstring checks, then the type check of src
check ty="uv run ty":
    uv run prek run --all-files
    {{ty}} check src
    ! grep -rn 'ty: ignore' src

# The quick tier: `just test` (dev profile), `just test ci`, `just test dev tests/test_x.py`
test profile="dev" *paths="tests":
    uv run pytest {{paths}} -n auto -m "not slow" --hypothesis-profile={{profile}}

# The slow tier
test-slow profile="dev":
    uv run pytest -n auto -m slow --hypothesis-profile={{profile}}

# The tests that run AUTO (needs `just install-auto`): `quick`, or `full` for the
# five-period oracle suite as well (decision G3)
test-auto scope="quick":
    uv run pytest -q -n auto -m "{{ if scope == "full" { "auto" } else { "auto and not slow" } }}"

# End to end: generate a tiny Bautin dataset, train 20 steps, reopen the run
smoke:
    uv run python scripts/generate_data.py --config-name bautin \
        n_trajectories=16 time.n=81 time.t1=16.0 out_dir=ci-data
    uv run python scripts/train.py \
        data.n_trajectories=16 data.time.n=81 data.time.t1=16.0 data.out_dir=ci-data \
        model.inn.blocks=2 windows.length=20 windows.batch=32 validation.fraction=0.25 \
        validation.batch=16 num_steps=20 eval_every=10 log_every=5 checkpoint.every=10 \
        hydra.run.dir=ci-run
    uv run python -c "from deep_isochron.experiment import load_run; cfg, state = load_run('ci-run'); assert int(state.step) == 20, int(state.step); print('reopened run at step', int(state.step))"
