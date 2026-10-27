#!/usr/bin/env bash
# Build AUTO-07p's Fortran tree at the commit that uv.lock pins for its Python layer
# (the `auto` dependency group; docs/auto.md). The tree goes to
# ${DEEP_ISOCHRON_AUTO_DIR:-$HOME/.local/opt/auto-07p}; a stamp file records the commit,
# and a tree already built at that commit is left alone.
#
# Usage: scripts/install_auto.sh            build, or confirm the build
#        scripts/install_auto.sh --commit   print the pinned commit and exit (CI's cache key)
#
# Needs git, gfortran and make; on Debian or Ubuntu:
#   apt-get update && apt-get install -y git gfortran make
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${DEEP_ISOCHRON_AUTO_DIR:-$HOME/.local/opt/auto-07p}
REPO=https://github.com/auto-07p/auto-07p
STAMP=.deep_isochron_commit

COMMIT=$(sed -n 's|.*auto-07p?rev=[0-9a-f]*#\([0-9a-f]\{40\}\)".*|\1|p' "$ROOT/uv.lock" | head -n 1)
if [ -z "$COMMIT" ]; then
    echo "install_auto.sh: no auto-07p commit found in $ROOT/uv.lock" >&2
    exit 1
fi
if [ "${1:-}" = "--commit" ]; then
    echo "$COMMIT"
    exit 0
fi

for tool in git gfortran make; do
    if ! command -v "$tool" > /dev/null; then
        echo "install_auto.sh: needs $tool (apt-get install -y git gfortran make)" >&2
        exit 1
    fi
done

if [ -f "$DEST/$STAMP" ] && [ "$(cat "$DEST/$STAMP")" = "$COMMIT" ] \
    && compgen -G "$DEST/lib/*.o" > /dev/null; then
    echo "AUTO-07p $COMMIT is already built in $DEST"
    exit 0
fi

if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$(dirname "$DEST")"
    git clone --quiet --filter=blob:none "$REPO" "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"

cd "$DEST"
# configure reports the optional plotting libraries it cannot find; both are disabled
./configure --enable-plaut=no --enable-plaut04=no > configure.log 2>&1
# a serial make: with -j, src/ reads support.mod before it is written
make > make.log 2>&1 || { echo "install_auto.sh: make failed; see $DEST/make.log" >&2; exit 1; }
echo "$COMMIT" > "$STAMP"
echo "AUTO-07p $COMMIT built in $DEST"
