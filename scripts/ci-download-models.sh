#!/usr/bin/env bash
# ci-download-models.sh — fetch HuggingFace models for CI, with retries and verification.
#
# Why this exists: the prefetch steps used to be `hf download <repo> || true`, which
# swallowed every failure. A stalled or partial download then surfaced much later as a
# weight-loading error ("Key ... not found", "Mismatched parameter ... shape") or as the
# server exceeding the test's 180 s startup window while it downloaded the model itself.
#
# Two guards:
#   1. HF_HUB_DOWNLOAD_TIMEOUT bounds each request, so a connection the server drops
#      (observed as sockets stuck in CLOSE_WAIT with 0 bytes moving) fails instead of
#      hanging until the job timeout.
#   2. Leftover *.incomplete files are treated as failure, so a half-downloaded model is
#      never handed to the tests — and never saved into the actions/cache entry.
#
# Usage: ci-download-models.sh <repo> [<repo> ...]

set -uo pipefail

: "${HF_HUB_DOWNLOAD_TIMEOUT:=60}"
export HF_HUB_DOWNLOAD_TIMEOUT

ATTEMPTS="${CI_DOWNLOAD_ATTEMPTS:-3}"
HUB_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

has_partial_files() {
    local dir="$1"
    [ -d "$dir" ] || return 1
    find "$dir" -name '*.incomplete' -print -quit 2>/dev/null | grep -q .
}

download_one() {
    local repo="$1"
    local dir="$HUB_DIR/models--${repo//\//--}"

    for attempt in $(seq 1 "$ATTEMPTS"); do
        echo "--- $repo (attempt $attempt/$ATTEMPTS, per-request timeout ${HF_HUB_DOWNLOAD_TIMEOUT}s)"
        if hf download "$repo"; then
            if has_partial_files "$dir"; then
                echo "::warning::$repo downloaded but .incomplete files remain; retrying"
            else
                echo "--- $repo OK"
                return 0
            fi
        else
            echo "::warning::$repo download failed on attempt $attempt"
        fi
        # Resume picks up from the existing .incomplete blobs on the next attempt.
        sleep $(( attempt * 10 ))
    done

    echo "::error::failed to download $repo after $ATTEMPTS attempts"
    if [ -d "$dir" ]; then
        echo "Partial state left in $dir:"
        find "$dir" -name '*.incomplete' -exec ls -lh {} + 2>/dev/null | head -10
    fi
    return 1
}

status=0
for repo in "$@"; do
    download_one "$repo" || status=1
done

if [ "$status" -ne 0 ]; then
    echo "::error::one or more models could not be downloaded — failing early rather than"
    echo "::error::letting the test suite load partial weights or time out on a cold download."
fi
exit "$status"
