#!/usr/bin/env python3
"""
FP8 MTP Speculative Decoding Harness
=====================================
1. Monitors the FP8 download until all 42 shards are fully present.
2. Kicks off profile_runner.py with Baseline / MTP Speculative / MTP+TurboQuant.
3. Prints a clean summary at the end.

Usage:
    python3 scripts/profiling/fp8_mtp_harness.py
"""

import os
import sys
import time
import subprocess

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3.6-35B-A3B-FP8"
PROFILE_SCRIPT   = "scripts/profiling/profile_runner.py"
OUTPUT_MD        = "./profiling_results_fp8_mtp.md"
CONTEXTS         = "512,4096"
POLL_INTERVAL    = 10   # seconds between download checks

# All 42 expected safetensors shards for the FP8 release
EXPECTED_SHARDS = (
    [f"layers-{i}.safetensors" for i in range(40)]
    + ["mtp.safetensors", "outside.safetensors"]
)

HF_CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots"
)

# ── Helpers ──────────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW= "\033[33m"
DIM   = "\033[2m"
RESET = "\033[0m"

def find_snapshot_dir():
    """Return the first (and only) snapshot hash directory."""
    try:
        snaps = os.listdir(HF_CACHE_PATH)
        if snaps:
            return os.path.join(HF_CACHE_PATH, snaps[0])
    except FileNotFoundError:
        pass
    return None

def check_download_complete(snap_dir):
    """Returns (present, total, missing_list).
    A shard counts as present only if its resolved blob has size > 0.
    """
    if not snap_dir or not os.path.isdir(snap_dir):
        return 0, len(EXPECTED_SHARDS), EXPECTED_SHARDS[:]
    present = [s for s in EXPECTED_SHARDS if shard_real_size(snap_dir, s) > 0]
    missing = [s for s in EXPECTED_SHARDS if s not in present]
    return len(present), len(EXPECTED_SHARDS), missing

def shard_real_size(snap_dir, shard_name):
    """HF cache stores snapshot files as symlinks into blobs/. Follow the symlink."""
    path = os.path.join(snap_dir, shard_name)
    if not os.path.exists(path):
        return 0
    real = os.path.realpath(path)
    try:
        return os.path.getsize(real)
    except:
        return 0

def dir_size_gb(path):
    """Total size of blobs/ (real data, not symlinks)."""
    blobs_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "blobs")
    if not os.path.isdir(blobs_dir):
        blobs_dir = path  # fallback
    total = 0
    for root, _, files in os.walk(blobs_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except:
                pass
    return total / 1e9

def bar(n, total, width=30):
    filled = int(width * n / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + "]"

# ── Phase 1: Wait for download ────────────────────────────────────────────────
def wait_for_download():
    print(f"\n{BOLD}{CYAN}{'═'*66}{RESET}")
    print(f"{BOLD}{CYAN}  Phase 1: Waiting for FP8 download to complete{RESET}")
    print(f"{CYAN}{'═'*66}{RESET}\n")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Shards : {len(EXPECTED_SHARDS)} total (40 layer + mtp + outside)\n")

    total_target_gb = 37.5

    while True:
        snap_dir = find_snapshot_dir()
        present, total, missing = check_download_complete(snap_dir)

        if snap_dir:
            downloaded_gb = dir_size_gb(snap_dir)
        else:
            downloaded_gb = 0.0

        pct = int(100 * present / total)
        b = bar(present, total)
        status_line = (
            f"\r  Shards: {b} {present}/{total} ({pct}%)  "
            f"|  {downloaded_gb:.1f} / {total_target_gb:.1f} GB on disk"
        )
        sys.stdout.write(status_line)
        sys.stdout.flush()

        if present == total:
            print(f"\n\n  {GREEN}{BOLD}✅ Download complete! All {total} shards present.{RESET}\n")
            return snap_dir

        # Show what's missing (first 5)
        if missing:
            missing_preview = ", ".join(missing[:5])
            if len(missing) > 5:
                missing_preview += f" … (+{len(missing)-5} more)"
            sys.stdout.write(f"\n  {DIM}Pending: {missing_preview}{RESET}\n")
            sys.stdout.flush()

        time.sleep(POLL_INTERVAL)


# ── Phase 2: Run benchmark ───────────────────────────────────────────────────
def run_benchmark():
    print(f"\n{BOLD}{CYAN}{'═'*66}{RESET}")
    print(f"{BOLD}{CYAN}  Phase 2: Running MTP Benchmark on FP8 model{RESET}")
    print(f"{CYAN}{'═'*66}{RESET}\n")
    print(f"  Configs   : Baseline | MTP Speculative | MTP + TurboQuant")
    print(f"  Contexts  : {CONTEXTS} tokens")
    print(f"  Max gen   : 60 tokens")
    print(f"  Output    : {OUTPUT_MD}\n")

    # Kill any stale SwiftLM
    subprocess.run(["killall", "SwiftLM"], stderr=subprocess.DEVNULL)
    time.sleep(2)

    cmd = [
        sys.executable, "-u", PROFILE_SCRIPT,
        "--model", MODEL_ID,
        "--contexts", CONTEXTS,
        "--out", OUTPUT_MD,
    ]

    print(f"  {DIM}Running: {' '.join(cmd)}{RESET}\n")

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    ret = proc.wait()

    if ret == 0:
        print(f"\n{GREEN}{BOLD}✅ Benchmark complete! Results saved to: {OUTPUT_MD}{RESET}\n")
        # Print the markdown result file inline
        if os.path.exists(OUTPUT_MD):
            print(f"{DIM}{'─'*66}{RESET}")
            with open(OUTPUT_MD) as f:
                print(f.read())
    else:
        print(f"\n{YELLOW}{BOLD}⚠️  Benchmark exited with code {ret}. Check profile_server.log for details.{RESET}\n")
    return ret


# ── Phase 3: Validate MTP acceleration ──────────────────────────────────────
def validate_acceleration(output_md):
    """Parse the results markdown and check for 2.2x MTP acceleration target."""
    print(f"\n{BOLD}{CYAN}{'═'*66}{RESET}")
    print(f"{BOLD}{CYAN}  Phase 3: Acceleration Validation{RESET}")
    print(f"{CYAN}{'═'*66}{RESET}\n")

    if not os.path.exists(output_md):
        print(f"  {YELLOW}⚠️  Results file not found, skipping validation.{RESET}")
        return

    import re
    with open(output_md) as f:
        content = f.read()

    # Parse markdown table rows: | config | ctx | ttft | tps | ... |
    rows = re.findall(r'\|\s*([\w\s+/]+?)\s*\|\s*(\d+)\s*\|\s*([\d.]+)s\s*\|\s*([\d.]+)\s*tok/s', content)

    if not rows:
        print(f"  {YELLOW}No parseable rows in results table.{RESET}")
        return

    tps_by_config = {}
    for config, ctx, ttft, tps in rows:
        config = config.strip()
        if config not in tps_by_config:
            tps_by_config[config] = []
        tps_by_config[config].append(float(tps))

    avg_tps = {c: sum(v)/len(v) for c, v in tps_by_config.items()}

    baseline = avg_tps.get("Baseline", None)
    mtp_turbo = avg_tps.get("MTP + TurboQuant", avg_tps.get("MTP Speculative", None))

    print(f"  {'Config':<22}  {'Avg TPS':>8}")
    print(f"  {'─'*32}")
    for cfg, tps in sorted(avg_tps.items(), key=lambda x: x[1], reverse=True):
        star = " ★" if tps == max(avg_tps.values()) else ""
        print(f"  {cfg:<22}  {tps:>7.2f} tok/s{star}")

    if baseline and mtp_turbo and baseline > 0:
        ratio = mtp_turbo / baseline
        target = 2.2
        if ratio >= target:
            print(f"\n  {GREEN}{BOLD}🎯 TARGET MET: {ratio:.2f}x speedup ≥ {target}x CI threshold{RESET}")
        else:
            print(f"\n  {YELLOW}⚡ Speedup: {ratio:.2f}x (target: {target}x — not yet there){RESET}")
            print(f"  {DIM}Consider tuning MLX_MOE_CACHE_SLOTS or expanding context sizes.{RESET}")
    else:
        print(f"\n  {DIM}Insufficient data for acceleration ratio calculation.{RESET}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*66}")
    print(f"  FP8 MTP Speculative Decoding Harness")
    print(f"  Qwen3.6-35B-A3B-FP8  |  MTP heads: ✅ mtp.safetensors")
    print(f"{'═'*66}{RESET}")

    # Phase 1
    snap_dir = wait_for_download()

    # Phase 2
    ret = run_benchmark()

    # Phase 3
    validate_acceleration(OUTPUT_MD)

    sys.exit(ret)
