"""
gossip_convergence.py — Gossip HLL convergence measurement for Nexus.

Launches a 2-node cluster, indexes a split corpus, then polls /stats on both
nodes every 2 seconds to track how estimated_global_terms converges toward the
true global term cardinality.

Produces:
  - A convergence table (elapsed_s, est_A, est_B, error_A%, error_B%)
  - An ASCII curve of error_A over time
  - Time-to-convergence at ±5%, ±10%, ±20% thresholds
  - Results saved to /tmp/nexus_bench/gossip_convergence.json

Usage:
    cargo build -p nexus-core --release
    python3 tools/gossip_convergence.py
    python3 tools/gossip_convergence.py --corpus-size 1000 --duration 300

Dependencies:
    pip install requests tqdm

Key insight:
    The gossip loop fires every 30s (GOSSIP_INTERVAL_S in main.rs).
    Each poll sample is labelled with the gossip round number (elapsed // 30).
    Convergence = estimated_global_terms within ±threshold% of true_global_terms.

    True global terms ≈ vocab_size_A + vocab_size_B (upper bound, since some
    terms overlap). We use local vocab_size from /stats as a proxy.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: pip install requests")

# Reuse helpers from benchmark.py (corpus fetch, index builder)
sys.path.insert(0, str(Path(__file__).parent))
from benchmark import (
    CORPUS_SIZE,
    fetch_corpus,
    build_nexus_index,
    fnv1a_32,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOSSIP_INTERVAL_S = 30          # Must match tokio::time::interval in main.rs
POLL_INTERVAL_S = 2             # How often we sample /stats
DEFAULT_DURATION_S = 180        # 3 minutes = 6 gossip rounds
STARTUP_TIMEOUT_S = 30
CONVERGENCE_THRESHOLDS = [0.05, 0.10, 0.20]  # ±5%, ±10%, ±20%

HTTP_A = 3091
HTTP_B = 3092
QUIC_A = 4091
QUIC_B = 4092


# ---------------------------------------------------------------------------
# Node lifecycle (mirrors sweep_dp_epsilon.py)
# ---------------------------------------------------------------------------

def wait_for_health(url: str, timeout: float = STARTUP_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def start_node(
    nexus_bin: str,
    http_port: int,
    quic_port: int,
    data_dir: Path,
    peers: str = "",
    epsilon: float = 1.0,
) -> subprocess.Popen:
    env = {
        **os.environ,
        "NEXUS_DP_EPSILON": str(epsilon),
        "NEXUS_DATA_DIR": str(data_dir),
        "NEXUS_HTTP_PORT": str(http_port),
        "NEXUS_QUIC_PORT": str(quic_port),
        "NEXUS_PEERS": peers,
    }
    return subprocess.Popen(
        [nexus_bin],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_node(proc: subprocess.Popen, port: int) -> None:
    import socket as _socket
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        proc.kill()
        proc.wait()
    deadline = time.time() + 10
    while time.time() < deadline:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Stats polling
# ---------------------------------------------------------------------------

def fetch_stats(url: str) -> dict | None:
    try:
        r = requests.get(f"{url}/stats", timeout=2)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


# ---------------------------------------------------------------------------
# Relative error helper
# ---------------------------------------------------------------------------

def rel_error(estimated: float, true_val: float) -> float:
    """Signed relative error: (estimated - true) / true."""
    if true_val <= 0:
        return 0.0
    return (estimated - true_val) / true_val


# ---------------------------------------------------------------------------
# ASCII curve
# ---------------------------------------------------------------------------

def ascii_curve(samples: list[dict], key: str, label: str, width: int = 50) -> None:
    values = [abs(s[key]) for s in samples if s[key] is not None]
    if not values:
        return
    max_v = max(values) if values else 1.0
    print(f"\n  {label}")
    print(f"  {'time':>6}  {'round':>5}  {'bar':<{width}}  abs_err%")
    print(f"  {'-' * (width + 22)}")
    for s in samples:
        v = abs(s.get(key) or 0.0)
        bar_len = int(v / (max_v + 1e-9) * width)
        bar = "█" * bar_len
        elapsed = s["elapsed_s"]
        rnd = s["gossip_round"]
        print(f"  {elapsed:>5.0f}s  {rnd:>5}  {bar:<{width}}  {v * 100:6.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus gossip convergence measurement")
    parser.add_argument("--nexus-bin", default="./target/release/nexus",
                        help="Path to nexus binary")
    parser.add_argument("--corpus-size", type=int, default=500,
                        help="Number of crates to index (split 50/50 across 2 nodes)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_S,
                        help="Measurement duration in seconds (default: 180 = 6 gossip rounds)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="DP epsilon for both nodes (default: 1.0)")
    args = parser.parse_args()

    if not Path(args.nexus_bin).exists():
        raise SystemExit(
            f"Binary not found: {args.nexus_bin}\n"
            "Run: cargo build -p nexus-core --release"
        )

    # ---------------------------------------------------------------------------
    # Prepare corpus
    # ---------------------------------------------------------------------------
    print("[convergence] Loading corpus ...")
    docs = fetch_corpus(skip=True)[: args.corpus_size]
    mid = len(docs) // 2
    docs_a, docs_b = docs[:mid], docs[mid:]

    print(f"[convergence] Corpus: {len(docs)} crates  "
          f"(A={len(docs_a)}, B={len(docs_b)})")
    print(f"[convergence] Gossip interval: {GOSSIP_INTERVAL_S}s  "
          f"| Poll interval: {POLL_INTERVAL_S}s  "
          f"| Duration: {args.duration}s ({args.duration // GOSSIP_INTERVAL_S} rounds)")
    print(f"[convergence] DP epsilon: {args.epsilon}")

    # ---------------------------------------------------------------------------
    # Start nodes
    # ---------------------------------------------------------------------------
    data_a = Path("/tmp/nexus_convergence_a")
    data_b = Path("/tmp/nexus_convergence_b")
    data_a.mkdir(parents=True, exist_ok=True)
    data_b.mkdir(parents=True, exist_ok=True)

    url_a = f"http://localhost:{HTTP_A}"
    url_b = f"http://localhost:{HTTP_B}"

    print(f"\n[convergence] Starting node B (port {HTTP_B}) ...")
    proc_b = start_node(args.nexus_bin, HTTP_B, QUIC_B, data_b, epsilon=args.epsilon)
    if not wait_for_health(url_b):
        stop_node(proc_b, HTTP_B)
        raise SystemExit("Node B failed to start")

    print(f"[convergence] Starting node A (port {HTTP_A}), peer=127.0.0.1:{QUIC_B} ...")
    proc_a = start_node(
        args.nexus_bin, HTTP_A, QUIC_A, data_a,
        peers=f"127.0.0.1:{QUIC_B}",
        epsilon=args.epsilon,
    )
    if not wait_for_health(url_a):
        stop_node(proc_a, HTTP_A)
        stop_node(proc_b, HTTP_B)
        raise SystemExit("Node A failed to start")

    try:
        # ---------------------------------------------------------------------------
        # Index corpus (split 50/50)
        # ---------------------------------------------------------------------------
        print(f"[convergence] Indexing {len(docs_a)} crates on node A ...")
        build_nexus_index(docs_a, url_a)
        print(f"[convergence] Indexing {len(docs_b)} crates on node B ...")
        build_nexus_index(docs_b, url_b)

        # Baseline: local vocab sizes right after indexing (before any gossip)
        stats_a0 = fetch_stats(url_a) or {}
        stats_b0 = fetch_stats(url_b) or {}
        vocab_a = stats_a0.get("vocab_size", 0)
        vocab_b = stats_b0.get("vocab_size", 0)

        # True global term cardinality: we use each node's local HLL estimate
        # of its own terms (estimated_global_terms before any gossip = local-only HLL).
        # The target after convergence: node A should see est ≈ node B's pre-gossip est,
        # and vice versa. We define truth as: max(local_est_A, local_est_B) * 2 / (1 + overlap).
        #
        # Simpler and more honest: poll both nodes right after indexing (before gossip)
        # to get their local-only estimates, then define true_global as the value we
        # expect after full merge. Since HLL is a CRDT (per-register max), the merged
        # estimate converges to the cardinality of the union of both term sets.
        # We approximate this by summing local estimates and applying a Jaccard-based
        # deduplication factor. For crates.io, empirical overlap is ~45-55%.
        #
        # However the cleanest ground truth is: build a single reference HLL locally.
        # We do this by collecting all terms from both halves in Python.
        import hashlib

        def hll_estimate_python(terms: list[str], b: int = 10) -> float:
            """Minimal HyperLogLog estimate (matching Nexus's b=10, 1024 registers)."""
            m = 1 << b
            regs = [0] * m
            for term in terms:
                h = int(hashlib.sha256(term.encode()).hexdigest(), 16)
                idx = h >> (128 - b)
                w = h & ((1 << (128 - b)) - 1)
                leading = 128 - b - w.bit_length() + 1 if w > 0 else (128 - b + 1)
                regs[idx] = max(regs[idx], leading)
            import math
            alpha = 0.7213 / (1 + 1.079 / m)
            raw = alpha * m * m / sum(2.0 ** (-r) for r in regs)
            # Small/large range corrections
            zeros = regs.count(0)
            if zeros > 0:
                raw = m * math.log(m / zeros)
            return raw

        # Collect all unique terms from both halves of the corpus.
        import re as _re
        all_terms_a = set()
        all_terms_b = set()
        for doc in docs_a:
            text = f"{doc.get('name', '')} {doc.get('description', '')} {' '.join(doc.get('keywords', []))}"
            for tok in _re.findall(r"[a-z]+", text.lower()):
                if len(tok) >= 2:
                    all_terms_a.add(tok)
        for doc in docs_b:
            text = f"{doc.get('name', '')} {doc.get('description', '')} {' '.join(doc.get('keywords', []))}"
            for tok in _re.findall(r"[a-z]+", text.lower()):
                if len(tok) >= 2:
                    all_terms_b.add(tok)
        all_terms = all_terms_a | all_terms_b
        true_global_exact = len(all_terms)
        overlap = len(all_terms_a & all_terms_b)
        overlap_pct = overlap / len(all_terms_a | all_terms_b) * 100 if all_terms else 0

        # Ground truth: after gossip converges, each node's HLL estimate should stabilize.
        # We use vocab_A + vocab_B as an upper bound and the max local vocab as a lower bound.
        # The real union (with stemming overlap) is between max(vocab_A, vocab_B) and vocab_A+vocab_B.
        # We track convergence as: does est_A stabilize? (delta between rounds < 1%)
        true_global_upper = vocab_a + vocab_b       # no overlap assumed
        true_global_lower = max(vocab_a, vocab_b)   # full overlap assumed
        true_global = (true_global_upper + true_global_lower) // 2  # midpoint heuristic
        print(f"[convergence] Local vocab: A={vocab_a}, B={vocab_b}")
        print(f"[convergence] Ground truth range: [{true_global_lower}, {true_global_upper}]  "
              f"| midpoint={true_global}")
        print(f"[convergence] Python union: {true_global_exact} tokens  "
              f"| Overlap: {overlap} ({overlap_pct:.1f}%)")
        print(f"[convergence] NOTE: Error% is relative to midpoint heuristic. "
              f"Stability (Δ between rounds) is the real convergence signal.")
        print(f"\n[convergence] Starting convergence measurement for {args.duration}s ...")
        print(f"  (gossip rounds will fire at t={GOSSIP_INTERVAL_S}s, "
              f"t={GOSSIP_INTERVAL_S*2}s, ...)\n")

        # ---------------------------------------------------------------------------
        # Poll loop
        # ---------------------------------------------------------------------------
        samples: list[dict] = []
        convergence_times: dict[str, dict[float, float | None]] = {
            "A": {t: None for t in CONVERGENCE_THRESHOLDS},
            "B": {t: None for t in CONVERGENCE_THRESHOLDS},
        }
        # Track first stable round: when est_A stops changing between gossip rounds.
        first_stable_round: int | None = None
        prev_round_est_a: float | None = None
        current_round = -1

        t_start = time.time()
        print(f"  {'elapsed':>7}  {'round':>5}  "
              f"{'est_A':>8}  {'err_A%':>7}  {'delta_A':>8}  "
              f"{'est_B':>8}  {'peers_A':>7}")

        while True:
            elapsed = time.time() - t_start
            gossip_round = int(elapsed // GOSSIP_INTERVAL_S)

            stats_a = fetch_stats(url_a)
            stats_b = fetch_stats(url_b)

            est_a = stats_a.get("estimated_global_terms", 0.0) if stats_a else None
            est_b = stats_b.get("estimated_global_terms", 0.0) if stats_b else None
            peers_a = stats_a.get("peer_count", 0) if stats_a else 0

            err_a = rel_error(est_a, true_global) if est_a is not None else None
            err_b = rel_error(est_b, true_global) if est_b is not None else None

            # Detect stabilization: first round where est_A doesn't change vs previous round.
            # Skip rounds where prev=0 (before first gossip tick) to avoid division artifacts.
            delta_a: float | None = None
            if gossip_round != current_round:
                if (prev_round_est_a is not None and est_a is not None
                        and prev_round_est_a > 0):
                    delta_a = abs(est_a - prev_round_est_a) / prev_round_est_a
                    if delta_a < 0.01 and first_stable_round is None and gossip_round > 0:
                        first_stable_round = gossip_round
                if est_a is not None:
                    prev_round_est_a = est_a
                current_round = gossip_round

            sample = {
                "elapsed_s": round(elapsed, 1),
                "gossip_round": gossip_round,
                "est_a": est_a,
                "est_b": est_b,
                "err_a": err_a,
                "err_b": err_b,
                "delta_a": delta_a,
                "peers_a": peers_a,
                "true_global": true_global,
            }
            samples.append(sample)

            # Track error-threshold convergence
            for thr in CONVERGENCE_THRESHOLDS:
                if err_a is not None and convergence_times["A"][thr] is None:
                    if abs(err_a) <= thr:
                        convergence_times["A"][thr] = elapsed
                if err_b is not None and convergence_times["B"][thr] is None:
                    if abs(err_b) <= thr:
                        convergence_times["B"][thr] = elapsed

            # Print row
            est_a_str = f"{est_a:8.0f}" if est_a is not None else "       ?"
            est_b_str = f"{est_b:8.0f}" if est_b is not None else "       ?"
            err_a_str = f"{err_a * 100:+7.1f}%" if err_a is not None else "      ?"
            delta_str = f"{delta_a * 100:+7.1f}%" if delta_a is not None else "        "
            print(f"  {elapsed:>6.1f}s  {gossip_round:>5}  "
                  f"{est_a_str}  {err_a_str}  {delta_str}  "
                  f"{est_b_str}  {peers_a:>7}")

            if elapsed >= args.duration:
                break
            time.sleep(POLL_INTERVAL_S)

        # ---------------------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------------------
        print("\n" + "=" * 65)
        print("  Gossip Convergence Report")
        print(f"  Nodes: 2  |  Corpus: {len(docs)} crates  "
              f"|  ε={args.epsilon}  |  Duration: {args.duration}s")
        print(f"  True global terms (upper bound): {true_global}")
        print("=" * 65)

        print("\n  Time to convergence (node A):")
        for thr in CONVERGENCE_THRESHOLDS:
            t = convergence_times["A"][thr]
            if t is not None:
                rounds = t / GOSSIP_INTERVAL_S
                print(f"    ±{thr * 100:.0f}%:  {t:.1f}s  ({rounds:.1f} gossip rounds)")
            else:
                print(f"    ±{thr * 100:.0f}%:  did not converge within {args.duration}s")

        print("\n  Time to convergence (node B):")
        for thr in CONVERGENCE_THRESHOLDS:
            t = convergence_times["B"][thr]
            if t is not None:
                rounds = t / GOSSIP_INTERVAL_S
                print(f"    ±{thr * 100:.0f}%:  {t:.1f}s  ({rounds:.1f} gossip rounds)")
            else:
                print(f"    ±{thr * 100:.0f}%:  did not converge within {args.duration}s")

        # Stability convergence
        if first_stable_round is not None:
            print(f"\n  Stabilization: est_A stopped changing at round {first_stable_round} "
                  f"(t≈{first_stable_round * GOSSIP_INTERVAL_S}s)")
        else:
            print(f"\n  Stabilization: est_A did not stabilize within {args.duration}s")

        # Final error
        last = samples[-1]
        print(f"\n  Final state (t={last['elapsed_s']}s, round {last['gossip_round']}):")
        if last["err_a"] is not None:
            print(f"    Node A: est={last['est_a']:.0f}  "
                  f"err vs midpoint={last['err_a'] * 100:+.1f}%  "
                  f"(range [{true_global_lower}, {true_global_upper}])")
        if last["err_b"] is not None:
            print(f"    Node B: est={last['est_b']:.0f}  err vs midpoint={last['err_b'] * 100:+.1f}%")

        ascii_curve(samples, "err_a", "Node A — |error%| over time (abs)")

        # Save results
        out_path = Path("/tmp/nexus_bench/gossip_convergence.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "corpus_size": len(docs),
                "vocab_a": vocab_a,
                "vocab_b": vocab_b,
                "true_global": true_global,
                "gossip_interval_s": GOSSIP_INTERVAL_S,
                "epsilon": args.epsilon,
                "duration_s": args.duration,
            },
            "convergence_times": {
                node: {str(k): v for k, v in times.items()}
                for node, times in convergence_times.items()
            },
            "samples": samples,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Results saved to {out_path}")
        print("=" * 65)

    finally:
        print("\n[convergence] Stopping nodes ...")
        stop_node(proc_a, HTTP_A)
        stop_node(proc_b, HTTP_B)


if __name__ == "__main__":
    main()
