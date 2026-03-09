"""
sweep_dp_epsilon.py — Privacy-utility tradeoff curve for Nexus ε-DP.

Measures NL MRR@10 across a range of dp_epsilon values by launching a fresh
Nexus node for each epsilon, indexing the corpus, and running the NL query set.

Usage:
    # Requires a release build to be up to date:
    cargo build -p nexus-core --release

    python3 tools/sweep_dp_epsilon.py
    python3 tools/sweep_dp_epsilon.py --nexus-bin ./target/release/nexus
    python3 tools/sweep_dp_epsilon.py --port 3099

Dependencies:
    pip install requests tqdm

Notes on what ε actually controls:
    - Nexus gossips HyperLogLog sketches with Laplace(0, 1/ε) noise per register.
    - This perturbs the estimated_global_terms (N) used for IDF in BM25.
    - Low ε → high noise → N estimate is unreliable → IDF scores are off.
    - High ε → low noise → N estimate is accurate → IDF is correct.
    - On a single-node benchmark the effect is subtle: global_n comes from gossip,
      and with no peers the gossip loop never fires. The effect is visible only when
      the node receives and merges an external sketch.
    - This script injects a synthetic peer sketch to force the noisy N estimate
      into the scorer, making ε's effect measurable even in single-node mode.
    - Caveat: corpus is 2000 docs. With more docs the noise signal would be stronger.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: pip install requests")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it

# Reuse helpers from benchmark.py
sys.path.insert(0, str(Path(__file__).parent))
from benchmark import (
    NL_QUERIES,
    CORPUS_CACHE,
    CORPUS_SIZE,
    SEARCH_LIMIT,
    fetch_corpus,
    build_nexus_index,
    search_nexus,
    fnv1a_32,
    reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEXUS_URL = "http://localhost:{port}"
STARTUP_TIMEOUT_S = 30
EPSILONS: list[float] = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 1000.0]
EPSILON_LABELS: dict[float, str] = {1000.0: "∞ (no noise)"}

# ---------------------------------------------------------------------------
# Node lifecycle
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
    epsilon: float,
    data_dir: Path,
    peers: str = "",
) -> subprocess.Popen:
    env = {
        **os.environ,
        "NEXUS_DP_EPSILON": str(epsilon),
        "NEXUS_DATA_DIR": str(data_dir),
        "NEXUS_HTTP_PORT": str(http_port),
        "NEXUS_QUIC_PORT": str(quic_port),
        "NEXUS_PEERS": peers,
    }
    proc = subprocess.Popen(
        [nexus_bin],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def stop_node(proc: subprocess.Popen, port: int) -> None:
    import socket as _socket
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        proc.kill()
        proc.wait()
    # Wait until the port is released before returning.
    deadline = time.time() + 10
    while time.time() < deadline:
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Per-epsilon run — two nodes with live gossip
# ---------------------------------------------------------------------------


def run_one(
    epsilon: float,
    nexus_bin: str,
    base_port: int,
    docs: list[dict],
    docid_to_name: dict[int, str],
) -> Optional[dict]:
    label = EPSILON_LABELS.get(epsilon, str(epsilon))
    print(f"\n[sweep] ε={label}  — starting 2-node cluster ...")

    # Node A: HTTP=base_port, QUIC=base_port+1000
    # Node B: HTTP=base_port+1, QUIC=base_port+1001
    # Node A bootstraps against Node B's QUIC port.
    http_a, quic_a = base_port, base_port + 1000
    http_b, quic_b = base_port + 1, base_port + 1001

    data_a = Path(f"/tmp/nexus_sweep_eps_{epsilon}_a")
    data_b = Path(f"/tmp/nexus_sweep_eps_{epsilon}_b")
    data_a.mkdir(parents=True, exist_ok=True)
    data_b.mkdir(parents=True, exist_ok=True)

    url_a = f"http://localhost:{http_a}"

    # Start node B first (so node A can bootstrap against it).
    proc_b = start_node(nexus_bin, http_b, quic_b, epsilon, data_b)
    if not wait_for_health(f"http://localhost:{http_b}"):
        print(f"[sweep] ε={label}  — node B failed to start, skipping")
        stop_node(proc_b, http_b)
        return None

    # Start node A, pointing to node B's QUIC address for bootstrap.
    proc_a = start_node(
        nexus_bin, http_a, quic_a, epsilon, data_a,
        peers=f"127.0.0.1:{quic_b}",
    )
    if not wait_for_health(url_a):
        print(f"[sweep] ε={label}  — node A failed to start, skipping")
        stop_node(proc_a, http_a)
        stop_node(proc_b, http_b)
        return None

    try:
        # Index half the corpus on each node so they have different term distributions.
        mid = len(docs) // 2
        print(f"[sweep] ε={label}  — indexing {mid} crates on node A, {len(docs)-mid} on node B ...")
        build_nexus_index(docs[:mid], url_a)
        build_nexus_index(docs[mid:], f"http://localhost:{http_b}")

        # Wait one gossip interval (30s) so sketches are exchanged.
        # We force an early gossip by waiting only 35s — enough for one full tick.
        print(f"[sweep] ε={label}  — waiting 35s for gossip to converge ...")
        time.sleep(35)

        # Check that estimated_global_terms on node A reflects both shards.
        try:
            stats = requests.get(f"{url_a}/stats", timeout=3).json()
            est = stats.get("estimated_global_terms", 0)
            print(f"[sweep] ε={label}  — estimated_global_terms={est:.0f} (local only ≈ {mid})")
        except Exception:
            pass

        print(f"[sweep] ε={label}  — running {len(NL_QUERIES)} NL queries on node A ...")
        rr_list: list[float] = []
        lat_list: list[float] = []

        for query, expected in NL_QUERIES:
            t0 = time.perf_counter()
            results = search_nexus(url_a, query, SEARCH_LIMIT, docid_to_name)
            lat_list.append(time.perf_counter() - t0)
            rr_list.append(reciprocal_rank(results, expected))

        n = len(rr_list)
        lat_ms = sorted(x * 1000 for x in lat_list)

        def pct(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = (len(data) - 1) * p / 100
            lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
            return data[lo] + (data[hi] - data[lo]) * (idx - lo)

        return {
            "epsilon": epsilon,
            "label": label,
            "mrr10": sum(rr_list) / n,
            "hits1": sum(1 for r in rr_list if r == 1.0) / n,
            "p50_ms": pct(lat_ms, 50),
            "p95_ms": pct(lat_ms, 95),
        }

    finally:
        stop_node(proc_a, http_a)
        stop_node(proc_b, http_b)


# ---------------------------------------------------------------------------
# ASCII curve
# ---------------------------------------------------------------------------


def ascii_curve(results: list[dict], key: str = "mrr10") -> None:
    values = [r[key] for r in results]
    min_v, max_v = min(values), max(values)
    width = 40
    print()
    for r in results:
        v = r[key]
        bar_len = int((v - min_v) / (max_v - min_v + 1e-9) * width) if max_v > min_v else width // 2
        bar = "█" * bar_len
        label = f"ε={r['label']:>12}"
        print(f"  {label}  {bar:<{width}}  {v:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus ε-DP privacy-utility sweep")
    parser.add_argument(
        "--nexus-bin",
        default="./target/release/nexus",
        help="Path to nexus binary (default: ./target/release/nexus)",
    )
    parser.add_argument("--port", type=int, default=3099, help="HTTP port for node (default: 3099)")
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=EPSILONS,
        help="Epsilon values to sweep",
    )
    args = parser.parse_args()

    nexus_bin = args.nexus_bin
    if not Path(nexus_bin).exists():
        raise SystemExit(
            f"[sweep] Binary not found: {nexus_bin}\n"
            "Run: cargo build -p nexus-core --release"
        )

    print("[sweep] Loading corpus ...")
    docs = fetch_corpus(skip=True)[:CORPUS_SIZE]

    docid_to_name: dict[int, str] = {}
    for doc in docs:
        url = f"https://crates.io/crates/{doc['name']}"
        docid_to_name[fnv1a_32(url)] = doc["name"]

    print(f"[sweep] Corpus: {len(docs)} crates, {len(NL_QUERIES)} NL queries")
    print(f"[sweep] Sweeping ε = {args.epsilons}")
    # Each run: 2 nodes + 35s gossip wait + ~2min indexing/querying ≈ ~3min/ε
    print(f"[sweep] This will take ~{len(args.epsilons) * 3} minutes ...\n")

    baseline_mrr: Optional[float] = None
    results: list[dict] = []

    for epsilon in args.epsilons:
        r = run_one(epsilon, nexus_bin, args.port, docs, docid_to_name)
        if r is None:
            continue
        if math.isclose(epsilon, 1.0):
            baseline_mrr = r["mrr10"]
        results.append(r)

    if not results:
        print("[sweep] No results collected.")
        return

    if baseline_mrr is None:
        baseline_mrr = results[0]["mrr10"]

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 62)
    print("  Nexus ε-DP privacy-utility tradeoff — NL MRR@10")
    print(f"  Corpus: {len(docs):,} crates  |  Queries: {len(NL_QUERIES)}")
    print("=" * 62)
    print(f"  {'ε':>14}  {'MRR@10':>8}  {'Hits@1':>8}  {'Delta':>8}  {'P50ms':>7}")
    print(f"  {'-' * 56}")

    for r in results:
        delta = r["mrr10"] - baseline_mrr
        sign = "+" if delta >= 0 else ""
        baseline_marker = " ← baseline" if math.isclose(r["epsilon"], 1.0) else ""
        print(
            f"  {r['label']:>14}  {r['mrr10']:>8.4f}  {r['hits1']:>8.4f}"
            f"  {sign}{delta:>7.4f}  {r['p50_ms']:>6.1f}ms{baseline_marker}"
        )

    print("=" * 62)
    best = max(results, key=lambda r: r["mrr10"])
    worst = min(results, key=lambda r: r["mrr10"])
    print(f"  Best  MRR@10: ε={best['label']} → {best['mrr10']:.4f}")
    print(f"  Worst MRR@10: ε={worst['label']} → {worst['mrr10']:.4f}")
    print(f"  Range: {best['mrr10'] - worst['mrr10']:.4f}")
    print()
    print("  NOTE: On a single-node benchmark, the DP noise affects only")
    print("  estimated_global_terms (gossip-propagated N). The BM25 IDF")
    print("  uses local doc counts for the main signal. Noise impact grows")
    print("  with corpus size and number of peers.")
    print()

    print("  NL MRR@10 curve (low ε = left, high ε = right):")
    ascii_curve(results, key="mrr10")

    # Save results as JSON for further analysis
    out_path = Path("/tmp/nexus_bench/dp_sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
