"""E3: pivot drift from a logged feature stream (exact W2 via assignment; CPU).

Reads <probe_dir>/stream_*.pt (per-step normalized batch features, labels, domains, written by
SODA with dump_stream=true) and queue_*.pt (the algorithm's actual reservoir every 100 steps).

For step t (every --every steps) and each K it computes
  A_K   W2(Q_t, Q_{t+1})   FIFO reservoir, consecutive steps (overlap K - m)
  B_K   W2(R_t, R'_t)      two *disjoint* fresh size-K pivots from adjacent windows of
                           ceil(K/m) steps (no overlap; same sample size as A_K)
  C_96  W2(batch_t, batch_{t+1})             pooled batch pivot (m points)
  C_32  W2(domain_j of batch_t, of batch_{t+1})  single-domain batch pivot (B points)
A_K vs B_K isolates the overlap effect at equal size; B_K vs C isolates the sample-size effect.
The Prop. 3.3 bound 2*sqrt(m/K) is reported alongside A_K.
It first checks that an exact FIFO simulation reproduces the dumped reservoir.
"""
import argparse
import glob
import json
import math
import os
from multiprocessing import Pool

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def w2(X, Y):
    """Exact W2 between two uniform empirical measures of equal size: for equal sizes and
    uniform weights the optimal coupling is a permutation (Birkhoff), i.e. an assignment problem."""
    assert len(X) == len(Y), (len(X), len(Y))
    M = cdist(X.astype(np.float64), Y.astype(np.float64), "sqeuclidean")
    r, c = linear_sum_assignment(M)
    return float(math.sqrt(max(M[r, c].mean(), 0.0)))


def load_stream(probe_dir):
    chunks = []
    for f in sorted(glob.glob(os.path.join(probe_dir, "stream_*.pt"))):
        chunks += torch.load(f, weights_only=False)
    chunks.sort(key=lambda r: r[0])
    steps = [r[0] for r in chunks]
    assert steps == list(range(steps[0], steps[0] + len(steps))), "stream has gaps"
    feats = torch.stack([r[1] for r in chunks]).float().numpy()   # [T, m, d]
    doms = torch.stack([r[3] for r in chunks]).numpy()             # [T, m]
    return steps[0], feats, doms


def fifo_after(feats, t, K):
    """Reservoir (as a set of rows) after inserting batches 0..t, K most recent rows."""
    m = feats.shape[1]
    nb = math.ceil(K / m)
    lo = max(0, t - nb + 1)
    rows = feats[lo:t + 1].reshape(-1, feats.shape[-1])
    return rows[-K:] if rows.shape[0] >= K else None


def check_against_dumps(probe_dir, feats, t0, K):
    """Exact ring-buffer simulation (slot positions) compared with queue_*.pt dumps."""
    m, d = feats.shape[1], feats.shape[2]
    report = []
    for f in sorted(glob.glob(os.path.join(probe_dir, "queue_*.pt")))[:5]:
        dump = torch.load(f, weights_only=False)
        s = dump["step"]
        if s - t0 + 1 < math.ceil(K / m) or s - t0 >= len(feats):
            continue
        Q = np.zeros((K, d), dtype=np.float32); ptr = 0
        for t in range(0, s - t0 + 1):
            slots = [(ptr + i) % K for i in range(m)]
            Q[slots] = feats[t]
            ptr = (ptr + m) % K
        err = float(np.abs(Q - dump["queue"].float().numpy()).max())
        report.append({"step": s, "max_abs_err": err, "ptr_ok": ptr == dump["ptr"]})
    return report


G = {}


def job(t):
    feats, doms, Ks = G["feats"], G["doms"], G["Ks"]
    m = feats.shape[1]
    out = {"t": t}
    for K in Ks:
        nb = math.ceil(K / m)
        Qa, Qb = fifo_after(feats, t - 1, K), fifo_after(feats, t, K)
        if Qa is not None:
            out[f"A_{K}"] = w2(Qa, Qb)
            out[f"bound_{K}"] = 2 * math.sqrt(min(m, K) / K)
        R1 = fifo_after(feats, t, K); R0 = fifo_after(feats, t - nb, K)
        if R1 is not None and R0 is not None:
            out[f"B_{K}"] = w2(R0, R1)
    out["C_96"] = w2(feats[t], feats[t + 1])
    j = 0
    out["C_32"] = w2(feats[t][doms[t] == j], feats[t + 1][doms[t + 1] == j])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe_dir", required=True)
    p.add_argument("--Ks", default="128,256,512,1024")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    Ks = [int(k) for k in args.Ks.split(",")]
    t0, feats, doms = load_stream(args.probe_dir)
    m = feats.shape[1]
    print(f"stream: steps {t0}..{t0 + len(feats) - 1}, m={m}, d={feats.shape[2]}", flush=True)
    check = check_against_dumps(args.probe_dir, feats, t0, int(os.environ.get("K_RUN", 512)))
    print("fifo check:", check, flush=True)
    start = 2 * math.ceil(max(Ks) / m) + 1
    ts = list(range(start, len(feats) - 1, args.every))
    G.update(feats=feats, doms=doms, Ks=Ks)   # inherited by forked workers, not pickled
    with Pool(args.procs) as pool:
        rows = pool.map(job, ts, chunksize=4)
    for r in rows:
        r["step"] = r.pop("t") + t0
    json.dump({"m": m, "Ks": Ks, "fifo_check": check, "rows": rows}, open(args.out, "w"))
    print("done", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
