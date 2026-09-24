"""E5: cost of the SODA alignment term vs reservoir size K, feature dim d', and #domains N.

Times forward+backward of  mean_i OT_eps(z_i, Q)  (GeomLoss sinkhorn, p=2) for random
unit-norm features, B=32 per domain, and reports peak extra GPU memory and the length of
GeomLoss' epsilon-scaling schedule (= number of Sinkhorn iterations).

    python -m domainbed.scripts.bench_scalability --out bench.json
"""
import argparse
import json
import time

import geomloss
import torch
import torch.nn.functional as F
from geomloss.sinkhorn_divergence import epsilon_schedule


def bench_one(N, B, K, d, blur, debias, backend, reps=20, warmup=3, device="cuda"):
    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=debias,
                                   backend=backend)
    Q = F.normalize(torch.randn(K, d, device=device), dim=1)
    z = torch.randn(N * B, d, device=device, requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for r in range(warmup + reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        zn = F.normalize(z, dim=1)
        loss = torch.stack([loss_fn(zn[i * B:(i + 1) * B], Q) for i in range(N)]).mean()
        g, = torch.autograd.grad(loss, z)
        torch.cuda.synchronize()
        if r >= warmup:
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() - base
    times.sort()
    return {"N": N, "B": B, "K": K, "d": d, "blur": blur, "debias": debias, "backend": backend,
            "ms_median": 1e3 * times[len(times) // 2], "ms_min": 1e3 * times[0],
            "peak_extra_mb": peak / 2 ** 20,
            "sinkhorn_iters": len(epsilon_schedule(2, 2.0, blur, 0.5))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--Ks", default="128,256,512,1024,2048,4096")
    p.add_argument("--ds", default="512,2048")
    p.add_argument("--Ns", default="2,3,5")
    p.add_argument("--blurs", default="0.1,1.0")
    p.add_argument("--backends", default="tensorized,online")
    args = p.parse_args()
    rows = []
    for backend in args.backends.split(","):
        for blur in map(float, args.blurs.split(",")):
            for d in map(int, args.ds.split(",")):
                for N in map(int, args.Ns.split(",")):
                    for K in map(int, args.Ks.split(",")):
                        for debias in (False, True):
                            try:
                                r = bench_one(N, 32, K, d, blur, debias, backend)
                            except Exception as e:  # e.g. pykeops unavailable for "online"
                                r = {"N": N, "K": K, "d": d, "blur": blur, "debias": debias,
                                     "backend": backend, "error": repr(e)[:200]}
                            print(json.dumps(r), flush=True)
                            rows.append(r)
                            torch.cuda.empty_cache()
    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "geomloss": geomloss.__version__, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
