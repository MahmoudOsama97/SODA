"""E2: pivot-induced variance of the SODA alignment gradient (two-way design).

For a saved SODA checkpoint, fix a source minibatches S_u (u=1..a) and, for each u, draw b
pivots P_uv independently of S_u under several pivot mechanisms. With
g_uv = grad_Z L_align(S_u, P_uv),  L_align = mean_i OT_eps(z_i, P)  (z_i: domain i of S_u),

  pivot variance   V_piv = mean_u  1/(b-1) sum_v ||g_uv - gbar_u||^2      (E_S tr Var_P(g|S))
  source variance  V_src = 1/(a-1) sum_u ||gbar_u - gbar||^2 - V_piv/b     (ANOVA-unbiased)
  relative         V_piv / mean_u ||gbar_u||^2
  cosine           mean pairwise cos(g_uv, g_uv') within u
  MSE to reference mean_uv ||g_uv - g_u^ref||^2, g_u^ref with a K_ref fresh pivot; bias2 part

Mechanisms: dfr_K (K fresh features from ceil(K/m) training-like batches), single (one source
domain of a separate batch, 32 pts), pooled (a separate full batch, m pts). Features are encoded
as in training (train-mode BN on whole batches, running stats untouched, dropout off).

    python -m domainbed.scripts.probe_grad_variance --ckpt .../ckpt_step01000.pt \
        --data_dir $SLURM_TMPDIR/data --dataset OfficeHome --test_env 0 --out e2_step1000.json
"""
import argparse
import copy
import json
import math

import geomloss
import numpy as np
import torch
import torch.nn.functional as F

from domainbed import algorithms, datasets
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader


def load_alg(path, device):
    d = torch.load(path, map_location="cpu", weights_only=False)
    hp = copy.deepcopy(d["hparams"])
    hp.update({"probe_every": 0, "dump_stream": False, "save_ckpt_steps": ""})
    alg = algorithms.SODA(d["input_shape"], d["num_classes"], d["num_domains"], hp)
    alg.load_state_dict(d["state_dict"])
    return alg.to(device), hp, d["step"]


def stats(G, ref):
    """G: [a, b, P] gradients, ref: [a, P] reference gradients."""
    a, b, _ = G.shape
    gbar_u = G.mean(1)                                            # [a, P]
    v_piv = ((G - gbar_u[:, None]) ** 2).sum(-1).sum(1).div(b - 1).mean().item()
    gbar = gbar_u.mean(0)
    v_src = ((gbar_u - gbar) ** 2).sum(-1).sum().div(a - 1).item() - v_piv / b
    norm2 = (gbar_u ** 2).sum(-1).mean().item()
    Gn = F.normalize(G, dim=-1)
    cos = torch.einsum("abp,acp->abc", Gn, Gn)
    off = ~torch.eye(b, dtype=torch.bool, device=G.device)
    mse_ref = ((G - ref[:, None]) ** 2).sum(-1).mean().item()
    bias2 = ((gbar_u - ref) ** 2).sum(-1).mean().item()
    return {"V_piv": v_piv, "V_src": v_src, "rel_V_piv": v_piv / max(norm2, 1e-30),
            "mean_grad_norm2": norm2, "cos_within_u": cos[:, off].mean().item(),
            "mse_to_ref": mse_ref, "bias2_to_ref": bias2, "rel_mse_to_ref": mse_ref / max(norm2, 1e-30)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset", default="OfficeHome")
    p.add_argument("--test_env", type=int, default=0)
    p.add_argument("--trial_seed", type=int, default=0)
    p.add_argument("--a", type=int, default=16)
    p.add_argument("--b", type=int, default=16)
    p.add_argument("--Ks", default="128,256,512,1024")
    p.add_argument("--K_ref", type=int, default=4096)
    p.add_argument("--blurs", default="0.1,1.0")
    p.add_argument("--param_grad", action="store_true", help="also gradients w.r.t. layer4 params")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    alg, hp, step = load_alg(args.ckpt, device)
    dataset = vars(datasets)[args.dataset](args.data_dir, [args.test_env], hp)
    loaders = []
    for env_i, env in enumerate(dataset):
        if env_i == args.test_env:
            continue
        _, in_ = misc.split_dataset(env, int(len(env) * 0.2), misc.seed_hash(args.trial_seed, env_i))
        loaders.append(iter(InfiniteDataLoader(dataset=in_, weights=None,
                                               batch_size=hp["batch_size"], num_workers=4)))
    N, B = len(loaders), hp["batch_size"]
    m = N * B

    def next_batch():
        xs = [next(it)[0] for it in loaders]
        return torch.cat(xs).to(device)

    Ks = [int(k) for k in args.Ks.split(",")]
    n_ref = math.ceil(args.K_ref / m)
    n_pool = max(args.b * math.ceil(max(Ks) / m), args.b) + n_ref + 8
    print(f"step={step} N={N} B={B} m={m} encoding pool of {n_pool} batches", flush=True)
    pool = torch.stack([alg._encode_like_training(next_batch()) for _ in range(n_pool)])  # [n_pool, m, d]
    ref_pool = pool[:n_ref].reshape(-1, pool.shape[-1])[:args.K_ref]
    draw_pool = pool[n_ref:]

    mechanisms = [f"dfr_{k}" for k in Ks] + ["single", "pooled"]
    params = list(alg.featurizer.network.layer4.parameters()) if args.param_grad else []
    results = {"ckpt": args.ckpt, "step": step, "N": N, "B": B, "m": m, "a": args.a, "b": args.b,
               "K_ref": args.K_ref, "hparams": {k: hp[k] for k in ("ema_pivot_size", "ot_epsilon",
               "ot_debias", "lambda_ot_align", "lr")}, "by_blur": {}}

    sources = [next_batch() for _ in range(args.a)]
    for blur in [float(x) for x in args.blurs.split(",")]:
        loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=hp.get("ot_debias", False))

        def L(z, P):
            return torch.stack([loss_fn(z[i * B:(i + 1) * B], P) for i in range(N)]).mean()

        G = {k: [] for k in mechanisms}
        Gp = {k: [] for k in mechanisms}
        refs, refs_p = [], []
        for u, xu in enumerate(sources):
            zu = alg._encode_like_training(xu).requires_grad_(True)
            if params:  # graph through layer4 for parameter gradients
                with torch.enable_grad():
                    saved = [(mod, mod.momentum) for mod in alg.featurizer.modules()
                             if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
                    for mod, _ in saved:
                        mod.momentum = 0.0
                    zp = F.normalize(alg.featurizer(xu), dim=1)
                    for mod, mom in saved:
                        mod.momentum = mom

            def grads(P):
                with torch.enable_grad():
                    g, = torch.autograd.grad(L(zu, P), zu)
                    gp = None
                    if params:
                        gp = torch.autograd.grad(L(zp, P), params, retain_graph=True)
                        gp = torch.cat([t.flatten() for t in gp])
                return g.flatten(), gp

            g, gp = grads(ref_pool)
            refs.append(g); refs_p.append(gp)
            order = rng.permutation(draw_pool.shape[0])
            for k in mechanisms:
                gs, gps = [], []
                for v in range(args.b):
                    if k.startswith("dfr_"):
                        K = int(k.split("_")[1]); nb = math.ceil(K / m)
                        idx = order[(v * nb) % len(order): (v * nb) % len(order) + nb]
                        if len(idx) < nb:
                            idx = rng.choice(draw_pool.shape[0], nb, replace=False)
                        P = draw_pool[idx].reshape(-1, draw_pool.shape[-1])[:K]
                    elif k == "single":
                        j = rng.randint(N)
                        P = draw_pool[order[v % len(order)], j * B:(j + 1) * B]
                    else:
                        P = draw_pool[order[v % len(order)]]
                    g, gp = grads(P)
                    gs.append(g); gps.append(gp)
                G[k].append(torch.stack(gs))
                if params:
                    Gp[k].append(torch.stack(gps))
            print(f"blur={blur} u={u} done", flush=True)
        ref = torch.stack(refs)
        out = {k: stats(torch.stack(G[k]), ref) for k in mechanisms}
        if params:
            ref_p = torch.stack(refs_p)
            out_p = {k: stats(torch.stack(Gp[k]), ref_p) for k in mechanisms}
            out = {"feature_space": out, "layer4_params": out_p}
        else:
            out = {"feature_space": out}
        results["by_blur"][str(blur)] = out
        print(json.dumps({"blur": blur, **{k: out["feature_space"][k]["rel_V_piv"] for k in mechanisms}}), flush=True)

    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
