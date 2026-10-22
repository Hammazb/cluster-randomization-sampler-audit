"""XY-QAOA (energy-selected, preprint protocol): Nelder-Mead, 300 evals, 10 starts per depth p=1..3."""
import json, numpy as np, qlib as Q
from multiprocessing import Pool

BOUNDS_CACHE = {}
def one(args):
    p, seed = args
    rng = np.random.default_rng(1000 * p + seed)
    x0 = Q.rand_start(rng, p)
    bounds = [(0, np.pi), (0, np.pi / 2)] * p
    r = Q.box_nm(lambda x: Q.energy_sub(x, p, Q.xy_mixer), x0, bounds, 300)
    raw, pA, cond = Q.tv_sub(r.x, p, Q.xy_mixer), None, None
    raw, pA, cond = Q.probs_to_metrics_sub(Q.run_sub(r.x, p, Q.xy_mixer))
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    atb = bool(np.any(np.isclose(r.x, lo) | np.isclose(r.x, hi)))
    return dict(p=p, seed=seed, energy=float(r.fun), nfev=int(r.nfev), stagnated=bool(r.nfev < 300),
                at_bound=atb, raw=float(raw), pA=float(pA), cond=float(cond), x=r.x.tolist())

if __name__ == '__main__':
    jobs = [(p, s) for p in (1, 2, 3) for s in range(10)]
    with Pool(2) as pool:
        res = pool.map(one, jobs)
    res.sort(key=lambda r: (r['energy'], r['p'], r['seed']))
    sel = res[0]
    print('selected by energy:', {k: sel[k] for k in ('p', 'seed', 'energy', 'raw', 'pA', 'cond')})
    for p in (1, 2, 3):
        rs = [r for r in res if r['p'] == p]
        print(f'p={p}: energy range [{min(r["energy"] for r in rs):.4f},{max(r["energy"] for r in rs):.4f}] '
              f'raw TV range [{min(r["raw"] for r in rs):.3f},{max(r["raw"] for r in rs):.3f}] '
              f'P(A) range [{min(r["pA"] for r in rs):.3f},{max(r["pA"] for r in rs):.3f}] '
              f'stagnated {sum(r["stagnated"] for r in rs)}/10 at-bound {sum(r["at_bound"] for r in rs)}/10')
    print('mean energy of uniform Dicke = ', float(Q.Cxy.mean()))
    json.dump(res, open('./xy_runs.json', 'w'))
