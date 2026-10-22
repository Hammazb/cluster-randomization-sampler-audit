"""Penalty-QAOA arm (lambda = 48 headline; 12 and 192 diagnostic): |+>^16, RX mixer, NM 300 evals, 10 starts/depth."""
import json, numpy as np, qlib as Q
from multiprocessing import Pool

def one(args):
    lam, p, seed = args
    Cn = Q.pen_cost(lam)[1]
    rng = np.random.default_rng(7000 + 100 * p + seed)
    x0 = Q.rand_start(rng, p)
    bounds = [(0, np.pi), (0, np.pi / 2)] * p
    def en(x):
        psi = Q.run_pen(x, p, Cn)
        return float(np.sum(np.abs(psi) ** 2 * Cn))
    r = Q.box_nm(en, x0, bounds, 300)
    m = Q.pen_metrics(Q.run_pen(r.x, p, Cn))
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    return dict(lam=lam, p=p, seed=seed, energy=float(r.fun), nfev=int(r.nfev), stagnated=bool(r.nfev < 300),
                at_bound=bool(np.any(np.isclose(r.x, lo) | np.isclose(r.x, hi))),
                x=r.x.tolist(), **{k: float(v) for k, v in m.items()})

if __name__ == '__main__':
    jobs = [(lam, p, s) for lam in (48, 12, 192) for p in (1, 2, 3) for s in range(10)]
    with Pool(2) as pool:
        res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open('./pen_runs.json', 'w'))
    for lam in (48, 12, 192):
        rs = sorted([r for r in res if r['lam'] == lam], key=lambda r: (r['energy'], r['p'], r['seed']))
        s = rs[0]
        print(f"lambda={lam} selected: p={s['p']} seed={s['seed']} energy={s['energy']:.4f} rawTV={s['raw']:.4f} "
              f"P(A)={s['pA']:.4f} P(w8 not acc)={s['p_w8_not_acc']:.4f} P(wrong)={s['p_wrong']:.4f} condTV={s['cond']:.4f}")
    print('done')
