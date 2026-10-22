"""Extension experiments (new, not in the preprint):
 (1) XY-QAOA ceiling: smallest raw TV the ansatz can reach when TV itself is minimized (diagnostic; uses target knowledge).
 (2) Grover-mixer QAOA with the balance-score phase separator, energy-selected under the same protocol.
 (3) Grover-mixer QAOA ceiling and the indicator-oracle special case (reproduces amplification)."""
import json, numpy as np, qlib as Q
from multiprocessing import Pool

IND = np.zeros(Q.N); IND[Q.acc] = 1.0

def job(args):
    kind, p, seed = args
    rng = np.random.default_rng(31000 + 100 * p + seed + (0 if kind.startswith('xy') else 5000))
    mixer = Q.xy_mixer if kind.startswith('xy') else Q.gm_mixer
    bounds = [(0, np.pi), (0, np.pi / 2 if kind.startswith('xy') else 2 * np.pi)] * p
    x0 = np.array([v for _ in range(p) for v in (rng.uniform(*bounds[0]), rng.uniform(*bounds[1]))])
    if kind.endswith('ceiling'):
        f = lambda x: Q.tv_sub(x, p, mixer)
        r = Q.box_nm(f, x0, bounds, 4000)
    else:
        f = lambda x: Q.energy_sub(x, p, mixer)
        r = Q.box_nm(f, x0, bounds, 300)
    raw, pA, cond = Q.probs_to_metrics_sub(Q.run_sub(r.x, p, mixer))
    return dict(kind=kind, p=p, seed=seed, obj=float(r.fun), raw=float(raw), pA=float(pA), cond=float(cond), x=r.x.tolist())

if __name__ == '__main__':
    jobs = []
    for p in (1, 2, 3, 4):
        jobs += [('xy_ceiling', p, s) for s in range(12)]
    for p in (1, 2, 3):
        jobs += [('gm_energy', p, s) for s in range(10)]
        jobs += [('gm_ceiling', p, s) for s in range(12)]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open('./ext_runs.json', 'w'))
    print('XY ceiling (min raw TV over parameters, 12 NM starts, 4000 evals):')
    for p in (1, 2, 3, 4):
        rs = [r for r in res if r['kind'] == 'xy_ceiling' and r['p'] == p]
        b = min(rs, key=lambda r: r['raw'])
        print(f'  p={p}: best raw TV {b["raw"]:.4f}  P(A)={b["pA"]:.4f}  condTV={b["cond"]:.4f}')
    rs = sorted([r for r in res if r['kind'] == 'gm_energy'], key=lambda r: (r['obj'], r['p'], r['seed']))
    s = rs[0]
    print('GM-QAOA (balance-score phase separator), energy-selected:', {k: round(s[k], 4) if isinstance(s[k], float) else s[k] for k in ('p', 'seed', 'obj', 'raw', 'pA', 'cond')})
    for p in (1, 2, 3):
        e = [r for r in res if r['kind'] == 'gm_energy' and r['p'] == p]
        print(f'  GM energy p={p}: raw TV range [{min(r["raw"] for r in e):.3f},{max(r["raw"] for r in e):.3f}]')
        c = [r for r in res if r['kind'] == 'gm_ceiling' and r['p'] == p]
        b = min(c, key=lambda r: r['raw'])
        print(f'  GM ceiling p={p}: best raw TV {b["raw"]:.4f} P(A)={b["pA"]:.4f} condTV={b["cond"]:.4f}')
    # indicator-oracle special case: gamma=pi, beta=pi, p=2 (+ extras) reproduces amplification
    for p in (1, 2, 3):
        x = np.array([np.pi, np.pi] * p)
        print(f'Indicator phase separator + Grover mixer, gamma=beta=pi, p={p}: raw TV',
              round(Q.probs_to_metrics_sub(Q.run_sub(x, p, Q.gm_mixer, cost=IND))[0], 6))
    print('done')
