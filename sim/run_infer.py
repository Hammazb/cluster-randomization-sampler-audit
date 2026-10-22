import json, sys, time, numpy as np
sys.path.insert(0, '.')
import infer as I, qlib as Q

L = I.laws()
wrong = {n: 0.0 for n in L}

# Grover amplification t*=2
ps = Q.dicke()
for _ in range(2):
    ps[Q.acc] *= -1
    amp = ps.sum() / np.sqrt(Q.N)
    ps = 2 * amp * np.ones(Q.N) / np.sqrt(Q.N) - ps
L['Amplification t*=2'] = np.abs(ps) ** 2; wrong['Amplification t*=2'] = 0.0

# XY-QAOA headline (energy-selected)
xy = sorted(json.load(open('./xy_runs.json')), key=lambda r: (r['energy'], r['p'], r['seed']))[0]
psi = Q.run_sub(np.array(xy['x']), xy['p'], Q.xy_mixer)
L['XY-QAOA raw'] = np.abs(psi) ** 2; wrong['XY-QAOA raw'] = 0.0

# Penalty-QAOA headline lambda=48 (energy-selected)
pen = sorted([r for r in json.load(open('./pen_runs.json')) if r['lam'] == 48],
             key=lambda r: (r['energy'], r['p'], r['seed']))[0]
Cn = Q.pen_cost(48)[1]
pr = np.abs(Q.run_pen(np.array(pen['x']), pen['p'], Cn)) ** 2
L['Penalty-QAOA raw'] = pr[Q.masks]; wrong['Penalty-QAOA raw'] = float(1 - pr[Q.masks].sum())

tv = {}
for n, q in L.items():
    tv[n] = 0.5 * (np.abs(q[Q.acc] - 1.0 / Q.MACC).sum() + (1 - q[Q.acc].sum()))
print('raw TV by law:', {k: round(v, 6) for k, v in tv.items()})

R = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
res = {}
for ci, cell in enumerate(I.CELLS):
    t0 = time.time()
    res[cell[0]] = I.run_cell(cell, L, wrong, R=R, seed=20260 + ci)
    print(cell[0], 'done in %.0fs' % (time.time() - t0), flush=True)
json.dump(dict(tv=tv, wrong=wrong, cells=res, R=R), open('./infer_results.json', 'w'), indent=1)
