"""Classical samplers: swap chain on the accepted set and biased steepest-descent control."""
import json
from fractions import Fraction
import numpy as np
import scipy.linalg as sla
import sys
sys.path.insert(0, '.')
import target as T

d = np.load('./target.npz')
combos, A, acc, inA = d['combos'], d['A'], d['acc'], d['inA']
N = len(A)
masks = (A.astype(np.int64) << np.arange(16)).sum(axis=1)
mask2idx = {int(m): i for i, m in enumerate(masks)}

df, cols = T.load_design()
sc = T.exact_scores(cols, combos)           # exact Fractions

def neighbors(k):
    """Indices of the 64 allocations reachable by one treated-untreated swap."""
    row = A[k]
    tr = np.where(row == 1)[0]; un = np.where(row == 0)[0]
    out = []
    m = int(masks[k])
    for i in tr:
        for u in un:
            out.append(mask2idx[m ^ (1 << int(i)) ^ (1 << int(u))])
    return out

# ---------------- swap chain ----------------
pos = {int(a): j for j, a in enumerate(acc)}
M = len(acc)
P = np.zeros((M, M))
for j, a in enumerate(acc):
    stay = 0
    for nb in neighbors(int(a)):
        if nb in pos:
            P[j, pos[nb]] += 1
        else:
            stay += 1
    P[j, j] += stay
P /= 64.0
sym = np.allclose(P, P.T)
selfloops = np.diag(P) * 64
print('chain symmetric:', sym, ' row sums all 1:', np.allclose(P.sum(1), 1),
      ' self-loop count range:', int(selfloops.min()), int(selfloops.max()))
# connectivity
from scipy.sparse.csgraph import connected_components
ncomp, _ = connected_components((P > 0).astype(int), directed=False)
print('components:', ncomp)
ev = np.sort(sla.eigvalsh((P + P.T) / 2))[::-1]
print('lambda2', ev[1], 'spectral gap', 1 - ev[1], 'lambda_min', ev[-1])

def tv_curve(start, steps):
    q = np.zeros(M); q[start] = 1.0
    out = {}; first25 = first01 = None
    for t in range(max(steps) + 1):
        tv = 0.5 * np.abs(q - 1.0 / M).sum()
        if t in steps: out[t] = tv
        if first25 is None and tv < 0.25: first25 = t
        if first01 is None and tv < 0.01: first01 = t
        q = q @ P
    return out, first25, first01

steps = [0, 10, 41, 170, 1000]
out, f25, f01 = tv_curve(0, steps)
print('start state 0 (first accepted allocation in index order):', {k: float(v) for k, v in out.items()}, 'first<0.25 at', f25, 'first<0.01 at', f01)
# sensitivity to the start state
f25s, f01s = [], []
for s in range(M):
    _, a, b = tv_curve(s, [0]) if False else (None, None, None)
# cheaper: iterate all starts simultaneously
Q = np.eye(M)
f25a = np.full(M, -1); f01a = np.full(M, -1)
for t in range(0, 1201):
    tv = 0.5 * np.abs(Q - 1.0 / M).sum(axis=1)
    f25a[(f25a < 0) & (tv < 0.25)] = t
    f01a[(f01a < 0) & (tv < 0.01)] = t
    Q = Q @ P
print('over all 1287 starts: first<0.25 range', int(f25a.min()), int(f25a.max()), ' first<0.01 range', int(f01a.min()), int(f01a.max()))

# ---------------- biased steepest descent ----------------
nbr_cache = {}
def nb(k):
    if k not in nbr_cache: nbr_cache[k] = neighbors(k)
    return nbr_cache[k]

def descend(k):
    while True:
        cur = sc[k]
        best, bs = None, cur
        for j in nb(k):
            if sc[j] < bs or (sc[j] == bs and best is not None and j < best):
                best, bs = j, sc[j]
        if best is None or bs >= cur:
            return k
        k = best

end = np.array([descend(k) for k in range(N)])
vals, cnt = np.unique(end, return_counts=True)
Qb = {int(v): Fraction(int(c), N) for v, c in zip(vals, cnt)}
inA_set = {int(a) for a in acc}
outside = sum(q for v, q in Qb.items() if v not in inA_set)
tv_acc = sum(abs(Qb.get(int(a), Fraction(0)) - Fraction(1, 1287)) for a in acc)
tvb = (tv_acc + outside) / 2
print('biased descent: distinct local minima', len(vals), ' mass outside A:', float(outside),
      ' exact TV', tvb, '=', float(tvb))
json.dump({'swap_tv': {str(k): float(v) for k, v in out.items()}, 'first25': f25, 'first01': f01,
           'gap': float(1 - ev[1]), 'self_loop_min': int(selfloops.min()), 'self_loop_max': int(selfloops.max()),
           'biased_minima': int(len(vals)), 'biased_tv': float(tvb), 'biased_tv_frac': str(tvb),
           'start_range_25': [int(f25a.min()), int(f25a.max())], 'start_range_01': [int(f01a.min()), int(f01a.max())]},
          open('./classical_summary.json', 'w'), indent=1)
np.save('./swapP.npy', P)
bq = np.zeros(N);
for v, q in Qb.items(): bq[v] = float(q)
np.save('./biased_law.npy', bq)
