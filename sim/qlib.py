"""Numpy statevector simulation of the 16-qubit sampler arms (no Qiskit needed).
Conventions: qubit i <-> bit i of the basis index; allocation index k <-> weight-8 bitstring masks[k]."""
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.insert(0, '.')
import target as T

d = np.load('./target.npz')
A, acc, scores = d['A'], d['acc'], d['scores']
N, J = len(A), 16
MACC = len(acc)
masks = (A.astype(np.int64) << np.arange(J)).sum(axis=1)          # weight-8 basis indices (N,)
inA = d['inA']

# standardized design for the quadratic extension to all 2^16 strings
df, cols = T.load_design()
Z = np.column_stack([(np.array(v, float) - np.mean(v)) / np.std(v, ddof=1) for v in cols.values()])
allx = np.arange(1 << J)
X = ((allx[:, None] >> np.arange(J)) & 1).astype(float)           # (65536,16)
BLall = ((X @ Z) ** 2).sum(axis=1)
wt = X.sum(axis=1)
assert np.allclose(BLall[masks], scores)                          # extension agrees with the exact score on weight 8

BLmin, BLmax = scores.min(), scores.max()
Bscale = scores.mean()

# ---- subspace (weight-8) objects ----
sub_of = -np.ones(1 << J, dtype=np.int64); sub_of[masks] = np.arange(N)
E0 = [(2 * i, 2 * i + 1) for i in range(8)]
E1 = [(2 * i + 1, (2 * i + 2) % 16) for i in range(8)]
edge_pairs = []
for (a, b) in E0 + E1:
    i01 = [k for k in range(N) if ((masks[k] >> a) & 1) == 0 and ((masks[k] >> b) & 1) == 1]
    i01 = np.array(i01)
    i10 = sub_of[masks[i01] ^ (1 << a) ^ (1 << b)]
    edge_pairs.append((i01, i10))

def dicke():
    return np.ones(N, complex) / np.sqrt(N)

def xy_mixer(psi, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for i01, i10 in edge_pairs:
        v01, v10 = psi[i01].copy(), psi[i10].copy()
        psi[i01] = c * v01 + s * v10
        psi[i10] = c * v10 + s * v01
    return psi

def gm_mixer(psi, beta):
    # exp(-i beta |D><D|) on the weight-8 subspace, |D> uniform
    amp = psi.sum() / np.sqrt(N)
    return psi + (np.exp(-1j * beta) - 1) * amp * (np.ones(N) / np.sqrt(N))

Cxy = (scores - BLmin) / (BLmax - BLmin)                           # normalized to [0,1] on weight 8

def run_sub(params, p, mixer, cost=Cxy):
    psi = dicke()
    for l in range(p):
        g, b = params[2 * l], params[2 * l + 1]
        psi = psi * np.exp(-1j * g * cost)
        psi = mixer(psi, b)
    return psi

def probs_to_metrics_sub(psi):
    """psi on weight-8 subspace -> (raw TV, P(A), cond TV)."""
    p = np.abs(psi) ** 2
    pa = p[acc]
    pA = pa.sum()
    raw = 0.5 * (np.abs(pa - 1.0 / MACC).sum() + (1 - pA))
    cond = 0.5 * np.abs(pa / pA - 1.0 / MACC).sum()
    return raw, pA, cond

def energy_sub(params, p, mixer, cost=Cxy):
    psi = run_sub(params, p, mixer, cost)
    return float(np.sum(np.abs(psi) ** 2 * cost))

def tv_sub(params, p, mixer, cost=Cxy):
    return probs_to_metrics_sub(run_sub(params, p, mixer, cost))[0]

# ---- full-space penalty arm ----
def pen_cost(lam):
    c = BLall + lam * (wt - 8) ** 2
    return c, (c - c.min()) / (c.max() - c.min())

def rx_mixer_full(psi, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(J):
        a = psi.reshape(-1, 2, 1 << q)
        psi = (c * a + s * a[:, ::-1, :]).reshape(-1)
    return psi

def run_pen(params, p, Cn):
    psi = np.ones(1 << J, complex) / np.sqrt(1 << J)
    for l in range(p):
        g, b = params[2 * l], params[2 * l + 1]
        psi = psi * np.exp(-1j * g * Cn)
        psi = rx_mixer_full(psi, b)
    return psi

def pen_metrics(psi):
    pr = np.abs(psi) ** 2
    pa = pr[masks[acc]]
    pA = pa.sum()
    pw = pr[wt == 8].sum()
    raw = 0.5 * (np.abs(pa - 1.0 / MACC).sum() + (1 - pA))
    cond = 0.5 * np.abs(pa / pA - 1.0 / MACC).sum()
    return dict(raw=raw, pA=pA, p_w8_not_acc=pw - pA, p_wrong=1 - pw, cond=cond)

def box_nm(fun, x0, bounds, maxfev):
    r = minimize(fun, x0, method='Nelder-Mead', bounds=bounds, options=dict(maxfev=maxfev, xatol=1e-6, fatol=1e-9))
    return r

def rand_start(rng, p):
    return np.array([v for _ in range(p) for v in (rng.uniform(0, np.pi), rng.uniform(0, np.pi / 2))])
