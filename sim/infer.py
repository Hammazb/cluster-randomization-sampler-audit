"""Randomization-inference study: type-I error and power of T1/T2 when the reference law Q replaces P*.
All 12,870 allocations are used as the reference domain; Z_obs is drawn from P*; same outcomes for every reference."""
import json, sys, numpy as np
sys.path.insert(0, '.')
import target as T

d = np.load('./target.npz')
A, acc, inA = d['A'].astype(float), d['acc'], d['inA']
N, J, MACC = len(A), 16, len(acc)
df, cols = T.load_design()
xt = (df['inciis'].to_numpy(float) - df['inciis'].mean()) / df['inciis'].std(ddof=1)   # standardized inciis

# T1 = mean(treated) - mean(control); T2 = OLS coefficient on treatment adjusting for xt
# for all allocations at once: T1 = (2 A y - sum y)/8 ; T2 = (A M y)/(a' M a) with M = I - H([1,xt])
Xd = np.column_stack([np.ones(J), xt])
Mres = np.eye(J) - Xd @ np.linalg.inv(Xd.T @ Xd) @ Xd.T
AM = A @ Mres                                   # (N,16)
den = np.einsum('ij,ij->i', AM, A)              # a' M a
assert den.min() > 1e-9

def stats(Y):                                    # Y: (16,R)
    sy = Y.sum(0, keepdims=True)
    T1 = (2 * (A @ Y) - sy) / 8.0
    T2 = (AM @ Y) / den[:, None]
    return T1, T2

def laws():
    L = {}
    pstar = np.zeros(N); pstar[acc] = 1.0 / MACC; L['Exact target'] = pstar
    P = np.load('./swapP.npy')
    q0 = np.zeros(MACC); q0[0] = 1
    qs, t = {}, 0
    cur = q0.copy()
    for t in range(1001):
        if t in (10, 41, 170, 1000):
            qs[t] = cur.copy()
        cur = cur @ P
    for t, q in qs.items():
        v = np.zeros(N); v[acc] = q; L[f'Swap t={t}'] = v
    L['Biased descent'] = np.load('./biased_law.npy')
    return L

CELLS = [
    ('F0 sharp null rho=.05', 0.05, 'null'),
    ('F1a tau=.25 rho=.05', 0.05, 0.25),
    ('F1b tau=.50 rho=.05', 0.05, 0.50),
    ('F1c hetero rho=.05', 0.05, 'het'),
    ('F3a null rho=0', 0.0, 'null'),
    ('F3b null rho=.20', 0.20, 'null'),
    ('F3c tau=.5 rho=0', 0.0, 0.50),
    ('F3d tau=.5 rho=.20', 0.20, 0.50),
]

def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

def run_cell(cell, laws_w8, wrong, R=5000, seed=0, alpha=0.05, chunk=500):
    name, rho, eff = cell
    rng = np.random.default_rng(seed)
    names = list(laws_w8)
    rej = {n: {'T1': 0, 'T2': 0} for n in names}                   # point rejections
    guar = {n: {'T1': 0, 'T2': 0} for n in names}                  # envelope: guaranteed rejection (p_hi <= alpha)
    poss = {n: {'T1': 0, 'T2': 0} for n in names}                  # envelope: possible rejection (p_lo <= alpha)
    dis = {n: {'T1': [0, 0], 'T2': [0, 0]} for n in names}         # [guaranteed, possible] disagreement with P*
    maxdiff = {n: 0.0 for n in names}                              # max |p_lo - p_P*|, |p_hi - p_P*| vs TV check
    pstar_rej = {'T1': 0, 'T2': 0}
    tvext = {}
    P0 = laws_w8['Exact target']
    for n in names:
        q = laws_w8[n]
        tvext[n] = 0.5 * (np.abs(q[acc] - 1.0 / MACC).sum() + (1 - q[acc].sum()))
    done = 0
    while done < R:
        r = min(chunk, R - done)
        a_ = rng.normal(0, np.sqrt(rho), (J, r)) if rho > 0 else np.zeros((J, r))
        e_ = rng.normal(0, np.sqrt((1 - rho) / 50.0), (J, r))
        Y0 = 0.5 * xt[:, None] + a_ + e_
        zobs = acc[rng.integers(0, MACC, r)]                        # drawn from P*
        if eff == 'null': tau = np.zeros((J, 1))
        elif eff == 'het': tau = (0.5 * (1 + 0.5 * xt))[:, None]
        else: tau = np.full((J, 1), float(eff))
        Y = Y0 + tau * A[zobs].T                                    # observed outcomes
        T1, T2 = stats(Y)
        for sname, S in (('T1', T1), ('T2', T2)):
            absS = np.abs(S)
            obs = absS[zobs, np.arange(r)]
            ext = (absS >= (obs - 1e-10)[None, :]).astype(float)    # (N,r) extreme indicator
            pP = P0 @ ext
            for n in names:
                plo = laws_w8[n] @ ext
                phi = plo + wrong[n]
                pr = (pP <= alpha)
                if wrong[n] == 0:
                    rej_n = (plo <= alpha)
                    rej[n][sname] += int(rej_n.sum())
                    dis[n][sname][0] += int((rej_n != pr).sum()); dis[n][sname][1] += int((rej_n != pr).sum())
                    guar[n][sname] += int(rej_n.sum()); poss[n][sname] += int(rej_n.sum())
                else:
                    g = phi <= alpha; ps = plo <= alpha
                    guar[n][sname] += int(g.sum()); poss[n][sname] += int(ps.sum())
                    dis[n][sname][0] += int(((pr & ~ps) | (~pr & g)).sum())
                    dis[n][sname][1] += int(((pr & ~g) | (~pr & ps)).sum())
                maxdiff[n] = max(maxdiff[n], float(np.abs(plo - pP).max()), float(np.abs(phi - pP).max()))
            pstar_rej[sname] += int((pP <= alpha).sum())
        done += r
    out = {}
    for n in names:
        out[n] = {}
        for s in ('T1', 'T2'):
            lo_, hi_ = wilson(poss[n][s], R)
            out[n][s] = dict(rej=rej[n][s] / R, guaranteed=guar[n][s] / R, possible=poss[n][s] / R,
                             ci=[float(x) for x in wilson(rej[n][s], R)],
                             dis_g=dis[n][s][0] / R, dis_p=dis[n][s][1] / R,
                             dis_ci=[float(x) for x in wilson(dis[n][s][0], R)])
        out[n]['TVext'] = float(tvext[n]); out[n]['maxdiff'] = maxdiff[n]
    return out
