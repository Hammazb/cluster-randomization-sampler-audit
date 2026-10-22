"""Exact covariate-constrained target for the Dickinson 16-county benchmark,
rebuilt independently (exact rational scoring, canonical index tie-break)."""
import itertools, json
from fractions import Fraction
import numpy as np
import pandas as pd

J, NT = 16, 8
CUT = 0.10

def load_design():
    df = pd.read_csv('./dickinson_design.csv')
    cols = {
        'urban': (df['location'] == 'Urban').astype(int).to_numpy(),
        'inciis': df['inciis'].to_numpy(),
        'uptodate': df['uptodateonimmunizations'].to_numpy(),
        'hispanic': df['hispanic'].to_numpy(),
        'inc_low': (df['incomecat'] == 'Low').astype(int).to_numpy(),
        'inc_med': (df['incomecat'] == 'Med').astype(int).to_numpy(),
    }
    return df, cols

def allocations():
    """All C(16,8) treated sets in lexicographic order -> (N,8) index array and (N,16) 0/1 matrix."""
    combos = np.array(list(itertools.combinations(range(J), NT)), dtype=np.int8)
    A = np.zeros((len(combos), J), dtype=np.int8)
    rows = np.arange(len(combos))[:, None]
    A[rows, combos] = 1
    return combos, A

def exact_scores(cols, combos):
    """Exact rational l2 balance score BL(z) = sum_j (sum_{i in T} Z_ij)^2, Z standardized (n-1)."""
    n = J
    num = [0] * len(combos)   # placeholder to keep memory modest
    scores = [Fraction(0)] * len(combos)
    for name, x in cols.items():
        x = [int(v) for v in x]
        sx = sum(x)
        mean = Fraction(sx, n)
        var = sum((Fraction(v) - mean) ** 2 for v in x) / (n - 1)
        # sum_T (x - mean) = (n*sum_T x - NT*sx)/n
        xs = np.array(x, dtype=np.int64)
        sT = xs[combos.astype(np.int64)].sum(axis=1)
        Sj = n * sT - NT * sx                      # integer, = n * sum_T (x-mean)
        for k, s in enumerate(Sj.tolist()):
            scores[k] += Fraction(s * s, n * n) / var
    return scores

if __name__ == '__main__':
    df, cols = load_design()
    combos, A = allocations()
    print('allocations', len(combos))
    sc = exact_scores(cols, combos)
    order = sorted(range(len(sc)), key=lambda k: (sc[k], k))
    m = round(len(sc) * CUT)
    acc = order[:m]
    cutoff = sc[order[m - 1]]
    ties = [k for k in range(len(sc)) if sc[k] == cutoff]
    f = np.array([float(s) for s in sc])
    print('m', m, 'min', f.min(), 'mean', f.mean(), 'max', f.max())
    print('cutoff', float(cutoff), 'allocations sharing cutoff:', ties,
          'complements:', [int(''.join(map(str, A[k])), 2) for k in ties])
    # complement check
    comp_idx = {tuple(row): i for i, row in enumerate(A)}
    cmpl = np.array([comp_idx[tuple(1 - A[k])] for k in range(len(A))])
    print('score(z)==score(complement) for all:', all(sc[k] == sc[cmpl[k]] for k in range(len(sc))))
    inA = np.zeros(len(sc), dtype=bool); inA[acc] = True
    acc_sorted = np.array(sorted(acc))   # canonical index order
    print('A closed under complement?', bool(all(inA[cmpl[k]] for k in acc)))
    p1 = A[acc_sorted].mean(axis=0) * 1287
    print('county treated counts (of 1287):', sorted(set(p1.round().astype(int).tolist())))
    np.savez('./target.npz', combos=combos, A=A, scores=f, acc=acc_sorted, inA=inA, cmpl=cmpl)
    json.dump({'min': float(f.min()), 'mean': float(f.mean()), 'max': float(f.max()), 'cutoff': float(cutoff),
               'ties': ties, 'm': m}, open('./target_summary.json', 'w'))
