"""8-cluster (4-vs-4, 8 qubits) hardware check of XY-QAOA.  Modes:
  python small_hw.py prep            offline: build target, train angles, ideal TV, transpiled gate counts (FakeFez)
  python small_hw.py run BACKEND N   submit p=0..3 circuits, N shots each, save counts to hw_counts.json
Requires: qiskit, qiskit-aer, qiskit-ibm-runtime, scipy, numpy, pandas; for `run`, a saved IBM account."""
import sys, json, itertools, numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
sys.path.insert(0, '.')
import target as T

J = 8
df, cols = T.load_design()
X = np.column_stack([np.asarray(v, float) for v in cols.values()])[:J]
X = X[:, X.std(0) > 1e-12]                                  # drop columns that are constant in these 8 clusters
Z = (X - X.mean(0)) / X.std(0, ddof=1)
combos = [c for c in itertools.combinations(range(J), 4)]
def score(c):
    m = np.zeros(J); m[list(c)] = 1
    return float(((m @ Z) ** 2).sum())
sc = np.array([score(c) for c in combos])
order = np.argsort(sc, kind='stable')
acc_idx = order[:8]
assert sc[order[7]] < sc[order[8]] - 1e-9, 'tie at the acceptance boundary'
bit = lambda c: sum(1 << i for i in c)                      # qubit i = cluster i
acc_bits = {bit(combos[i]) for i in acc_idx}
assert all((255 ^ b) in acc_bits for b in acc_bits), 'accepted set not complement-closed'
G = Z @ Z.T                                                 # BL(x) = x^T G x, x_i in {0,1}
w8 = [bit(c) for c in combos]
tgt = np.zeros(256); tgt[list(acc_bits)] = 1 / 8

def dicke():
    v = np.zeros(256, complex); v[w8] = 1 / np.sqrt(70); return v

def build(params, p, measure=True):
    qc = QuantumCircuit(J)
    for i in range(1, J, 2): qc.x(i)                         # Neel start |01010101>, weight 4 (shallow stand-in for the Dicke start)
    # BL = x^T G x with x=(1-s)/2; on the weight-4 subspace single-Z terms are constant, so only the ZZ terms matter
    for l in range(p):
        g, b = params[2 * l], params[2 * l + 1]
        for i in range(J):
            for j in range(i + 1, J):
                qc.rzz(g * G[i, j] / 2, i, j)
        for pairs in ([(2 * i, 2 * i + 1) for i in range(4)], [(2 * i + 1, (2 * i + 2) % J) for i in range(4)]):
            for a, c in pairs:
                qc.rxx(b, a, c); qc.ryy(b, a, c)
    if measure: qc.measure_all()
    return qc

def probs(params, p):
    sv = Statevector(build(params, p, False)); pr = sv.probabilities()
    return pr   # index = bitstring integer (qubit 0 = LSB)

def tv(q): return 0.5 * np.abs(q - tgt).sum()
GN = np.abs(G).max()

def train(p, starts=12, seed=1):
    rng = np.random.default_rng(seed); best = None
    cost = np.array([sc[[combos.index(c) for c in [tuple(i for i in range(J) if b >> i & 1)]][0]] for b in w8])
    def en(x):
        pr = probs(x, p); return float(sum(pr[b] * c for b, c in zip(w8, cost)))
    for _ in range(starts):
        x0 = np.array([v for _ in range(p) for v in (rng.uniform(0, np.pi) / GN * 4, rng.uniform(0, np.pi / 2))])
        r = minimize(en, x0, method='Nelder-Mead', options=dict(maxfev=600))
        if best is None or r.fun < best.fun: best = r
    return best.x

if __name__ == '__main__':
    mode = sys.argv[1]
    angles = {0: []}
    for p in (1, 2, 3): angles[p] = train(p).tolist()
    out = dict(angles=angles, ideal_tv={p: float(tv(probs(angles[p], p))) for p in angles},
               ideal_acc_mass={p: float(sum(probs(angles[p], p)[b] for b in acc_bits)) for p in angles},
               accepted=sorted(acc_bits), n_alloc=70)
    if mode == 'prep':
        from qiskit_ibm_runtime.fake_provider import FakeFez
        be = FakeFez()
        out['transpiled'] = {}
        for p in angles:
            t = transpile(build(angles[p], p), be, optimization_level=3, seed_transpiler=7)
            ops = t.count_ops()
            out['transpiled'][p] = dict(depth=t.depth(), two_qubit=int(sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'rzz'))))
        json.dump(out, open('./prep.json', 'w'), indent=1); print(json.dumps(out, indent=1))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        svc = QiskitRuntimeService(); be = svc.backend(sys.argv[2]); shots = int(sys.argv[3])
        circs = [transpile(build(angles[p], p), be, optimization_level=3, seed_transpiler=7) for p in angles]
        job = Sampler(mode=be).run(circs, shots=shots); print('job id', job.job_id())
        res = job.result(); counts = {}
        for p, r in zip(angles, res): counts[p] = r.data.meas.get_counts()
        out['counts'] = counts; out['backend'] = sys.argv[2]; out['shots'] = shots
        json.dump(out, open('./hw_counts.json', 'w'), indent=1); print('saved')
