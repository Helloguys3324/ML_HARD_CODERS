# Runbook 07 — Frontier: GNN / PINN / QML / QUBO (Antarctica)

> Template: `templates/07_frontier_gnn_pinn.py` · Track key: `frontier`
> CPU verdict: **viable for small graphs, 1D PINNs, QUBO and ≤10-qubit QML;
> not viable for large 3D meshes, deep GNNs, or high-qubit statevector simulation.**
> Modes: `gnn`, `pinn`, `qml`, `qubo`; `all` (the default) is the demo suite.
> `torch_geometric` (2.x) and `pennylane` **are installed** after the upstream
> dependency sync, but the template deliberately implements its own PyTorch GNN
> and statevector QML, so it does not need them. Do not follow PyG/PennyLane
> snippets blindly.

---

## 1. Triage (first 5 min)

| Archetype | Signal | Fastest baseline | CPU upgrade |
| :--- | :--- | :--- | :--- |
| Graph classification/regression | `edges.csv`, graph labels | topological features + LightGBM | shallow GIN/SAGE + OOF embeddings + LGB |
| Node/link prediction | `nodes.csv` with labels/masks | 1–2 hop neighbor aggregates + LGB | GNN + residual/label propagation |
| PDE / physics | grid/points, equation, BCs | `--mode pinn` with tanh MLP | RFF + hard BC + causal + L-BFGS |
| QUBO / Ising / routing | Q matrix, assignment | exact MILP (n ≤ 18–30) | parallel tempering / tabu |
| Quantum ML | small feature CSV | projected quantum kernel + SVC | data re-uploading VQC (≤10 qubits) |

```bash
export PY=/home/kozak/mllm/py REPO=/home/kozak/mllm/MML
cd "$REPO"
$PY lmml_master.py new lvl_11 --track frontier
$PY lmml_master.py solve lvl_11 -- --demo        # 3–4 min smoke test; optional
```

The demo runs all four modes and takes ~225 s on this box. For a real task, run
only the relevant mode. **`--mode all` is the default and runs the demo suite**;
`--mode gnn`, `--mode qml` and `--mode qubo` without their data arguments fall
back to synthetic benchmarks, so always pass `--nodes/--edges`, `--train/--test`
or `--qubo-matrix` for a real submission.

---

## 2. Commands per mode

```bash
# GNN on provided node/edge CSVs
$PY lmml_master.py solve lvl_11 -- \
  --mode gnn --nodes data/nodes.csv --edges data/edges.csv \
  --gnn-task node --gnn-arch gin --gnn-hidden 64 --gnn-layers 3 \
  --gnn-epochs 25 --gnn-folds 3 --device cpu

# PINN (1D Poisson / heat / Allen-Cahn)
$PY lmml_master.py solve lvl_11 -- \
  --mode pinn --pinn-pde poisson1d --dtype float64 \
  --adam-steps 200 --lbfgs-steps 25 --collocation 256 --device cpu

# QUBO: exact MILP for n<=18 inside the template, parallel tempering otherwise
$PY lmml_master.py solve lvl_11 -- \
  --mode qubo --qubo-matrix data/Q.npy --qubo-vars 32 --device cpu

# QML on a small CSV (projected quantum kernel + VQC)
$PY lmml_master.py solve lvl_11 -- \
  --mode qml --train data/train.csv --test data/test.csv \
  --qml-qubits 6 --qml-layers 3 --device cpu

# Validate the generated file
$PY lmml_master.py check lvl_11 --sample lvl_11/data/sample_submission.csv
```

Always check the output schema against `sample_submission.csv`; the template
writes task-specific columns (`x,u_pred`, `id,target`, `var,x`, etc.).

---

## 3. Best CPU approach per archetype

### Graphs
1. **Topological/structural features + LightGBM** is the baseline and often the
   best CPU model: degree stats, clustering, triangles, k-core, components,
   assortativity, spectral gap, label/feature aggregates over 1–2 hops.
2. **Shallow GNN** (3 layers, hidden 64–128, residual + JK + virtual node) on
   ≤2k graphs; 3 folds, 25–40 epochs. Template `ProFrontierGNN`.
3. **OOF GNN embeddings + LGB** usually beats the GNN head alone; concatenate
   mean/max/sum readouts with topological features.
4. **Label propagation / C&S** for node classification when edges are informative.

### PDE / PINN
- MLP 3–5 layers × 64–128, `tanh` or `sin`, multi-scale random Fourier features.
- Hard-constraint ansatz for BC/IC; causal time weighting; Sobol collocation.
- Adam 200–1000 steps, then L-BFGS 20–50. `float64` for stiff/high-order PDEs.
- Evaluate on a grid: report RMSE against the exact/reference solution and max
  boundary error. A low interior residual with a violated BC is wrong.

### QUBO / combinatorial
- Exact MILP for ≤18 vars (template) or ≤30 with a 5–10 s `scipy.optimize.milp`
  time limit. Use the exact energy as a certificate.
- Parallel tempering (template, vectorized NumPy `O(N)` ΔE) for 32–256 vars;
  multiple seeds, keep the best verified energy.
- For routing/scheduling, try `scipy.optimize.milp`/DP before any heuristic.

### QML
- Projected quantum kernel + `SVC(kernel="precomputed")` is the strong baseline
  and immune to barren plateaus.
- Data re-uploading VQC only for ≤10 qubits, ≤1k samples; local cost functions,
  near-identity init.
- Statevector simulation memory is `2^qubits` complex128: 20 qubits = 16 MB per
  state, but intermediate tensors and batches explode. Stay ≤12 qubits on CPU.

---

## 4. Feature engineering that pays

- Degree: mean/std/max/min, degree histogram, log-degree.
- Clustering coefficient, triangle count, k-core, assortativity, transitivity.
- Connected components: count, sizes, giant-component fraction.
- Spectral: Laplacian eigenvalues (λ2 = spectral gap), eigenvector centrality.
- RWSE: diagonal of `(D^-1 A)^k` for k=1..6 (closed-walk counts).
- LapPE: top-k non-trivial Laplacian eigenvectors, sign-flipped during training.
- Neighbor aggregates of node features/labels: mean/std/min/max/sum over 1–2 hops.
- Edge features: weight stats, common neighbors, Adamic-Adar, Jaccard.
- Graph size, edge density, diameter (approx), number of triangles.

---

## 5. Validation

- **Graph-level**: `StratifiedKFold` on graphs; if graphs come from families/sites,
  group by family. Never split nodes of the same graph across folds.
- **Node-level**: mask train/val/test nodes exactly as the task specifies; do not
  use test edges in structural features if the task hides them.
- **Link-level**: split edges, not nodes; ensure no reverse-edge leakage.
- **PINN**: report PDE residual on collocation points, BC/IC error, and RMSE on a
  reference grid. Validate with a different collocation seed.
- **QUBO**: compare heuristic energy to exact MILP for small n; for large n, run
  3–5 seeds and report best/median. Verify `x^T Q x` from the saved solution.
- **QML**: nested CV; kernel matrix must be computed per fold to avoid leakage.

---

## 6. Common pitfalls

| Pitfall | Symptom | Fix |
| :--- | :--- | :--- |
| Deep vanilla GNN (>4 layers) | All node embeddings collapse | Residual + JK + virtual node; 2–3 layers |
| Test edges used in features | CV 0.99, LB random | Mask test edges; build graph per split |
| ReLU PINN | Second derivative ≡ 0, PDE loss stuck | `tanh`/`sin`/`gelu`; never ReLU for PINN |
| float32 PINN on stiff PDE | Residual stalls | `--dtype float64` |
| QUBO penalty mis-tuned | Infeasible solution | Verify constraints; exact MILP for small n |
| `torch_geometric` import | ModuleNotFoundError (only if absent; it is installed now) | Use the template's own GNN regardless |
| `pennylane` import | ModuleNotFoundError (only if absent; it is installed now) | Use template QML (PyTorch statevector) |
| >16 qubits | RAM explosion | ≤10–12 qubits; projected kernel |
| Full-batch GNN on 10k graphs | OOM/slow | Mini-batch graphs; hidden 64 |

---

## 7. Snippets

### 7.1 Topological features with NetworkX (CPU baseline)

```python
import networkx as nx, numpy as np, polars as pl

def graph_feats(G):
    deg = np.array([d for _, d in G.degree()])
    clus = np.array(list(nx.clustering(G).values()))
    try: eig = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray()))[:3]
    except Exception: eig = np.zeros(3)
    return np.array([
        G.number_of_nodes(), G.number_of_edges(), G.number_of_edges() / max(G.number_of_nodes(), 1),
        deg.mean(), deg.std(), deg.max(), np.median(deg),
        clus.mean(), clus.std(),
        nx.number_connected_components(G),
        max((len(c) for c in nx.connected_components(G)), default=0) / max(G.number_of_nodes(), 1),
        nx.average_clustering(G), nx.transitivity(G),
        eig[1] if len(eig) > 1 else 0.0,
        sum(nx.triangles(G).values()) / 3,
    ], np.float32)
```

### 7.2 Exact QUBO check with SciPy (n ≤ 30)

```python
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def qubo_milp(Q, time_limit=5.0):
    Q = 0.5 * (Q + Q.T); n = len(Q); pairs = [(i, j) for i in range(n) for j in range(i+1, n) if abs(Q[i, j]) > 1e-12]
    m = len(pairs); c = np.zeros(n + m); c[:n] = np.diag(Q)
    rows, ub = [], []
    for k, (i, j) in enumerate(pairs):
        c[n+k] = 2 * Q[i, j]
        r = np.zeros(n+m); r[n+k] = 1; r[i] = -1; rows.append(r); ub.append(0)
        r = np.zeros(n+m); r[n+k] = 1; r[j] = -1; rows.append(r); ub.append(0)
        r = np.zeros(n+m); r[i] = 1; r[j] = 1; r[n+k] = -1; rows.append(r); ub.append(1)
    res = milp(c, integrality=np.ones(n+m), bounds=Bounds(0, 1),
               constraints=LinearConstraint(np.vstack(rows), -np.inf, np.array(ub)),
               options={"time_limit": time_limit})
    return res.x[:n], res.fun
```

### 7.3 Rust graph kernels

```python
from optimizations.rust_fast_ext import fast_dijkstra_sssp, fast_bfs_components
# Build CSR once (see template build_csr_from_edges), then:
n_comp, comp = fast_bfs_components(n_nodes, row_ptr, col_idx)
dist = fast_dijkstra_sssp(n_nodes, row_ptr, col_idx, weights, src=0)
```

---

## 8. Runtime on 4 cores (shared box)

| Operation | Time |
| :--- | :--- |
| Topological features, 1k small graphs | 1–3 min |
| LGB on graph features | seconds |
| GNN 60 graphs × 25 epochs × 3 folds | ~1–3 min |
| GNN 1k graphs × 40 epochs × 3 folds | 20–50 min |
| PINN Poisson, 200 Adam + 25 LBFGS, 256 collocation | 1–3 min |
| PINN heat/Allen-Cahn, float64, 1k steps | 5–15 min |
| QUBO 32 vars parallel tempering, 40 epochs × 8 replicas | 10–40 s |
| QUBO exact MILP, 18 vars | <1 s |
| QML 6 qubits, 500 samples, kernel + SVC | 1–5 min |
| Full template demo (all modes) | ~225 s measured |

## 9. Do not attempt without GPU

- `torch_geometric`/PyG large-batch training, >4-layer GNNs on >10k graphs.
- 3D/molecular/heterogeneous GNNs, MeshGraphNets, neural operators.
- PINNs with >50k collocation points, 2D/3D PDEs, float64 long sweeps.
- Statevector QML >16 qubits, quantum hardware backends, large VQC sweeps.
- QUBO branch-and-bound on >200 vars; use parallel tempering/tabu instead.
