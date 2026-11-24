# MSSI: Multi-Stage Framework with Sub-Problem Interleaving

Implementation of MSSI for the large-scale lexicographic
multi-objective vehicle routing problem with pickup, delivery and time windows
(LS-LMO-VRPPDTW), accompanying the manuscript submitted to Transportation
Research Part E.

MSSI first builds an initial solution with the Connectivity-Based Cyclic
Clustering Decomposition (C3D) and HGSADC, then interleaves neighbouring
sub-problems through the Rotation Scanning Optimization (RSO) in two stages,
minimizing the vehicle number (VN) and then the total distance (TD) under the
best-found VN. Finally, an Optimal Departure Time (ODT) post-optimization
minimizes the in-transit waiting time (WT) in closed form.

## Repository structure

```
src/mssi.py            C3D decomposition and the two RSO stages
src/odt.py             closed-form optimal departure time
scripts/run_mssi.py    run one instance and report VN, TD and WT
instances/600          VRPPDTW instances with 600 customers
instances/800          VRPPDTW instances with 800 customers
instances/1000         VRPPDTW instances with 1000 customers
requirements.txt       pinned Python dependencies
```

The instances in `instances/` are derived from the Gehring-Homberger VRPTW
benchmark by introducing simultaneous pickup and delivery.

## Environment

Python 3.11.9. A ready-to-use virtual environment is provided in `.venv`. To
recreate it:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The pinned versions are numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.0,
scikit-learn-extra 0.3.0, networkx 3.5, pyvrp 0.11.1 and matplotlib 3.10.3.

## Usage

```
.venv\Scripts\python.exe scripts\run_mssi.py --instance instances\600\C1_6_1.vrp --minutes 30 --seed 42
```

`--minutes` sets the wall-clock budget and `--seed` seeds both `random` and
`numpy`. The script prints a JSON object with the following fields:

- `VN`: number of vehicles of the best solution
- `TD`: total distance of the best solution
- `WT_default_departure`: total in-transit waiting time at departure time 0
- `WT_optimized`: total in-transit waiting time after the ODT post-optimization
- `runtime_seconds`, `routes`, `instance`, `seed`, `minutes`

## Random seeds

In the experiments reported in the manuscript, every independent run used a
seed derived deterministically from the algorithm, the instance and the run
index:

```python
seed = int(hashlib.sha256(f'MSSI|{instance}|{run}'.encode()).hexdigest()[:8], 16) % 2_000_000_000 + 1
```

with `run = 1, 2, ...`. Passing the resulting value to `--seed` reproduces the
corresponding run. If no seed is given, the script uses 1.
