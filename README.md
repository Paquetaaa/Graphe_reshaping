# Graph Reshaping and Multi-Restart Priority-Based Search for Multi-Agent Drone Routing

This repository contains the code and the experimental artefacts of the centralised half of my master's-thesis internship at the **Intelligent Computing Laboratory** (Pr. Donghui Lin), **Okayama University**, March-August 2026.

The internship addresses the **Drone Routing Problem (DRP)** on the [DRP-Challenge](https://drp-challenge.com/#/overview) benchmark, with two algorithmic contributions:

1. **Graph Reshape**, a solver-agnostic preprocessing that transforms an arbitrary weighted urban graph into a unit-time directed graph on which any grid-designed MAPF solver (CBS, ICBS, ECBS, PBS, ...) applies unchanged.
2. **Multi-Restart Priority-Based Search**, a meta-heuristic wrapper around PBS that compensates for PBS's sensitivity to its priority ordering by running several attempts from a diverse pool of orderings.

The combined approach was submitted to the **DRP-Challenge held at the AAMAS'26 opening session** (Paphos, 25-29 May 2026) and finished **3rd out of 42 participants**. The results were packaged into a research paper submitted to **PAAMS'26** (Napoli, October 2026).

> The second half of the internship (decentralised reinforcement learning) lives in a separate repository, [MARL4DRP](https://github.com/Paquetaaa/MARL4DRP).

---

## Table of Contents

- [Context](#context)
- [Graph Reshape](#graph-reshape)
- [Multi-Restart PBS](#multi-restart-pbs)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Context

The DRP-Challenge asks for collision-free trajectories that route a fleet of drones from start to goal nodes on three benchmark maps:

| Map | Nodes | Edges | Drones |
|---|---|---|---|
| `map_3x3` (grid) | 9 | 12 | 2-4 |
| `map_aoba01` (urban, Sendai) | 18 | 22 | 4-8 |
| `map_shibuya` (urban, Tokyo) | 28 | 39 | 8-12 |

The official cost function (lower is better) is
```
cost_p = (1/10) * sum over 10 episodes of sum over drones of cost_{i,j}
cost_{i,j} = step count if drone reached its goal, else 100 (collision or timeout).
```

The two urban maps break the standard grid-MAPF assumptions: edges have non-unit travel time, two drones on close but distinct edges can collide mid-edge, and edges traversed in opposite directions cannot be flagged with a single time-stamped vertex constraint. This is the gap the two contributions below address.

---

## Graph Reshape

A preprocessing step that converts a weighted undirected graph `G = (V, E, w)` into a unit-time directed graph `G' = (V', E')` suitable for any grid-MAPF solver.

For each edge `(u, v)` of weight `w(u, v)` and an agent speed `sigma`, the reshape splits the edge into
```
k(u, v) = ceil( w(u, v) / sigma )
```
unit-weight segments by inserting `k - 1` intermediate nodes. Crucially, it creates **two separate directed chains** for each physical edge (one for `u -> v`, one for `v -> u`), so that two drones can traverse the same physical edge in opposite directions simultaneously without head-on conflict, which matches the regulatory convention that opposite-direction urban flights are deconflicted by altitude.

The output graph `G'` has four properties that matter to grid solvers:

1. **Time-position alignment**: each step in `G'` takes exactly one simulator tick.
2. **Mid-edge conflict resolution**: vertex and proximity conflicts can be detected at intermediate positions.
3. **Implicit edge-swap detection**: opposite-direction intermediates on the same physical edge are pairwise within `sigma` of each other, so edge swaps are flagged as proximity conflicts on `G'` without any explicit edge-conflict reasoning.
4. **Solver agnosticism**: the reshape is purely preprocessing, so CBS, ICBS, ECBS, PBS and any future grid-based MAPF solver apply unmodified.

Pseudocode is in the paper; the Python implementation is `generate_intermediate_nodes.py`. Graph sizes before/after reshape on the three benchmark maps:

| Map | `|V|` | `|E|` | `|V'|` | `|E'|` | Expansion |
|---|---|---|---|---|---|
| `3x3`     | 9  | 12 | 91   | 106  | 10.1x |
| `aoba01`  | 18 | 22 | 548  | 574  | 30.4x |
| `shibuya` | 28 | 39 | 1008 | 1058 | 36.0x |

---

## Multi-Restart PBS

Priority-Based Search (PBS, Ma et al., 2019) is a suboptimal MAPF solver that fixes a priority ordering of the agents and plans them sequentially with A* under constraints from higher-priority agents. PBS is dramatically faster than CBS-family solvers on large instances, but the quality of its solution (and sometimes its existence) depends heavily on the priority ordering.

To compensate, **Multi-Restart PBS** runs up to `K` independent PBS attempts. The first 8 attempts use deterministic priority heuristics, the remaining `K - 8` attempts use random shuffles of the agent set. The best valid plan found is returned.

The 8 deterministic orderings:

| Heuristic | Idea |
|---|---|
| `ShortestFirst` | Shortest A* path first |
| `LongestFirst` | Longest A* path first |
| `Centrality` | Highest graph-centrality start first |
| `PathOverlap` | Most overlap with other agents' A* paths first |
| `GoalCriticality` | Highest-degree goal node first |
| `StartCriticality` | Highest-degree start node first |
| `Alternating` | Interleaves descending-distance head-to-tail |
| `ReverseID` | Reverse agent IDs (cheap deterministic seed) |

Empirically, no single deterministic heuristic dominates: the best one wins on only **46.7 % of instances**. The diversity of the pool is what makes Multi-Restart PBS robust across maps and start/goal configurations. The Python implementation is `policy/policy_PBS.py`.

---

## Results

Cost per map at each pipeline stage (lower is better). `N/A` means the solver did not converge within the compute budget; the row total uses a single-pass PBS fallback on timeout.

| Stage | Map 1 (3x3) | Map 2 (Aoba01) | Map 3 (Shibuya) | Total |
|---|---|---|---|---|
| Reactive baseline (A* + wait-on-conflict)          | 401 | 4112   | 7340   | 11853 |
| CBS with single-pass PBS fallback                  | 347 | N/A    | N/A    | 11532 |
| + ICBS (cardinal prioritisation)                   | 268 | N/A    | N/A    | 10377.5 |
| + Disjoint Splitting (vertex only)                 | 268 | N/A    | N/A    | 10303.5 |
| + Warm start + UB pruning                          | 268 | N/A    | N/A    | 10207.8 |
| ECBS (focal, w = 2)                                | 274 | N/A    | N/A    | 12716 |
| PBS, 100 attempts                                  | 264 | 2635.5 | 5744   | 8643.5 |
| PBS, 200 attempts                                  | 264 | 2634   | 5704   | 8602 |
| PBS, 500 attempts                                  | 264 | 2631   | 5544.5 | 8439.5 |
| **Multi-Restart PBS with caching (final)**         | **264** | **2631** | **5532.2** | **8427.2** |

**Final ranking**: 3rd / 42 at the DRP-Challenge held at AAMAS'26 (Paphos, 25-29 May 2026).

Detailed conflict-class statistics and the heuristic-diversity ablation are in the PAAMS'26 paper.

---

## Installation

The environment was developed and tested with `python==3.11.4`.

```bash
conda create -n drpdev python=3.11.4
conda activate drpdev

git clone https://github.com/Paquetaaa/Graphe_reshaping.git
cd Graphe_reshaping
pip install -e .
pip install -r requirements.txt
```

A scripted setup is available in `setup_conda.sh`.

The DRP-Challenge GUI is available with:

```bash
python policy_tester.py
```

---

## Usage

### Run a policy on the full benchmark

```bash
python calculate_cost.py
```

This evaluates `policy/policy_PBS.py` over the 30 problems defined in `problem/problems.py` and writes a JSON file in `Results/` with the final cost.

### Reproduce the paper numbers

The Multi-Restart PBS submission that achieved 8427.2 is saved as `Results/GRENECHE Lucas_8427.2.json`. The corresponding policy is `policy/policy_PBS.py` with `K = 2000` attempts.

### Reproduce the ablations

- A* expansion counts across maps: `python astar_expansions_comparison.py`
- CBS conflict-class statistics: `python cbs_stats_evaluation.py`
- Cost plots from the JSON results: `python result_plot.py`

---

## Repository layout

```
.
├── README.md                          this file
├── LICENSE
├── setup.py / requirements.txt        package metadata + Python deps
├── setup_conda.sh                     scripted conda env
├── drp_env/                           DRP-Challenge environment (Gym-compatible)
├── problem/problems.py                30 benchmark instances (read-only)
├── policy/
│   ├── policy_PBS.py                  Multi-Restart PBS (used for the 8427.2 submission)
│   ├── policy_PBS_early.py            earlier PBS variant kept for reproducibility
│   └── Old_Policy/                    historical baselines (reactive A*, CBS, ICBS, ECBS)
├── generate_intermediate_nodes.py     Graph Reshape preprocessing
├── policy_tester.py                   visual sandbox for a single policy
├── calculate_cost.py                  full-benchmark evaluator
├── astar_expansions_comparison.py     low-level A* expansion analysis
├── cbs_stats_evaluation.py            CBS conflict-class statistics
├── result_plot.py                     post-processing plots
├── example/                           reinforcement-learning example provided by the organisers
├── assets/                            images and markdown referenced by this README
└── Results/                           submitted JSON files (per-attempt and per-variant)
```

---

## Citation

If you use this code, please cite the PAAMS'26 paper:

<!-- ```bibtex
@unpublished{greneche2026graph,
  title  = {Graph Reshaping and Multi-Restart Priority-Based Search for Multi-Agent Drone Routing on Urban Graphs},
  author = {Gren{\`e}che, Lucas and Lin, Donghui},
  year   = {2026},
  note   = {Submitted to PAAMS'26, Milan, Italy}
}
``` -->
```bibtex
To come
```


The PBS algorithm itself is from Ma et al. 2019:

```bibtex
@inproceedings{ma2019pbs,
  title     = {Searching with Consistent Prioritization for Multi-Agent Path Finding},
  author    = {Ma, Hang and Harabor, Daniel and Stuckey, Peter J. and Li, Jiaoyang and Koenig, Sven},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2019}
}
```

---

## License

This repository is released under the MIT License (see `LICENSE`).

---

## Acknowledgements

I thank Professor Donghui Lin for the supervision throughout this internship, the Lin Lab members for their support, and the organisers of the DRP-Challenge for the benchmark and the simulator.

This work was carried out as part of the Okayama University International Research Internship Program, with mobility support from the Auvergne-Rhône-Alpes Region.
