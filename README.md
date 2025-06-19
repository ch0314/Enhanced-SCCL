# SCCL (Synthesized Collective Communication Library) Implementation with Hybrid SMT Framework

This project implements an enhanced version of the SCCL paper "Synthesizing Optimal Collective Algorithms" with a novel hybrid synthesis framework that addresses critical scalability bottlenecks and enables latency-aware modeling for heterogeneous networks.

---

## Features

- **Automatic Algorithm Synthesis**: Generates algorithms on the Pareto frontier (latency-optimal vs. bandwidth-optimal) using an SMT solver.
- **Multiple Collective Support**: Supports `AllGather`, `AllToAll`, `AllReduce`, `Broadcast`, `Scatter`, `Gather`, `ReduceScatter`.
- **Flexible Topology Modeling**: Includes Ring, DGX-1, and custom topologies.
- **TACCL-based Routing**: Optimized path generation using Gurobi.
- **Verification & Visualization**: Verifies correctness and visualizes Pareto frontier.

---

## Requirements

- Python 3.8+
- Z3 Solver: `pip install z3-solver`
- Gurobi Optimizer (license required)
- NetworkX: `pip install networkx`
- Matplotlib: `pip install matplotlib`
- NumPy: `pip install numpy`

---

## Project Structure

```
sccl/
├── topology.py              # Topology definitions and bandwidth constraints
├── collective.py            # Collective operation definitions
├── sccl/
│   ├── sccl_basic.py       # Basic SCCL SMT encoding
│   ├── pareto_synthesis.py # Pareto-optimal synthesis algorithm
│   ├── candidate_generator.py # Candidate filtering for scalability
│   ├── taccl_routing.py    # TACCL-based routing with Gurobi
│   └── path_generator.py   # Path generation utilities
├── util/
│   ├── algorithm_verifier.py # Algorithm correctness verification
│   └── visualization.py     # Pareto frontier visualization
└── test/
    ├── test_ring.py        # Ring topology tests
    └── test_dgx1.py        # DGX-1 topology tests
```

---

## Quick Start

```python
from topology import create_ring_topology
from collective import CollectiveType
from sccl.pareto_synthesis import ParetoSynthesizer

# Create topology
topology = create_ring_topology(num_nodes=4)

# Create synthesizer
synthesizer = ParetoSynthesizer(k=5, max_steps_offset=7)

# Synthesize algorithms
algorithms = synthesizer.synthesize(CollectiveType.ALLGATHER, topology)

# Print results
for alg in algorithms:
    print(f"S={alg['steps']}, R={alg['rounds']}, C={alg['chunks_per_node']}")
    print(f"Cost: {alg['latency_cost']}·α + {alg['bandwidth_cost']:.3f}·L·β")
```

---

## Synthesizing AllGather on DGX-1 Topology
```python
from topology import create_dgx1_topology
from test.test_dgx1 import test_dgx1_allgather

# Run complete test with visualization
test_dgx1_allgather()
```
---

## Advanced Features

### Multi-stage Candidate Filtering

The implementation includes sophisticated filtering to improve scalability:

```
1. Bandwidth constraint (R/C ≥ b_l)
2. Collective-specific constraints
3. Topology-specific constraints
4. Duplicate ratio removal
5. Pareto dominance pruning
```

---

## Algorithm Verification

Comprehensive verification ensures correctness:

```python
from util.algorithm_verifier import verify_synthesized_algorithms

verification_results = verify_synthesized_algorithms(
    algorithms, topology, collective_type)
```

---

## Visualization

Visualize the Pareto frontier of synthesized algorithms:

```python
from util.visualization import visualize_pareto_frontier

visualize_pareto_frontier(
    algorithms,
    title="DGX-1 AllGather Pareto Frontier",
    show_plot=True)
```

---

## Limitations and Future Work

- Currently supports only **k-synchronous** algorithms.
- Combining collectives (AllReduce, ReduceScatter) is implemented via inversion.
- Synthesis time can be long for large-scale topologies.
- **Future work**: Integration with heterogeneous network latencies (α-β model).

---

## References

- Cai, Z., Liu, Z., et al. (2021). _"Synthesizing Optimal Collective Algorithms."_ PPoPP '21.
- Shah, A., et al. (2023). _"TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches."_ NSDI '23.

---
