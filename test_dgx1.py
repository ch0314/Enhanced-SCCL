# test_dgx1.py
import sys
sys.path.append('.')

from .topology import Topology
from .collective import CollectiveType
from .pareto_synthesis import ParetoSynthesizer
from .visualization import visualize_pareto_frontier

def create_dgx1_topology() -> Topology:
    """
    Create DGX-1 topology as described in the SCCL paper.
    
    DGX-1 has 8 GPUs arranged in two groups:
    - Group 1: GPUs 0,1,2,3 (fully connected)
    - Group 2: GPUs 4,5,6,7 (fully connected)
    - Inter-group links: 0-4, 1-5, 2-6, 3-7
    
    Forms two non-overlapping rings:
    - Ring 1: 0-1-4-5-6-7-2-3 (2 NVLinks per edge)
    - Ring 2: 0-2-1-3-6-4-7-5 (1 NVLink per edge)
    """
    bandwidth = {}
    
    # Intra-group connections (within each group of 4)
    # Group 1: 0,1,2,3
    for i in range(4):
        for j in range(4):
            if i != j:
                # Check if this edge belongs to ring 1 (2 NVLinks)
                if (i, j) in [(0,1), (1,0), (2,3), (3,2)]:
                    bandwidth[(i, j)] = 2
                else:
                    bandwidth[(i, j)] = 1
    
    # Group 2: 4,5,6,7
    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                # Check if this edge belongs to ring 1 (2 NVLinks)
                if (i, j) in [(4,5), (5,6), (6,7), (7,4)]:
                    bandwidth[(i, j)] = 2
                else:
                    bandwidth[(i, j)] = 1
    
    # Inter-group connections
    inter_group_links = [
        (0, 4), (4, 0),  # These are part of ring 1 (2 NVLinks)
        (1, 5), (5, 1),  # These are part of ring 1 (2 NVLinks)
        (2, 6), (6, 2),  # These are part of ring 2 (1 NVLink)
        (3, 7), (7, 3)   # These are part of ring 2 (1 NVLink)
    ]
    
    for src, dst in inter_group_links:
        if (src, dst) in [(0,4), (1,5)]:
            bandwidth[(src, dst)] = 2
        else:
            bandwidth[(src, dst)] = 1
    
    return Topology(num_nodes=8, bandwidth=bandwidth)

def test_dgx1_allgather():
    """Test AllGather synthesis on DGX-1 topology"""
    print("="*60)
    print("Testing AllGather on DGX-1 Topology")
    print("="*60)
    
    # Create DGX-1 topology
    topology = create_dgx1_topology()
    
    # Print topology information
    print(f"\nTopology Information:")
    print(f"  Number of nodes: {topology.num_nodes}")
    print(f"  Number of links: {len(topology.bandwidth)}")
    print(f"  Diameter: {topology.compute_diameter()}")
    print(f"  Inverse bisection bandwidth: {topology.compute_inv_bisection_bandwidth():.3f}")
    
    # Count NVLinks
    total_nvlinks = sum(topology.bandwidth.values())
    print(f"  Total NVLinks: {total_nvlinks}")
    
    # Create synthesizer with k=7 to match paper results
    synthesizer = ParetoSynthesizer(k=0, max_steps_offset=6)
    
    # Synthesize algorithms
    algorithms = synthesizer.synthesize(CollectiveType.ALLGATHER, topology)
    
    # Print results
    print(f"\n{'='*60}")
    print("SYNTHESIS RESULTS")
    print(f"{'='*60}")
    print(f"Found {len(algorithms)} algorithms\n")
    
    for i, alg in enumerate(algorithms):
        print(f"Algorithm {i+1}:")
        print(f"  Steps (S): {alg['steps']}")
        print(f"  Rounds (R): {alg['rounds']}")
        print(f"  Chunks per node (C): {alg['chunks_per_node']}")
        print(f"  Latency cost: {alg['latency_cost']}")
        print(f"  Bandwidth cost (R/C): {alg['bandwidth_cost']:.3f}")
        print(f"  Cost formula: {alg['latency_cost']}·α + {alg['bandwidth_cost']:.3f}·L·β")
        
        # Check if this matches paper results
        if alg['steps'] == 2 and alg['rounds'] == 3:
            print("  *** This is the latency-optimal algorithm from the paper! ***")
        elif alg['steps'] == 3 and alg['rounds'] == 7 and alg['chunks_per_node'] == 6:
            print("  *** This is the bandwidth-optimal algorithm from the paper! ***")
        print()
    
    # Visualize Pareto frontier
    if algorithms:
        visualize_pareto_frontier(
            algorithms,
            title="DGX-1 AllGather Pareto Frontier",
            show_plot=True
        )


def analyze_dgx1_rings():
    """Analyze the ring structure of DGX-1"""
    print("\n" + "="*60)
    print("DGX-1 Ring Analysis")
    print("="*60)
    
    # Ring 1: 0->1->4->5->6->7->2->3->0 (2 NVLinks)
    ring1 = [(0,1), (1,4), (4,5), (5,6), (6,7), (7,2), (2,3), (3,0)]
    
    # Ring 2: 0->2->1->3->6->4->7->5->0 (1 NVLink)
    ring2 = [(0,2), (2,1), (1,3), (3,6), (6,4), (4,7), (7,5), (5,0)]
    
    print("Ring 1 (2 NVLinks per edge):")
    print(f"  Path: {' -> '.join(str(i) for i in [0,1,4,5,6,7,2,3,0])}")
    
    print("\nRing 2 (1 NVLink per edge):")
    print(f"  Path: {' -> '.join(str(i) for i in [0,2,1,3,6,4,7,5,0])}")
    
    # Verify rings are non-overlapping
    ring1_set = set(ring1)
    ring2_set = set(ring2)
    overlap = ring1_set.intersection(ring2_set)
    
    print(f"\nRings overlap: {len(overlap) > 0}")
    if overlap:
        print(f"  Overlapping edges: {overlap}")
    else:
        print("  ✓ Rings are non-overlapping (good for parallel communication)")

if __name__ == "__main__":
    # Run tests
    analyze_dgx1_rings()
    test_dgx1_allgather()