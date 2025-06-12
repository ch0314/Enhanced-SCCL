# pareto_synthesis.py
from z3 import *
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from .sccl_basic import *
from .collective import *
from .topology import *

class ParetoSynthesizer:
    """Implements the Pareto-optimal synthesis algorithm"""
    
    def __init__(self, 
                 k: int = 5,                    # k-synchronous parameter
                 max_steps_offset: int = 10,    # Max steps to search beyond diameter
                 ):
        """
        Initialize synthesizer with search parameters
        
        Args:
            k: k-synchronous parameter (R ≤ S + k)
            max_steps_offset: Maximum steps to search beyond network diameter
            early_termination: Whether to stop when bandwidth optimal is found
        """
        self.k = k
        self.max_steps_offset = max_steps_offset
    
    def synthesize(self, collective_type: CollectiveType, 
                  topology: Topology) -> List[Dict]:
        """
        Main Pareto synthesis algorithm 
        """
        print("=== Starting Pareto Synthesis ===")
        # Line 2: Compute diameter
        a_l = topology.compute_diameter()  # Latency lower bound
        print(f"Diameter (a_l): {a_l}")
        
        # Line 3: Compute inverse bisection bandwidth
        b_l = topology.compute_inv_bisection_bandwidth()
        print(f"Inverse bisection bandwidth (b_l): {b_l:.4f}")
        
        synthesized_algorithms = []
        
        # Line 5: Iterate over steps starting from diameter
        for S in range(a_l, a_l + self.max_steps_offset):  # Limit search to prevent infinite loop
            print(f"\n--- Steps S = {S} ---")
            
            # Line 6: Generate candidate (R, C) pairs
            candidates = []

            for R in range(S, S + self.k + 1):
                for C in range(1, int(R / b_l) + 1):
                    candidates.append((R, C))
            
            # Line 7: Sort by ascending R/C (bandwidth cost)
            candidates.sort(key=lambda x: x[0] / x[1])
            
            print(f"Candidates: {len(candidates)} pairs")
            
            # Try each candidate
            for R, C in candidates:               
                # Line 4, 8: Convert to global chunks
                collective = Collective.create_collective(
                    collective_type, topology.num_nodes, C
                )
                G = collective.num_chunks

                # Line 9: Try to synthesize
                print(f"Trying S={S}, R={R}, C={C} (R/C={R/C:.3f})...", end='')
                sccl = SCCLBasic(topology, collective, S, R)
                solution = sccl.solve()
                
                if solution and solution['status'] == 'sat':
                    # Line 10: Report synthesized algorithm
                    algorithm = {
                        'steps': S,
                        'rounds': R,
                        'chunks_per_node': C,
                        'global_chunks': G,
                        'latency_cost': S,
                        'bandwidth_cost': R / C,
                        'solution': solution
                    }
                    
                    synthesized_algorithms.append(algorithm)
                    print(f"\n    ✓ Synthesized algorithm: S={S}, R={R}, C={C}, cost=({S}, {R/C:.2f})")
                    
                    # Line 11-12: Check if we reached bandwidth lower bound
                    if abs(R/C - b_l) < 1e-6:
                        return synthesized_algorithms
                    
                    # Line 13: Break to next S value
                    break
            
            # Optional: Stop if we've found enough algorithms
            if len(synthesized_algorithms) > 5:
                print("\nFound sufficient algorithms, stopping search")
                break
        
        return synthesized_algorithms
    
def visualize_pareto_frontier(algorithms: List[Dict]):
    """Visualize the Pareto frontier"""
    import matplotlib.pyplot as plt
    
    if not algorithms:
        print("No algorithms to visualize")
        return
    
    # Extract costs
    latencies = [alg['latency_cost'] for alg in algorithms]
    bandwidths = [alg['bandwidth_cost'] for alg in algorithms]
    
    plt.figure(figsize=(10, 6))
    
    # Plot all algorithms
    plt.scatter(latencies, bandwidths, s=100, alpha=0.7, c='blue', 
                edgecolors='black', linewidth=2)
    
    # Annotate points
    for i, alg in enumerate(algorithms):
        plt.annotate(f"({alg['steps']},{alg['rounds']},{alg['chunks_per_node']})",
                    (alg['latency_cost'], alg['bandwidth_cost']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Connect Pareto points
    sorted_algs = sorted(algorithms, key=lambda x: x['latency_cost'])
    pareto_lat = [alg['latency_cost'] for alg in sorted_algs]
    pareto_bw = [alg['bandwidth_cost'] for alg in sorted_algs]
    plt.plot(pareto_lat, pareto_bw, 'r--', alpha=0.5, linewidth=2)
    
    plt.xlabel('Latency Cost (Steps)', fontsize=12)
    plt.ylabel('Bandwidth Cost (R/C)', fontsize=12)
    plt.title('Pareto Frontier for Collective Algorithms', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend explaining annotation format
    plt.text(0.02, 0.98, 'Format: (steps, rounds, chunks/node)', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Test on 4-node ring
    print("=== Testing Pareto Synthesis on 4-node Ring ===")

    # Create synthesizer with k=3
    synthesizer = ParetoSynthesizer(k=3)
    
    # Run Pareto synthesis
    algorithms = synthesizer.synthesize(
        collective_type=CollectiveType.ALLGATHER,
        topology=create_ring_topology(4)
    )
    
    print(f"\n=== Found {len(algorithms)} algorithms ===")
    for alg in sorted(algorithms, key=lambda x: x['latency_cost']):
        print(f"Steps={alg['steps']}, Rounds={alg['rounds']}, "
              f"Chunks/node={alg['chunks_per_node']}, "
              f"Latency={alg['latency_cost']}, "
              f"Bandwidth={alg['bandwidth_cost']:.3f}")
    
    # Visualize
    if algorithms:
        visualize_pareto_frontier(algorithms)
    
    # Test on heterogeneous topology
    print("\n\n=== Testing on Heterogeneous Topology ===")
    
    # Create a heterogeneous topology (2 fast groups connected by slow link)
    hetero_bandwidth = {
        # Fast intra-group links
        (0, 1): 2, (1, 0): 2,  # Group 1
        (2, 3): 2, (3, 2): 2,  # Group 2
        # Slow inter-group links
        (1, 2): 1, (2, 1): 1
    }
    hetero_topology = Topology(num_nodes=4, bandwidth=hetero_bandwidth)
    
    # Run synthesis
    hetero_algorithms = synthesizer.synthesize(
        collective_type=CollectiveType.ALLGATHER,
        topology=hetero_topology
    )    
    print(f"\n=== Found {len(hetero_algorithms)} Pareto-optimal algorithms for heterogeneous topology ===")
    for alg in sorted(hetero_algorithms, key=lambda x: x['latency_cost']):
        print(f"Steps={alg['steps']}, Rounds={alg['rounds']}, "
              f"Chunks/node={alg['chunks_per_node']}, "
              f"Latency={alg['latency_cost']}, "
              f"Bandwidth={alg['bandwidth_cost']:.3f}")