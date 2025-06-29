# pareto_synthesis.py
from z3 import *
from typing import List, Tuple, Set, Dict, Optional
from .sccl_basic import *
from ..collective import *
from ..topology import *
from .candidate_generator import *
from ..util.visualization import *
from ..util.algorithm_verifier import *
from .path_generator import *
from .taccl_routing import *

class ParetoSynthesizer:
    """Implements the Pareto-optimal synthesis algorithm"""
    
    def __init__(self, 
                 k: int = 5,                    # k-synchronous parameter
                 max_steps_offset: int = 10,    # Max steps to search beyond diameter
                 max_algorithms: int = 10
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
        self.max_algorithms = max(k, max_algorithms)
        self.candidate_generator = CandidateGenerator(k)
    
    def synthesize(self, collective_type: CollectiveType, 
                  topology: Topology) -> List[Dict]:
        """
        Main Pareto synthesis algorithm 
        """        

        # Compute lower bounds
        a_l = topology.compute_diameter()
        b_l = topology.compute_inv_bisection_bandwidth()

        if(collective_type in [CollectiveType.ALLGATHER, CollectiveType.REDUCESCATTER]):
            # For ALLTOALL, we need to ensure each node can send/receive all chunks
            b_l *= topology.num_nodes - 1
        
        P = topology.num_nodes
        synthesized_algorithms = []
        self.candidate_generator.reset_pareto_tracking()
        total_stats = {
            'total_candidates': 0,
            'total_smt_calls': 0,
            'total_filtered': 0
        }
        
        print(f"Synthesis parameters:")
        print(f"  Collective: {collective_type.value}")
        print(f"  Topology: {P} nodes, {len(topology.bandwidth)} links")
        print(f"  k-synchronous: k={self.k}")
        print(f"  Latency lower bound (diameter): {a_l}")
        print(f"  Bandwidth lower bound (inv bisection): {b_l:.3f}")
        
        # Line 5: Iterate over steps starting from diameter
        for S in range(a_l, a_l + self.max_steps_offset):  # Limit search to prevent infinite loop
            print(f"\n--- Steps S = {S} ---")
            # Generate filtered candidates
            candidates, stats = self.candidate_generator.generate_candidates(
                collective_type, topology, S, b_l
            )

            # candidates, stats = self.candidate_generator.generate_candidates_simple(S, b_l)
            
            total_stats['total_candidates'] += stats.total_possible
            total_stats['total_filtered'] += (stats.total_possible - stats.final_count)

            # Line 7: Sort by ascending R/C (bandwidth cost)
            candidates.sort(key=lambda x: x[0] / x[1])
            found_solution_this_step = False
            
            # Try each candidate
            for R, C in candidates:               
                # Line 4, 8: Convert to global chunks
                collective = Collective.create_collective(
                    collective_type, topology.num_nodes, C
                )
                # paths = generate_optimized_ring_paths(topology, collective)
                print("Step 1: Generating paths with TACCL routing...")
                taccl_routing = TACCLRouter(topology, collective, S, R)
                paths = taccl_routing.generate_paths()
                print(f"Found {len(paths)} paths")


                verifier = AlgorithmVerifier(topology, collective)
                G = collective.num_chunks

                # Line 9: Try to synthesize
                print(f"Trying S={S}, R={R}, C={C} (R/C={R/C:.3f})...\n", end='')
                sccl = SCCLBasic(topology, collective, S, R, paths)
                solution = sccl.solve()
                total_stats['total_smt_calls'] += 1
                
                if solution and solution['status'] == 'sat':
                    # Line 10: Report synthesized algorithm
                    bandwidth_cost = R / C
                    algorithm = {
                        'steps': S,
                        'rounds': R,
                        'chunks_per_node': C,
                        'global_chunks': G,
                        'latency_cost': S,
                        'bandwidth_cost': bandwidth_cost,
                        'solution': solution
                    }   
                    
                    result = verifier.verify_algorithm(solution)
                    if result.errors:
                        print("Errors:")
                        for error in result.errors[:3]:  # Show first 3 errors
                            print(f"  • {error}")
                        if len(result.errors) > 3:
                            print(f"  ... and {len(result.errors) - 3} more errors")
                        continue  # Skip this candidate if verification fails

                    synthesized_algorithms.append(algorithm)
                    found_solution_this_step = True
                    print(f" SUCCESS!")
                    print(f"    ✓ Synthesized algorithm: S={S}, R={R}, C={C}, cost=({S}, {bandwidth_cost:.2f})")
                    print_solution_details(solution)
                    # Update Pareto tracking with new solution
                    self.candidate_generator.update_best_bandwidth_cost(S, bandwidth_cost)

                    # Line 11-12: Check if we reached bandwidth lower bound
                    if abs(R/C - b_l) < 1e-6:
                        return synthesized_algorithms
                    
                    # Line 13: Break to next S value
                    break
            
            # Optional: Stop if we've found enough algorithms
            if len(synthesized_algorithms) > self.max_algorithms:
                print("\nFound sufficient algorithms, stopping search")
                break
        
        return synthesized_algorithms
    
    def _print_summary(self, stats):
        """Print summary statistics"""
        print(f"\nSynthesis Summary:")
        print(f"  Total candidate space explored: {stats['total_candidates']}")
        print(f"  Candidates filtered out: {stats['total_filtered']} "
              f"({stats['total_filtered']/stats['total_candidates']*100:.1f}%)")
        print(f"  SMT solver calls: {stats['total_smt_calls']}")
        print(f"  Filtering efficiency: {stats['total_filtered']/stats['total_candidates']*100:.1f}% "
              f"reduction in SMT calls") 
    
# Example usage
if __name__ == "__main__":
    # Ring topology
    num_nodes=4
    ring_topology = create_ring_topology(num_nodes)
    
    synthesizer = ParetoSynthesizer(k=5, max_steps_offset=7)
    
    # Test with different collectives to see filtering effects
    for coll_type in [CollectiveType.ALLGATHER]:
        print(f"\n{'='*60}")
        print(f"Testing {coll_type.value.upper()}")
        print(f"{'='*60}")
        
        algorithms = synthesizer.synthesize(
            coll_type, ring_topology
        )

        verification_results = verify_synthesized_algorithms(
            algorithms, ring_topology, coll_type
        )

        print_verification_summary(algorithms, verification_results)

        visualize_pareto_frontier(algorithms)