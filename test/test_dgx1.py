# test_dgx1.py
import sys
sys.path.append('.')

from ..topology import *
from ..collective import CollectiveType
from ..sccl.pareto_synthesis import ParetoSynthesizer
from ..util.algorithm_verifier import *
from ..util.visualization import visualize_pareto_frontier
from ..sccl.path_generator import *
from ..sccl.taccl_routing import *
import time

def test_dgx1_allgather():
    """Test AllGather synthesis on DGX-1 topology"""
    print("="*60)
    print("Testing AllGather on DGX-1 Topology")
    print("="*60)
    
    # Create DGX-1 topology
    topology = create_dgx1_topology()
    coll_type = CollectiveType.ALLGATHER
    
    # Print topology information
    print(f"\nTopology Information:")
    print(f"  Number of nodes: {topology.num_nodes}")
    print(f"  Number of links: {len(topology.bandwidth)}")
    print(f"  Diameter: {topology.compute_diameter()}")
    print(f"  Inverse bisection bandwidth: {topology.compute_inv_bisection_bandwidth():.3f}")
    
    # Count NVLinks
    total_nvlinks = sum(topology.bandwidth.values())
    print(f"  Total NVLinks: {total_nvlinks}")
    start_time = time.time()       
    
    # S = 2
    # R = 2
    # C = 1
    # collective = Collective.create_collective(
    #     CollectiveType.ALLGATHER,
    #     num_nodes=topology.num_nodes,
    #     chunks_per_node=C
    # )

    # taccl_routing = TACCLRouter(topology, collective, S, R)
    # paths = taccl_routing.generate_paths()
    # sccl = SCCLBasic(topology, collective, S, R, paths)
    # solution = sccl.solve()
    
    
    # # Create synthesizer with k=7 to match paper results
    synthesizer = ParetoSynthesizer(k=0, max_steps_offset=6)
    
    # # Synthesize algorithms
    algorithms = synthesizer.synthesize(coll_type, topology)
    
    # Print results
    # print(f"\n{'='*60}")
    # print("SYNTHESIS RESULTS")
    # print(f"{'='*60}")
    # print(f"Found {len(algorithms)} algorithms\n")

    end_time = time.time()
    print(f"\nRouting paths generated in {end_time - start_time:.2f} seconds")
    
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

    verification_results = verify_synthesized_algorithms(
        algorithms, topology, coll_type
    )
    print_verification_summary(algorithms, verification_results)

    # Visualize Pareto frontier
    if algorithms:
        visualize_pareto_frontier(
            algorithms,
            title="DGX-1 AllGather Pareto Frontier",
            show_plot=True
        )

if __name__ == "__main__":
    # Run tests
    test_dgx1_allgather()