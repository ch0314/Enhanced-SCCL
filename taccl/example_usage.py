# example_usage.py
"""
Example usage of TACCL for synthesizing collective algorithms.

This script demonstrates:
1. Basic synthesis without sketches
2. Synthesis with communication sketches
3. Pareto-optimal synthesis
4. Backend code generation
"""

import sys
sys.path.append('..')

from ..collective import CollectiveType
from ..topology import create_dgx1_topology
from ..taccl import (
    TACCLSynthesizer,
    CommunicationSketch,
    create_dgx2_topology,
    create_ndv2_topology,
    generate_implementation
)
from ..util.algorithm_verifier import AlgorithmVerifier

def example_basic_synthesis():
    """Example 1: Basic synthesis without sketches"""
    print("=" * 60)
    print("Example 1: Basic TACCL Synthesis")
    print("=" * 60)
    
    # Create topology
    topology = create_dgx1_topology()
    
    # Create synthesizer
    synthesizer = TACCLSynthesizer(topology, "DGX-1")
    
    # Synthesize ALLGATHER algorithm
    algorithm = synthesizer.synthesize(
        collective_type=CollectiveType.ALLGATHER,
        chunks_per_node=2,
        max_steps=8,
        time_limit=60,
        verbose=True
    )
    
    print(f"\nSynthesized algorithm:")
    print(f"  Steps: {algorithm.num_steps}")
    print(f"  Total rounds: {algorithm.total_rounds}")
    print(f"  Latency cost: {algorithm.latency_cost}")
    print(f"  Bandwidth cost: {algorithm.bandwidth_cost:.3f}")
    
    # Verify the algorithm
    collective = algorithm.collective_type
    verifier = AlgorithmVerifier(topology, collective)
    result = verifier.verify_algorithm({
        'sends': algorithm.sends,
        'rounds': algorithm.rounds,
        'schedule': algorithm.schedule
    })
    
    print(f"\nVerification: {'PASSED' if result.is_valid else 'FAILED'}")
    if not result.is_valid:
        for error in result.errors:
            print(f"  Error: {error}")

def example_sketch_synthesis():
    """Example 2: Synthesis with communication sketches"""
    print("\n" + "=" * 60)
    print("Example 2: Synthesis with Communication Sketches")
    print("=" * 60)
    
    # Create larger topology
    topology = create_dgx2_topology()
    synthesizer = TACCLSynthesizer(topology, "DGX-2")
    
    # Create ring sketch
    sketch = CommunicationSketch.create_ring_sketch(
        topology, 
        CollectiveType.ALLGATHER
    )
    
    print(f"Using sketch: {sketch.sketch_type.value}")
    
    # Synthesize with sketch
    algorithm = synthesizer.synthesize(
        collective_type=CollectiveType.ALLGATHER,
        chunks_per_node=1,
        sketch=sketch,
        max_steps=10,
        verbose=True
    )
    
    print(f"\nSynthesized algorithm with sketch:")
    print(f"  Performance: {algorithm.num_steps} steps, "
          f"{algorithm.bandwidth_cost:.3f} bandwidth cost")

def example_pareto_synthesis():
    """Example 3: Pareto-optimal synthesis"""
    print("\n" + "=" * 60)
    print("Example 3: Pareto-Optimal Synthesis")
    print("=" * 60)
    
    # Create topology
    topology = create_ndv2_topology()
    synthesizer = TACCLSynthesizer(topology, "NDv2")
    
    # Try different chunk sizes
    chunk_sizes = [1, 2, 4, 8]
    
    # Create different sketches
    sketches = [
        None,  # No sketch
        CommunicationSketch.create_ring_sketch(topology, CollectiveType.ALLTOALL),
        CommunicationSketch.create_hierarchical_sketch(
            topology, 
            CollectiveType.ALLTOALL,
            intra_node_groups=[[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15]]
        )
    ]
    
    # Synthesize Pareto-optimal algorithms
    algorithms = synthesizer.synthesize_pareto_optimal(
        collective_type=CollectiveType.ALLTOALL,
        chunks_per_node_range=chunk_sizes,
        sketches=sketches,
        max_steps=12,
        time_limit=30,
        verbose=False
    )
    
    print(f"\nFound {len(algorithms)} Pareto-optimal algorithms:")
    print("\nC\tSteps\tRounds\tLatency\tBandwidth\tSketch")
    print("-" * 60)
    
    for algo in algorithms:
        sketch_name = algo.sketch_type.value if algo.sketch_type else "none"
        print(f"{algo.chunks_per_node}\t{algo.num_steps}\t"
              f"{algo.total_rounds}\t{algo.latency_cost}\t"
              f"{algo.bandwidth_cost:.3f}\t\t{sketch_name}")

def example_backend_generation():
    """Example 4: Backend code generation"""
    print("\n" + "=" * 60)
    print("Example 4: Backend Code Generation")
    print("=" * 60)
    
    # Create simple topology and synthesize
    topology = create_dgx1_topology()
    synthesizer = TACCLSynthesizer(topology, "DGX-1")
    
    algorithm = synthesizer.synthesize(
        collective_type=CollectiveType.ALLREDUCE,
        chunks_per_node=1,
        max_steps=6,
        verbose=False
    )
    
    print("Synthesized ALLREDUCE algorithm")
    
    # Generate NCCL implementation
    print("\n--- NCCL Implementation ---")
    nccl_code = generate_implementation(algorithm, "nccl")
    print(nccl_code[:500] + "...\n")  # Print first 500 chars
    
    # Generate MPI implementation
    print("\n--- MPI Implementation ---")
    mpi_code = generate_implementation(algorithm, "mpi")
    print(mpi_code[:500] + "...\n")
    
    # Generate custom implementation
    print("\n--- Custom Implementation ---")
    custom_code = generate_implementation(algorithm, "custom")
    print(custom_code[:500] + "...")

def main():
    """Run all examples"""
    print("TACCL Example Usage")
    print("==================\n")
    
    try:
        # Basic synthesis
        example_basic_synthesis()
        
        # Sketch-based synthesis
        example_sketch_synthesis()
        
        # Pareto-optimal synthesis
        example_pareto_synthesis()
        
        # Backend generation
        example_backend_generation()
        
    except ImportError as e:
        print(f"\nError: Missing required dependency: {e}")
        print("Please install required packages:")
        print("  pip install gurobipy networkx")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()