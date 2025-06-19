import sys
sys.path.append('.')

from ..topology import *
from ..collective import CollectiveType
from ..sccl.pareto_synthesis import ParetoSynthesizer
from ..util.algorithm_verifier import *
from ..util.visualization import visualize_pareto_frontier
from ..sccl.path_generator import *
from ..sccl.taccl_routing import *
from ..sccl.sccl_basic import SCCLBasic
import time

def test_ring_allgather():
    topology = create_ring_topology(4)  # Create a ring topology with 8 nodes
    coll_type = CollectiveType.ALLGATHER

    S = 2
    R = 3
    C = 1
    collective = Collective.create_collective(
        CollectiveType.ALLGATHER,
        num_nodes=topology.num_nodes,
        chunks_per_node=C
    )

    taccl_routing = TACCLRouter(topology, collective, S, R)
    paths = taccl_routing.generate_paths()
    print(paths)

    sccl = SCCLBasic(topology, collective, S, R, paths)
    solution = sccl.solve()

    print(solution)

    synthesizer = ParetoSynthesizer(k=0, max_steps_offset=6)
    
    # # Synthesize algorithms
    algorithms = synthesizer.synthesize(coll_type, topology)

    verification_results = verify_synthesized_algorithms(
        algorithms, topology, coll_type
    )
    print_verification_summary(algorithms, verification_results)

if __name__ == "__main__":
    # Run tests
    test_ring_allgather()