# taccl_synthesizer.py
"""
Main TACCL Synthesizer

This is the main entry point for TACCL synthesis. It coordinates the 
communication sketch, MILP solver, and routing heuristics to generate
efficient collective algorithms.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ..topology import Topology
from ..collective import Collective, CollectiveType
from .communication_sketch import CommunicationSketch, SketchType
from .milp_synthesizer import MILPSynthesizer, MILPSolution
from .routing_heuristics import select_routing_heuristic, RoutingHeuristic

@dataclass
class TACCLAlgorithm:
    """A synthesized TACCL algorithm"""
    collective_type: CollectiveType
    topology_name: str
    num_nodes: int
    num_chunks: int
    chunks_per_node: int
    
    # Solution details
    sends: List[Dict]  # List of sends
    rounds: List[int]  # Rounds per step
    schedule: Dict[Tuple[int, int], int]  # (chunk, node) -> time
    
    # Performance metrics
    num_steps: int
    total_rounds: int
    bandwidth_cost: float
    latency_cost: int
    synthesis_time: float
    
    # Sketch used (if any)
    sketch_type: Optional[SketchType] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'collective_type': self.collective_type.value,
            'topology_name': self.topology_name,
            'num_nodes': self.num_nodes,
            'num_chunks': self.num_chunks,
            'chunks_per_node': self.chunks_per_node,
            'sends': self.sends,
            'rounds': self.rounds,
            'schedule': {f"{k[0]},{k[1]}": v for k, v in self.schedule.items()},
            'num_steps': self.num_steps,
            'total_rounds': self.total_rounds,
            'bandwidth_cost': self.bandwidth_cost,
            'latency_cost': self.latency_cost,
            'synthesis_time': self.synthesis_time,
            'sketch_type': self.sketch_type.value if self.sketch_type else None
        }

class TACCLSynthesizer:
    """
    Main TACCL synthesizer that coordinates all components.
    
    This class provides the high-level interface for TACCL synthesis,
    handling communication sketches, MILP solving, and algorithm generation.
    """
    
    def __init__(self, topology: Topology, topology_name: str = "custom"):
        self.topology = topology
        self.topology_name = topology_name
        
    def synthesize(self, 
                  collective_type: CollectiveType,
                  chunks_per_node: int,
                  sketch: Optional[CommunicationSketch] = None,
                  max_steps: int = 10,
                  time_limit: int = 300,
                  verbose: bool = False) -> TACCLAlgorithm:
        """
        Synthesize a collective algorithm using TACCL.
        
        Args:
            collective_type: Type of collective (ALLGATHER, ALLTOALL, etc.)
            chunks_per_node: Number of chunks per node
            sketch: Optional communication sketch to guide synthesis
            max_steps: Maximum number of steps to consider
            time_limit: Time limit for synthesis in seconds
            verbose: Whether to print progress
            
        Returns:
            TACCLAlgorithm with the synthesized algorithm
        """
        start_time = time.time()
        
        # Create collective
        collective = Collective.create_collective(
            collective_type, 
            self.topology.num_nodes,
            chunks_per_node
        )
        
        if verbose:
            print(f"\nSynthesizing {collective_type.value} algorithm")
            print(f"Topology: {self.topology_name} ({self.topology.num_nodes} nodes)")
            print(f"Chunks: {collective.num_chunks} total, {chunks_per_node} per node")
            if sketch:
                print(f"Using sketch: {sketch.sketch_type.value}")
        
        # Create MILP synthesizer
        synthesizer = MILPSynthesizer(
            self.topology,
            collective,
            sketch=sketch,
            verbose=verbose
        )
        
        # Run synthesis
        solution = synthesizer.synthesize(max_steps, time_limit)
        
        # Compute metrics
        num_steps = len(solution.rounds)
        total_rounds = sum(solution.rounds)
        bandwidth_cost = total_rounds / chunks_per_node
        latency_cost = num_steps
        
        synthesis_time = time.time() - start_time
        
        if verbose:
            print(f"\nSynthesis completed in {synthesis_time:.2f}s")
            print(f"Algorithm: {num_steps} steps, {total_rounds} rounds")
            print(f"Costs: Latency={latency_cost}, Bandwidth={bandwidth_cost:.3f}")
        
        return TACCLAlgorithm(
            collective_type=collective_type,
            topology_name=self.topology_name,
            num_nodes=self.topology.num_nodes,
            num_chunks=collective.num_chunks,
            chunks_per_node=chunks_per_node,
            sends=solution.sends,
            rounds=solution.rounds,
            schedule=solution.schedule,
            num_steps=num_steps,
            total_rounds=total_rounds,
            bandwidth_cost=bandwidth_cost,
            latency_cost=latency_cost,
            synthesis_time=synthesis_time,
            sketch_type=sketch.sketch_type if sketch else None
        )
    
    def synthesize_with_auto_sketch(self,
                                   collective_type: CollectiveType,
                                   chunks_per_node: int,
                                   **kwargs) -> TACCLAlgorithm:
        """
        Synthesize with automatically selected communication sketch.
        
        This method analyzes the topology and collective type to automatically
        select an appropriate communication sketch.
        """
        # Auto-select sketch based on topology analysis
        sketch = self._auto_select_sketch(collective_type)
        
        return self.synthesize(
            collective_type=collective_type,
            chunks_per_node=chunks_per_node,
            sketch=sketch,
            **kwargs
        )
    
    def synthesize_pareto_optimal(self,
                                 collective_type: CollectiveType,
                                 chunks_per_node_range: List[int],
                                 sketches: Optional[List[CommunicationSketch]] = None,
                                 **kwargs) -> List[TACCLAlgorithm]:
        """
        Synthesize Pareto-optimal algorithms for different chunk sizes.
        
        Args:
            collective_type: Type of collective
            chunks_per_node_range: List of chunk sizes to try
            sketches: Optional list of sketches to try
            
        Returns:
            List of Pareto-optimal algorithms
        """
        algorithms = []
        
        # If no sketches provided, generate some defaults
        if sketches is None:
            sketches = [None]  # At least try without sketch
            
            # Add ring sketch if applicable
            try:
                ring_sketch = CommunicationSketch.create_ring_sketch(
                    self.topology, collective_type
                )
                sketches.append(ring_sketch)
            except:
                pass
        
        # Try all combinations
        for chunks_per_node in chunks_per_node_range:
            for sketch in sketches:
                try:
                    algo = self.synthesize(
                        collective_type=collective_type,
                        chunks_per_node=chunks_per_node,
                        sketch=sketch,
                        **kwargs
                    )
                    algorithms.append(algo)
                except Exception as e:
                    if kwargs.get('verbose', False):
                        print(f"Failed to synthesize with C={chunks_per_node}, "
                              f"sketch={sketch.sketch_type if sketch else 'None'}: {e}")
        
        # Filter Pareto-optimal algorithms
        pareto_algorithms = self._filter_pareto_optimal(algorithms)
        
        return pareto_algorithms
    
    def _auto_select_sketch(self, collective_type: CollectiveType) -> Optional[CommunicationSketch]:
        """Automatically select communication sketch based on topology analysis"""
        
        # Analyze topology characteristics
        is_symmetric = self._is_topology_symmetric()
        has_hierarchy = self._detect_hierarchy()
        
        # For small symmetric topologies, ring is often good
        if self.topology.num_nodes <= 8 and is_symmetric:
            try:
                return CommunicationSketch.create_ring_sketch(
                    self.topology, collective_type
                )
            except:
                pass
        
        # For hierarchical topologies (e.g., multi-node), use hierarchical sketch
        if has_hierarchy:
            groups = self._detect_node_groups()
            if len(groups) > 1:
                return CommunicationSketch.create_hierarchical_sketch(
                    self.topology, collective_type, groups
                )
        
        # Default: no sketch (let synthesizer explore freely)
        return None
    
    def _is_topology_symmetric(self) -> bool:
        """Check if topology has symmetric structure"""
        # Simple check: all nodes have same in/out degree
        in_degree = {}
        out_degree = {}
        
        for i in range(self.topology.num_nodes):
            in_degree[i] = 0
            out_degree[i] = 0
        
        for (src, dst) in self.topology.get_links():
            out_degree[src] += 1
            in_degree[dst] += 1
        
        # Check if all nodes have same degrees
        in_values = set(in_degree.values())
        out_values = set(out_degree.values())
        
        return len(in_values) == 1 and len(out_values) == 1
    
    def _detect_hierarchy(self) -> bool:
        """Detect if topology has hierarchical structure"""
        # Simple heuristic: check bandwidth heterogeneity
        bandwidths = list(self.topology.bandwidth.values())
        if not bandwidths:
            return False
        
        min_bw = min(bandwidths)
        max_bw = max(bandwidths)
        
        # If there's significant bandwidth difference, likely hierarchical
        return max_bw > 2 * min_bw
    
    def _detect_node_groups(self) -> List[List[int]]:
        """Detect node groups for hierarchical sketch"""
        # Import here to avoid circular dependency
        from .routing_heuristics import detect_node_groups
        return detect_node_groups(self.topology)
    
    def _filter_pareto_optimal(self, algorithms: List[TACCLAlgorithm]) -> List[TACCLAlgorithm]:
        """Filter algorithms to keep only Pareto-optimal ones"""
        if not algorithms:
            return []
        
        pareto_optimal = []
        
        for algo in algorithms:
            # Check if dominated by any existing Pareto-optimal algorithm
            is_dominated = False
            
            for pareto_algo in pareto_optimal:
                if (pareto_algo.latency_cost <= algo.latency_cost and
                    pareto_algo.bandwidth_cost <= algo.bandwidth_cost and
                    (pareto_algo.latency_cost < algo.latency_cost or
                     pareto_algo.bandwidth_cost < algo.bandwidth_cost)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove any algorithms dominated by this one
                pareto_optimal = [
                    p for p in pareto_optimal
                    if not (algo.latency_cost <= p.latency_cost and
                           algo.bandwidth_cost <= p.bandwidth_cost and
                           (algo.latency_cost < p.latency_cost or
                            algo.bandwidth_cost < p.bandwidth_cost))
                ]
                pareto_optimal.append(algo)
        
        # Sort by latency cost
        pareto_optimal.sort(key=lambda x: x.latency_cost)
        
        return pareto_optimal


def create_dgx2_topology() -> Topology:
    """
    Create DGX-2 topology (2 nodes with 16 GPUs each).
    
    DGX-2 has a more complex topology than DGX-1:
    - 16 GPUs per node arranged in groups
    - NVSwitch provides full connectivity within node
    - InfiniBand between nodes
    """
    bandwidth = {}
    nodes_per_machine = 16
    num_machines = 2
    total_nodes = nodes_per_machine * num_machines
    
    # Intra-node: Full connectivity via NVSwitch (high bandwidth)
    for machine in range(num_machines):
        offset = machine * nodes_per_machine
        for i in range(nodes_per_machine):
            for j in range(nodes_per_machine):
                if i != j:
                    src = offset + i
                    dst = offset + j
                    bandwidth[(src, dst)] = 12  # NVSwitch bandwidth
    
    # Inter-node: InfiniBand connections (lower bandwidth)
    # Connect corresponding GPUs across nodes
    for i in range(nodes_per_machine):
        src1 = i
        src2 = nodes_per_machine + i
        bandwidth[(src1, src2)] = 1  # InfiniBand
        bandwidth[(src2, src1)] = 1
    
    return Topology(num_nodes=total_nodes, bandwidth=bandwidth)


def create_ndv2_topology() -> Topology:
    """
    Create Azure NDv2 topology (2 nodes with 8 GPUs each).
    
    NDv2 characteristics:
    - 8 V100 GPUs per node
    - NVLink within node (but not full connectivity)
    - InfiniBand between nodes
    """
    bandwidth = {}
    nodes_per_machine = 8
    num_machines = 2
    total_nodes = nodes_per_machine * num_machines
    
    # Intra-node: Partial NVLink connectivity
    # Similar to DGX-1 pattern within each node
    for machine in range(num_machines):
        offset = machine * nodes_per_machine
        
        # Create two groups within each machine
        group1 = list(range(offset, offset + 4))
        group2 = list(range(offset + 4, offset + 8))
        
        # Full connectivity within groups
        for group in [group1, group2]:
            for i in group:
                for j in group:
                    if i != j:
                        bandwidth[(i, j)] = 2  # NVLink
        
        # Cross-group connections
        for i in range(4):
            bandwidth[(offset + i, offset + i + 4)] = 1
            bandwidth[(offset + i + 4, offset + i)] = 1
    
    # Inter-node: InfiniBand connections
    for i in range(nodes_per_machine):
        src1 = i
        src2 = nodes_per_machine + i
        bandwidth[(src1, src2)] = 1  # InfiniBand
        bandwidth[(src2, src1)] = 1
    
    return Topology(num_nodes=total_nodes, bandwidth=bandwidth)