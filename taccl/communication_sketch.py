# communication_sketch.py
"""
Communication Sketches for TACCL

Communication sketches allow algorithm designers to provide high-level 
intuitions that constrain the search space of algorithms. Inspired by 
program sketching, these provide partial specifications that guide synthesis.
"""

from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
from enum import Enum
from ..topology import Topology
from ..collective import CollectiveType

class SketchType(Enum):
    """Types of communication sketches"""
    RING = "ring"                # Ring-based communication pattern
    TREE = "tree"                # Tree-based communication pattern
    ALLTOALL = "alltoall"        # All-to-all communication pattern
    HIERARCHICAL = "hierarchical" # Hierarchical (multi-level) pattern
    CUSTOM = "custom"            # Custom pattern with explicit constraints

@dataclass
class CommunicationPattern:
    """Defines a communication pattern constraint"""
    source_nodes: Optional[Set[int]] = None      # Allowed source nodes
    dest_nodes: Optional[Set[int]] = None        # Allowed destination nodes
    allowed_links: Optional[Set[Tuple[int, int]]] = None  # Specific allowed links
    forbidden_links: Optional[Set[Tuple[int, int]]] = None  # Forbidden links
    max_hops: Optional[int] = None               # Maximum hops for chunk routing
    locality_constraint: Optional[str] = None     # e.g., "intra-node", "inter-node"

@dataclass
class TemporalConstraint:
    """Temporal constraints on communication"""
    min_steps: Optional[int] = None
    max_steps: Optional[int] = None
    concurrent_sends: Optional[int] = None       # Max concurrent sends per node
    pipelined: bool = False                      # Whether pipelining is allowed

@dataclass 
class ChunkConstraint:
    """Constraints on chunk routing"""
    chunk_groups: Optional[List[Set[int]]] = None  # Group chunks that should follow same path
    chunk_ordering: Optional[List[Tuple[int, int]]] = None  # (chunk1, chunk2) - chunk1 before chunk2
    aggregation_points: Optional[Set[int]] = None   # Nodes where aggregation can happen

@dataclass
class CommunicationSketch:
    """
    High-level communication pattern specification.
    
    This abstraction allows algorithm designers to specify:
    1. Communication patterns (ring, tree, etc.)
    2. Temporal constraints (steps, pipelining)
    3. Routing constraints (paths, locality)
    4. Chunk-specific constraints
    """
    sketch_type: SketchType
    topology: Topology
    collective_type: CollectiveType
    
    # Pattern constraints
    patterns: List[CommunicationPattern]
    
    # Temporal constraints
    temporal: TemporalConstraint
    
    # Chunk constraints
    chunk_constraints: Optional[ChunkConstraint] = None
    
    # Bandwidth allocation hints
    bandwidth_allocation: Optional[Dict[str, float]] = None  # e.g., {"intra": 0.7, "inter": 0.3}
    
    # Custom constraints as lambda functions
    custom_constraints: Optional[List] = None
    
    def validate(self) -> bool:
        """Validate the sketch is well-formed"""
        # Check pattern constraints are compatible with topology
        for pattern in self.patterns:
            if pattern.allowed_links:
                for link in pattern.allowed_links:
                    if link not in self.topology.bandwidth:
                        return False
        return True
    
    @staticmethod
    def create_ring_sketch(topology: Topology, collective_type: CollectiveType,
                          bidirectional: bool = True) -> 'CommunicationSketch':
        """Create a ring-based communication sketch"""
        # Find ring(s) in topology
        rings = CommunicationSketch._find_rings(topology)
        if not rings:
            raise ValueError("No rings found in topology")
        
        # Use first ring found
        ring = rings[0]
        allowed_links = set()
        
        for i in range(len(ring)):
            src = ring[i]
            dst = ring[(i + 1) % len(ring)]
            allowed_links.add((src, dst))
            if bidirectional:
                allowed_links.add((dst, src))
        
        pattern = CommunicationPattern(allowed_links=allowed_links)
        temporal = TemporalConstraint(pipelined=True)
        
        return CommunicationSketch(
            sketch_type=SketchType.RING,
            topology=topology,
            collective_type=collective_type,
            patterns=[pattern],
            temporal=temporal
        )
    
    @staticmethod
    def create_hierarchical_sketch(topology: Topology, collective_type: CollectiveType,
                                 intra_node_groups: List[List[int]]) -> 'CommunicationSketch':
        """Create a hierarchical communication sketch for multi-node setups"""
        patterns = []
        
        # Intra-node communication pattern
        intra_links = set()
        for group in intra_node_groups:
            for i in group:
                for j in group:
                    if i != j and (i, j) in topology.bandwidth:
                        intra_links.add((i, j))
        
        intra_pattern = CommunicationPattern(
            allowed_links=intra_links,
            locality_constraint="intra-node"
        )
        patterns.append(intra_pattern)
        
        # Inter-node communication pattern (between group representatives)
        inter_links = set()
        for i, group1 in enumerate(intra_node_groups):
            for j, group2 in enumerate(intra_node_groups):
                if i != j:
                    # Find links between groups
                    for n1 in group1:
                        for n2 in group2:
                            if (n1, n2) in topology.bandwidth:
                                inter_links.add((n1, n2))
        
        if inter_links:
            inter_pattern = CommunicationPattern(
                allowed_links=inter_links,
                locality_constraint="inter-node"
            )
            patterns.append(inter_pattern)
        
        temporal = TemporalConstraint(pipelined=True)
        
        # Bandwidth allocation hint: prioritize intra-node
        bandwidth_allocation = {"intra-node": 0.8, "inter-node": 0.2}
        
        return CommunicationSketch(
            sketch_type=SketchType.HIERARCHICAL,
            topology=topology,
            collective_type=collective_type,
            patterns=patterns,
            temporal=temporal,
            bandwidth_allocation=bandwidth_allocation
        )
    
    @staticmethod
    def create_custom_sketch(topology: Topology, collective_type: CollectiveType) -> 'CommunicationSketch':
        """Create an empty custom sketch for manual specification"""
        return CommunicationSketch(
            sketch_type=SketchType.CUSTOM,
            topology=topology,
            collective_type=collective_type,
            patterns=[],
            temporal=TemporalConstraint()
        )
    
    @staticmethod
    def _find_rings(topology: Topology) -> List[List[int]]:
        """Find rings in the topology using cycle detection"""
        import networkx as nx
        
        # Convert to networkx graph
        G = nx.DiGraph()
        for i in range(topology.num_nodes):
            G.add_node(i)
        for (src, dst) in topology.get_links():
            G.add_edge(src, dst)
        
        # Find all simple cycles
        cycles = list(nx.simple_cycles(G))
        
        # Filter to get rings that visit all or most nodes
        rings = []
        for cycle in cycles:
            if len(cycle) >= topology.num_nodes // 2:  # At least half the nodes
                rings.append(cycle)
        
        # Sort by length (prefer longer rings)
        rings.sort(key=len, reverse=True)
        
        return rings[:2]  # Return top 2 rings

def sketch_to_constraints(sketch: CommunicationSketch) -> Dict:
    """
    Convert a communication sketch to constraints for the synthesizer.
    
    Returns a dictionary of constraints that can be used by the MILP solver.
    """
    constraints = {
        'allowed_links': set(),
        'forbidden_links': set(),
        'temporal_bounds': {},
        'chunk_groups': [],
        'bandwidth_hints': {}
    }
    
    # Collect all allowed and forbidden links
    for pattern in sketch.patterns:
        if pattern.allowed_links:
            constraints['allowed_links'].update(pattern.allowed_links)
        if pattern.forbidden_links:
            constraints['forbidden_links'].update(pattern.forbidden_links)
    
    # Set temporal bounds
    if sketch.temporal.min_steps:
        constraints['temporal_bounds']['min_steps'] = sketch.temporal.min_steps
    if sketch.temporal.max_steps:
        constraints['temporal_bounds']['max_steps'] = sketch.temporal.max_steps
    
    # Chunk grouping constraints
    if sketch.chunk_constraints and sketch.chunk_constraints.chunk_groups:
        constraints['chunk_groups'] = sketch.chunk_constraints.chunk_groups
    
    # Bandwidth allocation hints
    if sketch.bandwidth_allocation:
        constraints['bandwidth_hints'] = sketch.bandwidth_allocation
    
    return constraints