# __init__.py
"""
TACCL: Topology Aware Collective Communication Library

A tool for synthesizing efficient collective communication algorithms
using communication sketches and MILP optimization.
"""

# Core components
from .taccl_synthesizer import (
    TACCLSynthesizer,
    TACCLAlgorithm,
    create_dgx2_topology,
    create_ndv2_topology
)

# Communication sketches
from .communication_sketch import (
    CommunicationSketch,
    CommunicationPattern,
    TemporalConstraint,
    ChunkConstraint,
    SketchType,
    sketch_to_constraints
)

# MILP synthesis
from .milp_synthesizer import (
    MILPSynthesizer,
    MILPSolution
)

# Routing heuristics
from .routing_heuristics import (
    RoutingHeuristic,
    ShortestPathRouting,
    RingRouting,
    HierarchicalRouting,
    select_routing_heuristic,
    Route
)

# Backend implementations
from .backend import (
    TACCLBackend,
    NCCLBackend,
    MPIBackend,
    CustomBackend,
    BackendConfig,
    create_backend,
    generate_implementation
)

__all__ = [
    # Main synthesizer
    'TACCLSynthesizer',
    'TACCLAlgorithm',
    
    # Topologies
    'create_dgx2_topology',
    'create_ndv2_topology',
    
    # Communication sketches
    'CommunicationSketch',
    'CommunicationPattern',
    'TemporalConstraint', 
    'ChunkConstraint',
    'SketchType',
    
    # MILP
    'MILPSynthesizer',
    'MILPSolution',
    
    # Routing
    'RoutingHeuristic',
    'ShortestPathRouting',
    'RingRouting',
    'HierarchicalRouting',
    
    # Backends
    'TACCLBackend',
    'BackendConfig',
    'create_backend',
    'generate_implementation'
]

__version__ = '0.1.0'