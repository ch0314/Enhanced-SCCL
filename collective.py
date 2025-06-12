from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional
from enum import Enum


class CollectiveType(Enum):
    """Supported collective operations"""
    GATHER = "gather"
    SCATTER = "scatter"
    ALLGATHER = "allgather"
    ALLTOALL = "alltoall"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    ALLREDUCE = "allreduce"
    REDUCESCATTER = "reducescatter"

class RelationType(Enum):
    """Relation types for pre/post conditions"""
    ALL = "all"           # All chunks on all nodes
    ROOT = "root"         # All chunks on root node only
    SCATTERED = "scattered"  # Chunks distributed: chunk c on node (c mod P)
    TRANSPOSE = "transpose"  # Transpose pattern for alltoall

@dataclass
class Collective:
    """Collective communication definition"""
    collective_type: CollectiveType
    num_nodes: int                      # P
    num_chunks: int                     # G (global chunks)
    chunks_per_node: int                # C (per-node chunks)
    precondition: Set[Tuple[int, int]]  # (chunk_id, node_id)
    postcondition: Set[Tuple[int, int]] # (chunk_id, node_id)

    @staticmethod
    def _create_relation(relation_type: RelationType, num_chunks: int, 
                        num_nodes: int, root_node: int = 0) -> Set[Tuple[int, int]]:
        """Create a relation based on the type"""
        if relation_type == RelationType.ALL:
            # [G] × [P] - All chunks on all nodes
            return {(c, n) for c in range(num_chunks) for n in range(num_nodes)}
        
        elif relation_type == RelationType.ROOT:
            # [G] × {n_root} - All chunks on root node only
            return {(c, root_node) for c in range(num_chunks)}
        
        elif relation_type == RelationType.SCATTERED:
            # {(c, n) ∈ [G] × [P] | n = c mod P} - Scattered pattern
            return {(c, c % num_nodes) for c in range(num_chunks)}
        
        elif relation_type == RelationType.TRANSPOSE:
            # {(c, n) ∈ [G] × [P] | n = ⌊c/P⌋ mod P} - Transpose pattern
            # For alltoall, chunk c goes to node (c // num_nodes) % num_nodes
            return {(c, (c // num_nodes) % num_nodes) for c in range(num_chunks)}
        
        else:
            raise ValueError(f"Unknown relation type: {relation_type}")
    
    @staticmethod
    def create_collective(coll_type: CollectiveType, num_nodes: int, 
                         chunks_per_node: int, root_node: int = 0) -> 'Collective':
        """Factory method to create collective with proper pre/post conditions"""
        
        # Define pre/post condition mappings based on Table 2 from the paper
        collective_specs = {
            CollectiveType.GATHER: {
                'pre': RelationType.SCATTERED,
                'post': RelationType.ROOT,
                'num_chunks': num_nodes * chunks_per_node
            },
            CollectiveType.ALLGATHER: {
                'pre': RelationType.SCATTERED,
                'post': RelationType.ALL,
                'num_chunks': num_nodes * chunks_per_node
            },
            CollectiveType.ALLTOALL: {
                'pre': RelationType.SCATTERED,
                'post': RelationType.TRANSPOSE,
                'num_chunks': num_nodes * chunks_per_node
            },
            CollectiveType.BROADCAST: {
                'pre': RelationType.ROOT,
                'post': RelationType.ALL,
                'num_chunks': chunks_per_node  # Only root's chunks are broadcast
            },
            CollectiveType.SCATTER: {
                'pre': RelationType.ROOT,
                'post': RelationType.SCATTERED,
                'num_chunks': num_nodes * chunks_per_node
            }
        }
        
        if coll_type not in collective_specs:
            raise NotImplementedError(f"Collective type {coll_type} not implemented")
        
        spec = collective_specs[coll_type]
        num_chunks = spec['num_chunks']
        
        # Create pre and post conditions using the relations
        precondition = Collective._create_relation(
            spec['pre'], num_chunks, num_nodes, root_node
        )
        postcondition = Collective._create_relation(
            spec['post'], num_chunks, num_nodes, root_node
        )
        
        return Collective(
            collective_type=coll_type,
            num_nodes=num_nodes,
            num_chunks=num_chunks,
            chunks_per_node=chunks_per_node,
            precondition=precondition,
            postcondition=postcondition
        )
