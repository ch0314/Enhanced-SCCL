from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional
from enum import Enum
import networkx as nx
import itertools


@dataclass
class Topology:
    """Network Topology Definition"""
    num_nodes: int
    bandwidth: Dict[Tuple[int, int], int]  # (src, dst) -> bandwidth in chunks per round
    
    def get_links(self) -> List[Tuple[int, int]]:
        """Get all links in the topology"""
        return list(self.bandwidth.keys())
    
    def compute_diameter(self) -> int:
        """Compute the diameter of the topology (longest shortest path)"""
        # Convert to networkx graph
        G = nx.DiGraph()
        for i in range(self.num_nodes):
            G.add_node(i)
        for (src, dst) in self.get_links():
            G.add_edge(src, dst)
        
        # Compute all shortest paths
        try:
            # For strongly connected graphs
            all_paths = dict(nx.all_pairs_shortest_path_length(G))
            diameter = 0
            for src in all_paths:
                for dst in all_paths[src]:
                    if src != dst:
                        diameter = max(diameter, all_paths[src][dst])
            return diameter
        except:
            # If not strongly connected, return a large value
            return self.num_nodes
        
    
    def compute_inv_bisection_bandwidth(self) -> float:
        """
        Compute inverse bisection bandwidth.
        For simplicity, we use the minimum cut between any partition.
        """
        n = self.num_nodes
        if n < 2:
            return float('inf')
        
        # Build adjacency matrix for easier computation
        bandwidth_matrix = {}
        for (src, dst), bw in self.bandwidth.items():
            bandwidth_matrix[(src, dst)] = bw
        
        min_bisection = float('inf')
        partition_size = n // 2
        
        # Enumerate all balanced partitions
        for partition_a in itertools.combinations(range(n), partition_size):
            partition_a = set(partition_a)
            partition_b = set(range(n)) - partition_a

            # Calculate total bandwidth between partitions
            cross_bandwidth = 0
            for a in partition_a:
                for b in partition_b:
                    # Check both directions
                    cross_bandwidth += bandwidth_matrix.get((a, b), 0)
                    cross_bandwidth += bandwidth_matrix.get((b, a), 0)
                
            min_bisection = min(min_bisection, cross_bandwidth)
        
        # Inverse bisection bandwidth
        return 1.0 / min_bisection if min_bisection > 0 else float('inf')


# Create 4-node ring topology
def create_ring_topology(num_nodes: int) -> Topology:
    """Create ring topology"""
    bandwidth = {}
    for i in range(num_nodes):
        # Bidirectional ring
        bandwidth[(i, (i + 1) % num_nodes)] = 1
        bandwidth[((i + 1) % num_nodes, i)] = 1
    
    return Topology(num_nodes=num_nodes, bandwidth=bandwidth)


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
