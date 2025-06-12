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