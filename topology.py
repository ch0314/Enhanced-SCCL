from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, Optional, FrozenSet
from enum import Enum
import networkx as nx
import itertools
from z3 import *

@dataclass
class BandwidthConstraint:
    """Represents a bandwidth constraint on a set of links"""
    link_set: FrozenSet[Tuple[int, int]]  # Set of (src, dst) links
    bandwidth: int  # Maximum total bandwidth for this set
    
    def __repr__(self):
        sorted_links = sorted(list(self.link_set))
        return f"BandwidthConstraint(links=frozenset({sorted_links}), bw={self.bandwidth})"
    
    def __lt__(self, other):
        if self.bandwidth != other.bandwidth:
            return self.bandwidth < other.bandwidth
        return sorted(list(self.link_set)) < sorted(list(other.link_set))

    
@dataclass
class Topology:
    """
    Network Topology Definition
    
    Following SCCL paper notation:
    - B: Set of bandwidth constraints, where each element is (L, b)
    - L: Set of links (subset of all possible links)
    - b: Bandwidth limit for that set of links
    """
    num_nodes: int

    # Individual link bandwidths (for backward compatibility)
    bandwidth: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Bandwidth constraint sets: List of (link_set, bandwidth) constraints
    bandwidth_constraints: List[BandwidthConstraint] = field(default_factory=list)

    def add_node_outgoing_constraint(self, node: int, bandwidth: int):
        """Add constraint on total outgoing bandwidth from a node"""
        outgoing_links = {(src, dst) for (src, dst) in self.get_all_possible_links() 
                         if src == node}
        if outgoing_links:
            constraint = BandwidthConstraint(frozenset(outgoing_links), bandwidth)
            self.bandwidth_constraints.append(constraint)
    
    def add_node_incoming_constraint(self, node: int, bandwidth: int):
        """Add constraint on total incoming bandwidth to a node"""
        incoming_links = {(src, dst) for (src, dst) in self.get_all_possible_links() 
                         if dst == node}
        if incoming_links:
            constraint = BandwidthConstraint(frozenset(incoming_links), bandwidth)
            self.bandwidth_constraints.append(constraint)
    
    def add_shared_bus_constraint(self, nodes: Set[int], bandwidth: int):
        """Add shared bus constraint - only 'bandwidth' transfers among nodes per round"""
        bus_links = {(a, b) for a in nodes for b in nodes if a != b}
        constraint = BandwidthConstraint(frozenset(bus_links), bandwidth)
        self.bandwidth_constraints.append(constraint)
    
    def add_bidirectional_constraint(self, node1: int, node2: int, bandwidth: int):
        """Add constraint for bidirectional link (both directions share bandwidth)"""
        links = {(node1, node2), (node2, node1)}
        constraint = BandwidthConstraint(frozenset(links), bandwidth)
        self.bandwidth_constraints.append(constraint)
    
    def get_links(self) -> List[Tuple[int, int]]:
        """Get all links in the topology"""
        return list(self.bandwidth.keys())
    
    def get_all_possible_links(self) -> Set[Tuple[int, int]]:
        """Get all possible links based on individual bandwidth dict"""
        return set(self.bandwidth.keys())
    
    def get_links(self) -> List[Tuple[int, int]]:
        """Get all links with non-zero bandwidth"""
        return [(src, dst) for (src, dst), bw in self.bandwidth.items() if bw > 0]
    
    def compute_diameter(self) -> int:
        """Compute the diameter of the topology (longest shortest path)"""
        # Convert to networkx graph
        G = nx.DiGraph()
        G.add_edges_from(self.get_links())
        
        if not nx.is_strongly_connected(G):
            return float('inf')
        
        return nx.diameter(G)
        
    
    def compute_constrained_bisection_bandwidth(self) -> float:
        """
        Compute inverse bisection bandwidth considering bandwidth constraint sets.
        
        This is more complex than simple summation because:
        1. Links in the same constraint set compete for bandwidth
        2. We need to find the maximum achievable cross-partition bandwidth
        under all constraints
        """
        n = self.num_nodes
        if n < 2:
            return float('inf')
        
        min_bisection = float('inf')
        partition_size = n // 2
        
        # Get all bandwidth constraints
        constraints = self.get_bandwidth_constraints_for_step()
        
        # Enumerate all balanced partitions
        for partition_a in itertools.combinations(range(n), partition_size):
            partition_a = set(partition_a)
            partition_b = set(range(n)) - partition_a
            
            # Find all cross-partition links
            cross_links = set()
            for a in partition_a:
                for b in partition_b:
                    if (a, b) in self.bandwidth:
                        cross_links.add((a, b))
                    if (b, a) in self.bandwidth:
                        cross_links.add((b, a))
            
            if not cross_links:
                min_bisection = 0
                continue
            
            # Compute maximum bandwidth achievable with constraints
            max_bandwidth = self.compute_max_bandwidth_with_constraints(
                cross_links, constraints, self.bandwidth
            )
            
            min_bisection = min(min_bisection, max_bandwidth)
        
        return 1.0 / min_bisection if min_bisection > 0 else float('inf')


    def compute_max_bandwidth_with_constraints(
        self,
        target_links: Set[Tuple[int, int]], 
        constraints: List[Tuple[Set[Tuple[int, int]], int]],
        individual_bandwidths: Dict[Tuple[int, int], int]
    ) -> float:
        """
        Compute maximum bandwidth achievable on target_links under constraints.
        
        This is a linear programming problem:
        - Variables: x_{(s,d)} = bandwidth used on link (s,d)
        - Objective: maximize sum of x_{(s,d)} for (s,d) in target_links
        - Constraints: 
        - For each constraint set (L, b): sum of x_{(s,d)} for (s,d) in L ≤ b
        - For each link: 0 ≤ x_{(s,d)} ≤ individual_bandwidth_{(s,d)}
        """
        # Use Z3's optimization solver
        opt = Optimize()
        
        # Create variables for each link's utilized bandwidth
        link_vars = {}
        for link in target_links:
            if link in individual_bandwidths:
                var = Real(f'x_{link[0]}_{link[1]}')
                link_vars[link] = var
                # Individual link capacity constraint
                opt.add(var >= 0)
                opt.add(var <= individual_bandwidths[link])
        
        if not link_vars:
            return 0
        
        # Add constraint set limitations
        for link_set, bandwidth in constraints:
            # Find which links in this constraint set are in our target
            constrained_links = []
            for link in link_set:
                if link in link_vars:
                    constrained_links.append(link_vars[link])
            
            # Add constraint if any target links are in this set
            if constrained_links:
                opt.add(Sum(constrained_links) <= bandwidth)
        
        # Objective: maximize total bandwidth on target links
        opt.maximize(Sum(list(link_vars.values())))
        
        # Solve
        if opt.check() == sat:
            model = opt.model()
            total = 0
            for var in link_vars.values():
                val = model.evaluate(var, model_completion=True)
                if is_real(val):
                    total += float(val.as_fraction())
                elif is_int_value(val):
                    total += val.as_long()
            return total
        else:
            return 0


    def compute_inv_bisection_bandwidth(self) -> float:
        """
        Compute approximate bisection bandwidth for common topologies.
        This is faster but less accurate than the full constraint-aware computation.
        """
        n = self.num_nodes
        
        # Special case: Ring topology
        if self.is_ring_topology():
            # For a ring, bisection bandwidth is 2 (or 1 if bidirectional constraints)
            if self.bandwidth_constraints:
                # Check if bidirectional constraints exist
                for bc in self.bandwidth_constraints:
                    if len(bc.link_set) == 2:  # Likely bidirectional
                        return 0.5  # Inverse of 2 constrained links
            return 0.5  # Inverse of 2 unconstrained links
        
        # Special case: Shared bus
        if any(len(bc.link_set) == n*(n-1) for bc in self.bandwidth_constraints):
            # All links share bandwidth
            bus_bandwidth = min(bc.bandwidth for bc in self.bandwidth_constraints 
                            if len(bc.link_set) == n*(n-1))
            return 1.0 / bus_bandwidth
        
        # Otherwise use full computation
        return self.compute_constrained_bisection_bandwidth()
    
    def get_bandwidth_constraints_for_step(self) -> List[Tuple[Set[Tuple[int, int]], int]]:
        """Get bandwidth constraints in format for SMT encoding"""
        constraints = []
        
        # Add individual link constraints
        for (src, dst), bw in self.bandwidth.items():
            if bw > 0:
                constraints.append(({(src, dst)}, bw))
        
        # Add constraint sets
        for bc in self.bandwidth_constraints:
            constraints.append((bc.link_set, bc.bandwidth))
        
        return constraints
    
    def get_max_in_degree(self) -> int:
        """Get maximum incoming bandwidth for any node"""
        in_degree = {}
        for (src, dst), bw in self.bandwidth.items():
            in_degree[dst] = in_degree.get(dst, 0) + bw
        return max(in_degree.values()) if in_degree else 0
    
    def get_max_out_degree(self) -> int:
        """Get maximum outgoing bandwidth for any node"""
        out_degree = {}
        for (src, dst), bw in self.bandwidth.items():
            out_degree[src] = out_degree.get(src, 0) + bw
        return max(out_degree.values()) if out_degree else 0
    
    def get_node_out_degree(self, node: int) -> int:
        """Get outgoing bandwidth for specific node"""
        out_degree = 0
        for (src, dst), bw in self.bandwidth.items():
            if src == node:
                out_degree += bw
        return out_degree
    
    def get_node_in_degree(self, node: int) -> int:
        """Get incoming bandwidth for specific node"""
        in_degree = 0
        for (src, dst), bw in self.bandwidth.items():
            if dst == node:
                in_degree += bw
        return in_degree
    
    def is_ring_topology(self) -> bool:
        """Check if topology is a ring"""
        # Simple check: each node has exactly 2 connections (in and out)
        in_degree = {}
        out_degree = {}
        for (src, dst), _ in self.bandwidth.items():
            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[dst] = in_degree.get(dst, 0) + 1
        
        for node in range(self.num_nodes):
            if in_degree.get(node, 0) != 1 or out_degree.get(node, 0) != 1:
                return False
        return True
    
    def get_bottleneck_bandwidth(self) -> int:
        """Find bottleneck link bandwidth (simplified)"""
        # For now, return minimum bandwidth
        if not self.bandwidth:
            return 0
        return min(self.bandwidth.values())


# Create 4-node ring topology
def create_ring_topology(num_nodes: int) -> Topology:
    """Create ring topology"""
    
    topology = Topology(num_nodes=num_nodes)

    for i in range(num_nodes):
        # Bidirectional ring
        next_node = (i + 1) % num_nodes
            
        # Add individual links
        topology.bandwidth[(i, next_node)] = 1
        topology.bandwidth[(next_node, i)] = 1
            
        topology.add_bidirectional_constraint(i, next_node, 1)
    
    return topology


def create_dgx1_topology() -> Topology:
    """
    Create DGX-1 topology as described in the SCCL paper.
    
    DGX-1 has 8 GPUs arranged in two groups:
    - Group 1: GPUs 0,1,2,3 (fully connected)
    - Group 2: GPUs 4,5,6,7 (fully connected)
    - Inter-group links: 1-4, 0-5, 2-7, 3-6
    
    Forms two non-overlapping rings:
    - Ring 1: 0-1-4-5-6-7-2-3 (2 NVLinks per edge)
    - Ring 2: 0-2-1-3-6-4-7-5 (1 NVLink per edge)
    """
    
    topology = Topology(num_nodes=8)
    
    # Intra-group connections (within each group of 4)
    # Group 1: 0,1,2,3
    for i in range(4):
        for j in range(4):
            if i != j:
                # Check if this edge belongs to ring 1 (2 NVLinks)
                if (i, j) in [(0,1), (1,0), (2,3), (3,2), (0,3), (3,0)]:
                    topology.bandwidth[(i, j)] = 2
                    if(i < j):
                        topology.add_bidirectional_constraint(i, j, 2)
                else:
                    topology.bandwidth[(i, j)] = 1
                    if(i < j):
                        topology.add_bidirectional_constraint(i, j, 1)
    
    # Group 2: 4,5,6,7
    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                # Check if this edge belongs to ring 1 (2 NVLinks)
                if (i, j) in [(4,5), (5,4), (5,6), (6,5), (6,7), (7,6)]:
                    topology.bandwidth[(i, j)] = 2
                    if(i < j):
                        topology.add_bidirectional_constraint(i, j, 2)
                else:
                    topology.bandwidth[(i, j)] = 1
                    if(i < j):
                        topology.add_bidirectional_constraint(i, j, 1)
    
    # Inter-group connections
    inter_group_links = [
        (1, 4), (4, 1),  # These are part of ring 1 (2 NVLinks)
        (2, 7), (7, 2),  # These are part of ring 2 (2 NVLinks)
        (0, 5), (5, 0),  # These are part of ring 1 (1 NVLink)
        (3, 6), (6, 3)   # These are part of ring 2 (1 NVLink)
    ]
    
    for src, dst in inter_group_links:
        if (src, dst) in [(1,4), (4,1), (2,7), (7,2)]:
            topology.bandwidth[(src, dst)] = 2
            if(src < dst):
                topology.add_bidirectional_constraint(src, dst, 2)
        else:
            topology.bandwidth[(src, dst)] = 1
            if(src < dst):
                topology.add_bidirectional_constraint(src, dst, 1)

    # sorted_items_by_key = sorted(topology.bandwidth.items())
    # for key, value in sorted_items_by_key:
    #     print(f"{key}: {value}")

    # sorted_data_by_bw = sorted(topology.bandwidth_constraints, key=lambda bc: bc.bandwidth)
    # for constraint in sorted_data_by_bw:
    #     print(constraint)

    return topology

def create_shared_bus_topology(num_nodes: int, bus_bandwidth: int = 1) -> Topology:
    """
    Create shared bus topology where only one transfer can happen per round.
    """
    topology = Topology(num_nodes=num_nodes)
    
    # All nodes can communicate with each other
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                topology.bandwidth[(i, j)] = 1  # Individual link capacity
    
    # But only bus_bandwidth transfers can happen globally per round
    all_nodes = set(range(num_nodes))
    topology.add_shared_bus_constraint(all_nodes, bus_bandwidth)
    
    return topology


def create_star_topology(num_nodes: int, center_bandwidth: int = 2) -> Topology:
    """
    Create star topology with bandwidth constraints on center node.
    """
    topology = Topology(num_nodes=num_nodes)
    center = 0  # Node 0 is the center
    
    # Connect all nodes to center
    for i in range(1, num_nodes):
        topology.bandwidth[(center, i)] = 1
        topology.bandwidth[(i, center)] = 1
    
    # Add constraints on center node's total bandwidth
    topology.add_node_outgoing_constraint(center, center_bandwidth)
    topology.add_node_incoming_constraint(center, center_bandwidth)
    
    return topology
