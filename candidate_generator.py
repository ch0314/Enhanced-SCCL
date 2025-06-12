from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional
from enum import Enum
import math
from .topology import *
from .collective import *

@dataclass
class CandidateStats:
    """Statistics about candidate generation and filtering"""
    total_possible: int          # Total (R,C) pairs in the range
    after_bandwidth: int         # After R/C >= b_l constraint
    after_collective: int        # After collective-specific constraints
    after_topology: int          # After topology-specific constraints
    final_count: int            # Final candidates
    filtered_percentage: float   # Percentage filtered out

class CandidateGenerator:
    """Generate and filter (R,C) candidates for collective synthesis"""
    
    def __init__(self, k: int):
        self.k = k
    
    def generate_candidates(self, 
                          collective_type: CollectiveType,
                          topology: Topology,
                          S: int,
                          b_l: float) -> Tuple[List[Tuple[int, int]], CandidateStats]:
        """
        Generate (R,C) candidates with multi-stage filtering
        
        Returns:
            Tuple of (candidates list, statistics)
        """
        P = topology.num_nodes
        stats = CandidateStats(0, 0, 0, 0, 0, 0.0)
        
        # Stage 1: Generate all possible (R,C) pairs
        all_candidates = []
        for R in range(S, S + self.k + 1):
            for C in range(1, int(R / b_l) + 1):
                all_candidates.append((R, C))
        stats.total_possible = len(all_candidates)
        
        # Stage 2: Apply bandwidth constraint (R/C >= b_l)
        bandwidth_filtered = []
        for R, C in all_candidates:
            if R/C >= b_l:
                bandwidth_filtered.append((R, C))
        stats.after_bandwidth = len(bandwidth_filtered)
        
        # Stage 3: Apply collective-specific constraints
        collective_filtered = []
        for R, C in bandwidth_filtered:
            if self._check_collective_constraints(collective_type, R, C, P, topology):
                collective_filtered.append((R, C))
        stats.after_collective = len(collective_filtered)
        
        # Stage 4: Apply topology-specific constraints
        topology_filtered = []
        for R, C in collective_filtered:
            if self._check_topology_constraints(collective_type, R, C, S, topology):
                topology_filtered.append((R, C))
        stats.after_topology = len(topology_filtered)
        
        # Final candidates
        final_candidates = topology_filtered
        stats.final_count = len(final_candidates)
        stats.filtered_percentage = ((stats.total_possible - stats.final_count) / 
                                   stats.total_possible * 100 if stats.total_possible > 0 else 0)
        
        # Print filtering statistics
        self._print_stats(S, stats, collective_type)
        
        return final_candidates, stats
    
    def generate_candidates_simple(self, S: int, b_l: float) -> List[Tuple[int, int]]:
        """Original Method - R/C ≥ b_l"""
        candidates = []
        for R in range(S, S + self.k + 1):
            max_c = int(R / b_l)
            for C in range(1, min(max_c + 1, self.max_chunks_per_node + 1)):
                candidates.append((R, C))
        return candidates
    
    def _check_collective_constraints(self, 
                                    collective_type: CollectiveType,
                                    R: int, C: int, P: int,
                                    topology: Topology) -> bool:
        """Check if (R,C) satisfies collective-specific constraints"""
        
        if collective_type == CollectiveType.ALLGATHER:
            # Each node needs to receive (P-1)*C chunks from others
            # In R rounds, maximum possible transfers
            min_rounds_needed = P - 1  # Minimum for diameter
            if R < min_rounds_needed:
                return False
            # C should not exceed what can be distributed in R rounds
            max_in_degree = self._get_max_in_degree(topology)
            if C > R * max_in_degree:
                return False
                
        elif collective_type == CollectiveType.ALLTOALL:
            # Need at least P chunks per node for transpose
            if C < P:
                return False
            # Each node sends (P-1)*C chunks total
            # Check if possible in R rounds
            max_out_degree = self._get_max_out_degree(topology)
            if (P - 1) * C > R * max_out_degree:
                return False
                
        elif collective_type == CollectiveType.BROADCAST:
            # Root sends C chunks to P-1 nodes
            root_out_degree = self._get_node_out_degree(0, topology)
            total_sends = C * (P - 1)
            if total_sends > R * root_out_degree:
                return False
                
        elif collective_type == CollectiveType.GATHER:
            # Root receives C chunks from P-1 nodes
            root_in_degree = self._get_node_in_degree(0, topology)
            total_receives = C * (P - 1)
            if total_receives > R * root_in_degree:
                return False
                
        elif collective_type == CollectiveType.SCATTER:
            # Root sends different C chunks to each of P-1 nodes
            root_out_degree = self._get_node_out_degree(0, topology)
            total_sends = C * (P - 1)
            if total_sends > R * root_out_degree:
                return False
        
        return True
    
    def _check_topology_constraints(self,
                                  collective_type: CollectiveType,
                                  R: int, C: int, S: int,
                                  topology: Topology) -> bool:
        """Check if (R,C) is feasible given topology structure"""
        
        # Check if topology is a ring
        if self._is_ring_topology(topology):
            P = topology.num_nodes
            
            if collective_type == CollectiveType.ALLGATHER:
                # In ring, data must travel through all nodes
                # Need at least P-1 steps
                if S < P - 1:
                    return False
                # In R rounds on ring, can move at most R/(P-1) chunks around
                if C > R / (P - 1):
                    return False
                    
        # Check if topology has bottlenecks
        bottleneck_bw = self._get_bottleneck_bandwidth(topology)
        if bottleneck_bw > 0:
            # Check if the bottleneck can handle the data movement
            if collective_type in [CollectiveType.ALLGATHER, CollectiveType.ALLREDUCE]:
                # All data must pass through bottleneck
                total_data = C * topology.num_nodes
                if total_data > R * bottleneck_bw:
                    return False
        
        return True
    
    def _get_max_in_degree(self, topology: Topology) -> int:
        """Get maximum incoming bandwidth for any node"""
        in_degree = {}
        for (src, dst), bw in topology.bandwidth.items():
            in_degree[dst] = in_degree.get(dst, 0) + bw
        return max(in_degree.values()) if in_degree else 0
    
    def _get_max_out_degree(self, topology: Topology) -> int:
        """Get maximum outgoing bandwidth for any node"""
        out_degree = {}
        for (src, dst), bw in topology.bandwidth.items():
            out_degree[src] = out_degree.get(src, 0) + bw
        return max(out_degree.values()) if out_degree else 0
    
    def _get_node_out_degree(self, node: int, topology: Topology) -> int:
        """Get outgoing bandwidth for specific node"""
        out_degree = 0
        for (src, dst), bw in topology.bandwidth.items():
            if src == node:
                out_degree += bw
        return out_degree
    
    def _get_node_in_degree(self, node: int, topology: Topology) -> int:
        """Get incoming bandwidth for specific node"""
        in_degree = 0
        for (src, dst), bw in topology.bandwidth.items():
            if dst == node:
                in_degree += bw
        return in_degree
    
    def _is_ring_topology(self, topology: Topology) -> bool:
        """Check if topology is a ring"""
        # Simple check: each node has exactly 2 connections (in and out)
        in_degree = {}
        out_degree = {}
        for (src, dst), _ in topology.bandwidth.items():
            out_degree[src] = out_degree.get(src, 0) + 1
            in_degree[dst] = in_degree.get(dst, 0) + 1
        
        for node in range(topology.num_nodes):
            if in_degree.get(node, 0) != 1 or out_degree.get(node, 0) != 1:
                return False
        return True
    
    def _get_bottleneck_bandwidth(self, topology: Topology) -> int:
        """Find bottleneck link bandwidth (simplified)"""
        # For now, return minimum bandwidth
        if not topology.bandwidth:
            return 0
        return min(topology.bandwidth.values())
    
    def _print_stats(self, S: int, stats: CandidateStats, 
                    collective_type: CollectiveType):
        """Print filtering statistics"""
        print(f"\n  S={S} Candidate Generation for {collective_type.value}:")
        print(f"    Total possible (R,C) pairs: {stats.total_possible}")
        print(f"    After bandwidth constraint (R/C ≥ b_l): {stats.after_bandwidth} "
              f"(-{stats.total_possible - stats.after_bandwidth})")
        print(f"    After collective constraints: {stats.after_collective} "
              f"(-{stats.after_bandwidth - stats.after_collective})")
        print(f"    After topology constraints: {stats.after_topology} "
              f"(-{stats.after_collective - stats.after_topology})")
        print(f"    Final candidates: {stats.final_count}")
        print(f"    Total filtered: {stats.filtered_percentage:.1f}%")