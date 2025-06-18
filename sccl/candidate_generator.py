from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional
from enum import Enum
import math
from ..topology import *
from ..collective import *

@dataclass
class CandidateStats:
    """Statistics about candidate generation and filtering"""
    total_possible: int             # Total (R,C) pairs in the range
    after_bandwidth: int            # After R/C >= b_l constraint
    after_collective: int           # After collective-specific constraints
    after_topology: int             # After topology-specific constraints
    after_duplicate_ratio: int      # After removing duplicate R/C ratios
    after_pareto_pruning: int       # After Pareto dominance pruning
    final_count: int                # Final candidates
    filtered_percentage: float      # Percentage filtered out
    pareto_pruned: int              # Number pruned by Pareto dominance
    duplicate_ratios_removed: int   # Number of duplicate ratios removed

class CandidateGenerator:
    """Generate and filter (R,C) candidates for collective synthesis"""
    
    def __init__(self, k: int):
        self.k = k

    def reset_pareto_tracking(self):
        """Reset Pareto tracking for new synthesis run"""
        self.best_bandwidth_cost = float('inf')
        self.step_history = {}
    
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
        stats = CandidateStats(0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0)
        
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

        # Stage 5: Remove duplicate R/C ratios
        from fractions import Fraction
        ratio_seen = {}  # Map ratio to best (R,C) pair with that ratio
        unique_ratio_candidates = []
        
        for R, C in topology_filtered:
            ratio = Fraction(R, C)
            if ratio not in ratio_seen:
                ratio_seen[ratio] = (R, C)
                unique_ratio_candidates.append((R, C))
            else:
                # Keep the one with smaller R (fewer rounds is better)
                existing_R, existing_C = ratio_seen[ratio]
                if R < existing_R:
                    # Remove old one, add new one
                    unique_ratio_candidates.remove((existing_R, existing_C))
                    unique_ratio_candidates.append((R, C))
                    ratio_seen[ratio] = (R, C)
        
        stats.after_duplicate_ratio = len(unique_ratio_candidates)
        stats.duplicate_ratios_removed = stats.after_topology - stats.after_duplicate_ratio

        # Stage 6: Apply Pareto dominance pruning
        final_candidates = []
        pareto_pruned_count = 0
        
        for R, C in unique_ratio_candidates:
            bandwidth_cost = R / C
            
            # Check if dominated by best seen so far
            if bandwidth_cost >= self.best_bandwidth_cost:
                pareto_pruned_count += 1
                continue  # Skip dominated candidates
                
            # Check if dominated by best in previous steps
            dominated = False
            for prev_step, prev_best_rc in self.step_history.items():
                if prev_step < S and bandwidth_cost >= prev_best_rc:
                    dominated = True
                    break
            
            if not dominated:
                final_candidates.append((R, C))
            else:
                pareto_pruned_count += 1

        
        stats.after_pareto_pruning = len(final_candidates)
        stats.pareto_pruned = pareto_pruned_count
        
        # Final candidates
        stats.final_count = len(final_candidates)
        stats.filtered_percentage = ((stats.total_possible - stats.final_count) / 
                                   stats.total_possible * 100 if stats.total_possible > 0 else 0)
        
        # Print filtering statistics
        self._print_stats(S, stats, collective_type)
        
        return final_candidates, stats
    
    def update_best_bandwidth_cost(self, S: int, bandwidth_cost: float):
        """Update best bandwidth cost after finding a solution"""
        self.best_bandwidth_cost = min(self.best_bandwidth_cost, bandwidth_cost)
        
        # Update step history
        if S not in self.step_history:
            self.step_history[S] = bandwidth_cost
        else:
            self.step_history[S] = min(self.step_history[S], bandwidth_cost)
    
    def generate_candidates_simple(self, S: int, b_l: float) -> Tuple[List[Tuple[int, int]], CandidateStats]:
        """Original Method - R/C ≥ b_l"""
        candidates = []
        stats = CandidateStats(0, 0, 0, 0, 0, 0.0)  
        for R in range(S, S + self.k + 1):
            for C in range(1, int(R / b_l)):
                candidates.append((R, C))

        stats.total_possible = len(candidates)
        return candidates, stats
    
    def _check_collective_constraints(self, 
                                    collective_type: CollectiveType,
                                    R: int, C: int, P: int,
                                    topology: Topology) -> bool:
        """Check if (R,C) satisfies collective-specific constraints"""
        
        if collective_type == CollectiveType.ALLGATHER:
            # Each node needs to receive (P-1)*C chunks from others
            # In R rounds, maximum possible transfers
            topology_diameter = topology.compute_diameter()
            if R < topology_diameter:
                return False
            
            # C should not exceed what can be distributed in R rounds
            max_in_degree = topology.get_max_in_degree()
            if C > R * max_in_degree:
                return False
                
        elif collective_type == CollectiveType.ALLTOALL:
            # Need at least P chunks per node for transpose
            if C < P:
                return False
            # Each node sends (P-1)*C chunks total
            # Check if possible in R rounds
            max_out_degree = topology.get_max_out_degree()
            if (P - 1) * C > R * max_out_degree:
                return False
                
        elif collective_type == CollectiveType.BROADCAST:
            # Root sends C chunks to P-1 nodes
            root_out_degree = topology.get_node_out_degree(0)
            total_sends = C * (P - 1)
            if total_sends > R * root_out_degree:
                return False
                
        elif collective_type == CollectiveType.GATHER:
            # Root receives C chunks from P-1 nodes
            root_in_degree = topology.get_node_in_degree(0)
            total_receives = C * (P - 1)
            if total_receives > R * root_in_degree:
                return False
                
        elif collective_type == CollectiveType.SCATTER:
            # Root sends different C chunks to each of P-1 nodes
            root_out_degree = topology.get_node_out_degree(0)
            total_sends = C * (P - 1)
            if total_sends > R * root_out_degree:
                return False
        
        return True
    
    def _check_topology_constraints(self,
                                  collective_type: CollectiveType,
                                  R: int, C: int, S: int,
                                  topology: Topology) -> bool:
        """Check if (R,C) is feasible given topology structure"""
        
        # # Check if topology is a ring
        P = topology.num_nodes
            
        if collective_type == CollectiveType.ALLGATHER:
            # In ring, data must travel through all nodes
            # Need at least P-1 steps
            # In R rounds on ring, can move at most R/(P-1) chunks around
            if C > 6 :
                return False
                    
        # # Check if topology has bottlenecks
        # bottleneck_bw = self._get_bottleneck_bandwidth(topology)
        # if bottleneck_bw > 0:
        #     # Check if the bottleneck can handle the data movement
        #     if collective_type in [CollectiveType.ALLGATHER, CollectiveType.ALLREDUCE]:
        #         # All data must pass through bottleneck
        #         total_data = C * topology.num_nodes
        #         if total_data > R * bottleneck_bw:
        #             return False
        
        return True
    
    
    def _print_stats(self, S: int, stats: CandidateStats, 
                            collective_type: CollectiveType):
        """Print detailed filtering statistics"""
        print(f"\n  S={S} Candidate Generation for {collective_type.value}:")
        print(f"    Total possible (R,C) pairs: {stats.total_possible}")
        
        print(f"    After bandwidth constraint (R/C ≥ b_l): {stats.after_bandwidth} "
              f"(-{stats.total_possible - stats.after_bandwidth})")
        
        print(f"    After collective constraints: {stats.after_collective} "
              f"(-{stats.after_bandwidth - stats.after_collective})")
        
        print(f"    After topology constraints: {stats.after_topology} "
              f"(-{stats.after_collective - stats.after_topology})")
        
        print(f"    After removing duplicate R/C ratios: {stats.after_duplicate_ratio} "
              f"(-{stats.duplicate_ratios_removed})")
        
        if stats.pareto_pruned > 0:
            print(f"    After Pareto dominance pruning: {stats.after_pareto_pruning} "
                  f"(-{stats.pareto_pruned} dominated)")
        
        print(f"    Final candidates: {stats.final_count}")
        print(f"    Total filtered: {stats.filtered_percentage:.1f}%")
        
        # Show current Pareto tracking state
        if self.best_bandwidth_cost < float('inf'):
            print(f"    Current best R/C: {self.best_bandwidth_cost:.3f}")
        if self.step_history:
            print(f"    Step history: {dict(sorted(self.step_history.items()))}")