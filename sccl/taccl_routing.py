# taccl_router.py
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from ..topology import Topology
from ..collective import Collective

class TACCLRouter:
    """
    TACCL-based routing solver that generates paths for SMT solver.
    
    Generates (src, chunk, dest) paths using Gurobi MILP solver.
    """
    
    def __init__(self, topology: Topology, collective: Collective, 
                 num_steps: int, num_rounds: int, 
                 initial_paths: Optional[List[Tuple[int, int, int]]] = None,
                 verbose: bool = False):
        """
        Initialize TACCL Router.
        
        Args:
            topology: Network topology
            collective: Collective operation (pre/post conditions)
            num_steps: Number of steps (S)
            num_rounds: Total rounds (R)
            initial_paths: Optional initial paths to refine
            verbose: Print solver output
        """
        self.topology = topology
        self.collective = collective
        self.num_steps = num_steps
        self.num_rounds = num_rounds
        self.initial_paths = initial_paths
        self.verbose = verbose
        
        self.num_nodes = topology.num_nodes
        self.num_chunks = collective.num_chunks
        
    def generate_paths(self, time_limit: int = 60) -> List[Tuple[int, int, int]]:
        """
        Generate routing paths using MILP.
        
        Returns:
            List of (src, chunk, dest) tuples
        """
        model = gp.Model("TACCL_Routing")
        model.Params.TimeLimit = time_limit
        
        if not self.verbose:
            model.Params.LogToConsole = 0
        
        # Variables: is_sent[c,i,j]
        is_sent = {}
        for c in range(self.num_chunks):
            for (i, j) in self.topology.get_links():
                is_sent[c,i,j] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"is_sent_{c}_{i}_{j}"
                )
        
        # Auxiliary variable: max congestion on any link
        max_congestion = model.addVar(vtype=GRB.INTEGER, name="max_congestion")
        
        # Objective: Minimize maximum congestion (primary) and total hops (secondary)
        total_hops = gp.quicksum(is_sent[c,i,j] 
                                for c in range(self.num_chunks)
                                for (i,j) in self.topology.get_links())
        
        model.setObjective(
            max_congestion ,
            GRB.MINIMIZE
        )
        
        # Constraints
        
        # 1. Flow constraints for each chunk
        for c in range(self.num_chunks):
            # Find source and destinations
            sources = [n for (ch, n) in self.collective.precondition if ch == c]
            destinations = [n for (ch, n) in self.collective.postcondition if ch == c]
            
            for n in range(self.num_nodes):
                inflow = gp.quicksum(
                    is_sent[c,i,n]
                    for i in range(self.num_nodes)
                    if (i,n) in self.topology.get_links()
                )
                outflow = gp.quicksum(
                    is_sent[c,n,j]
                    for j in range(self.num_nodes)
                    if (n,j) in self.topology.get_links()
                )
                
                if n in sources:
                    # Source constraint
                    if n in destinations:
                        # Source that is also destination
                        model.addConstr(outflow >= 0)
                    else:
                        # Pure source
                        model.addConstr(outflow - inflow >= 1)
                elif n in destinations:
                    # Pure destination
                    model.addConstr(inflow - outflow >= 1)
                else:
                    # Intermediate node
                    model.addConstr(inflow == outflow)
        
        # 2. Congestion constraints
        for (i, j) in self.topology.get_links():
            link_usage = gp.quicksum(
                is_sent[c,i,j] for c in range(self.num_chunks)
            )
            bandwidth = self.topology.bandwidth.get((i,j), 1)
            
            # Max congestion tracks the worst link utilization
            model.addConstr(
                max_congestion >= link_usage / bandwidth
            )
        
        # 3. Path coherence (optional): avoid too many hops
        for c in range(self.num_chunks):
            path_length = gp.quicksum(
                is_sent[c,i,j]
                for (i,j) in self.topology.get_links()
            )
            # Reasonable upper bound on path length
            model.addConstr(path_length <= 2 * self.topology.compute_diameter())
        
        # Solve
        model.optimize()
        
        if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return []
        
        # Extract paths
        paths = []
        for c in range(self.num_chunks):
            for (i, j) in self.topology.get_links():
                if is_sent[c,i,j].x > 0.5:
                    paths.append((i, c, j))
        
        # Print congestion info
        if self.verbose:
            print(f"\nMax congestion: {max_congestion.x:.2f}")
            self._print_link_usage(paths)
        
        return paths
    
    def _print_link_usage(self, paths):
        """Print link usage statistics"""
        link_usage = defaultdict(int)
        for (src, chunk, dst) in paths:
            link_usage[(src, dst)] += 1
        
        print("\nLink usage:")
        for (src, dst), count in sorted(link_usage.items()):
            bandwidth = self.topology.bandwidth.get((src, dst), 1)
            utilization = count / bandwidth
            print(f"  ({src},{dst}): {count} chunks, "
                  f"bandwidth={bandwidth}, utilization={utilization:.1f}")