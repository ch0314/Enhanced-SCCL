# milp_synthesizer.py
"""
MILP-based Synthesis Engine for TACCL

This module implements the novel MILP encoding that allows TACCL to scale
beyond single-node topologies by:
1. First solving a bandwidth-relaxed routing problem
2. Applying ordering heuristics
3. Solving a bandwidth-constrained scheduling problem
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from ..topology import Topology
from ..collective import Collective, CollectiveType
from .communication_sketch import CommunicationSketch, sketch_to_constraints

@dataclass
class MILPSolution:
    """Solution from MILP solver"""
    sends: List[Dict]  # List of sends with chunk, src, dst, step
    rounds: List[int]  # Number of rounds at each step
    schedule: Dict[Tuple[int, int], int]  # (chunk, node) -> arrival time
    objective_value: float
    solve_time: float
    
class MILPSynthesizer:
    """
    MILP-based synthesizer for collective algorithms.
    
    Key innovation: Two-phase approach
    1. Bandwidth-relaxed routing problem
    2. Bandwidth-constrained scheduling problem
    """
    
    def __init__(self, topology: Topology, collective: Collective, 
                 sketch: Optional[CommunicationSketch] = None,
                 verbose: bool = False):
        self.topology = topology
        self.collective = collective
        self.sketch = sketch
        self.verbose = verbose
        
        # Extract sketch constraints if provided
        self.sketch_constraints = sketch_to_constraints(sketch) if sketch else None
        
        # Problem parameters
        self.num_nodes = topology.num_nodes
        self.num_chunks = collective.num_chunks
        self.num_steps = None  # To be determined
        
    def synthesize(self, max_steps: int = 10, time_limit: int = 300) -> MILPSolution:
        """
        Main synthesis method using two-phase approach.
        
        Args:
            max_steps: Maximum number of steps to consider
            time_limit: Time limit in seconds for each phase
            
        Returns:
            MILPSolution with the synthesized algorithm
        """
        # Phase 1: Solve bandwidth-relaxed routing problem
        if self.verbose:
            print("Phase 1: Solving bandwidth-relaxed routing problem...")
        
        routing_solution = self._solve_routing_phase(max_steps, time_limit // 2)
        
        if routing_solution is None:
            raise ValueError("No feasible routing solution found")
        
        # Phase 2: Apply ordering heuristics and solve scheduling
        if self.verbose:
            print("Phase 2: Solving bandwidth-constrained scheduling problem...")
        
        final_solution = self._solve_scheduling_phase(
            routing_solution, 
            time_limit // 2
        )
        
        return final_solution
    
    def _solve_routing_phase(self, max_steps: int, time_limit: int) -> Optional[Dict]:
        """
        Phase 1: Bandwidth-relaxed routing problem.
        
        Find valid routes for chunks without strict bandwidth constraints.
        """
        model = gp.Model("TACCL_Routing")
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 0.01
        
        if not self.verbose:
            model.Params.LogToConsole = 0
        
        # Variables
        # x[c,n,t]: whether chunk c is at node n at time t
        x = {}
        for c in range(self.num_chunks):
            for n in range(self.num_nodes):
                for t in range(max_steps + 1):
                    x[c,n,t] = model.addVar(vtype=GRB.BINARY, 
                                           name=f"x_{c}_{n}_{t}")
        
        # y[c,i,j,t]: whether chunk c is sent from i to j at time t
        y = {}
        for c in range(self.num_chunks):
            for (i, j) in self.topology.get_links():
                # Apply sketch constraints if available
                if self._is_link_allowed(i, j):
                    for t in range(max_steps):
                        y[c,i,j,t] = model.addVar(vtype=GRB.BINARY,
                                                 name=f"y_{c}_{i}_{j}_{t}")
        
        # Objective: Minimize total time
        obj = gp.LinExpr()
        for c in range(self.num_chunks):
            for n in range(self.num_nodes):
                if (c, n) in self.collective.postcondition:
                    for t in range(max_steps + 1):
                        obj += t * x[c,n,t]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        
        # 1. Preconditions
        for (c, n) in self.collective.precondition:
            model.addConstr(x[c,n,0] == 1, f"pre_{c}_{n}")
        
        # 2. Postconditions must be satisfied
        for (c, n) in self.collective.postcondition:
            model.addConstr(
                gp.quicksum(x[c,n,t] for t in range(max_steps + 1)) == 1,
                f"post_{c}_{n}"
            )
        
        # 3. Chunk can only be at one place at a time
        for c in range(self.num_chunks):
            for t in range(max_steps + 1):
                model.addConstr(
                    gp.quicksum(x[c,n,t] for n in range(self.num_nodes)) >= 1,
                    f"presence_{c}_{t}"
                )
        
        # 4. Flow conservation
        for c in range(self.num_chunks):
            for n in range(self.num_nodes):
                for t in range(1, max_steps + 1):
                    # Incoming flow
                    incoming = gp.quicksum(
                        y[c,i,n,t-1] 
                        for i in range(self.num_nodes)
                        if (i,n) in self.topology.get_links() and (c,i,n,t-1) in y
                    )
                    
                    # Outgoing flow
                    outgoing = gp.quicksum(
                        y[c,n,j,t-1]
                        for j in range(self.num_nodes)
                        if (n,j) in self.topology.get_links() and (c,n,j,t-1) in y
                    )
                    
                    # Conservation
                    if (c, n) not in self.collective.precondition:
                        model.addConstr(
                            x[c,n,t] <= x[c,n,t-1] + incoming,
                            f"flow_in_{c}_{n}_{t}"
                        )
                    
                    model.addConstr(
                        outgoing <= x[c,n,t-1],
                        f"flow_out_{c}_{n}_{t}"
                    )
        
        # 5. Bandwidth relaxation: Just ensure total flow is reasonable
        for t in range(max_steps):
            total_flow = gp.quicksum(
                y[c,i,j,t]
                for c in range(self.num_chunks)
                for (i,j) in self.topology.get_links()
                if (c,i,j,t) in y
            )
            # Relaxed constraint - allow more flow than strict bandwidth
            model.addConstr(
                total_flow <= 2 * sum(self.topology.bandwidth.values()),
                f"relaxed_bw_{t}"
            )
        
        # Solve
        model.optimize()
        
        if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
            return None
        
        # Extract solution
        solution = {
            'routes': {},
            'schedule': {},
            'num_steps': 0
        }
        
        # Extract routes
        for c in range(self.num_chunks):
            solution['routes'][c] = []
            for t in range(max_steps):
                for (i, j) in self.topology.get_links():
                    if (c,i,j,t) in y and y[c,i,j,t].x > 0.5:
                        solution['routes'][c].append((t, i, j))
        
        # Extract schedule
        for c in range(self.num_chunks):
            for n in range(self.num_nodes):
                for t in range(max_steps + 1):
                    if x[c,n,t].x > 0.5:
                        solution['schedule'][(c,n)] = t
                        solution['num_steps'] = max(solution['num_steps'], t)
        
        return solution
    
    def _solve_scheduling_phase(self, routing_solution: Dict, 
                               time_limit: int) -> MILPSolution:
        """
        Phase 2: Bandwidth-constrained scheduling.
        
        Given routes from phase 1, find valid scheduling respecting bandwidth.
        """
        model = gp.Model("TACCL_Scheduling")
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 0.01
        
        if not self.verbose:
            model.Params.LogToConsole = 0
        
        num_steps = routing_solution['num_steps']
        routes = routing_solution['routes']
        
        # Variables
        # t[c,i,j]: time when chunk c is sent from i to j
        t = {}
        for c in range(self.num_chunks):
            for (step, i, j) in routes.get(c, []):
                t[c,i,j] = model.addVar(vtype=GRB.INTEGER, 
                                       lb=0, ub=num_steps,
                                       name=f"t_{c}_{i}_{j}")
        
        # r[s]: rounds at step s
        r = {}
        for s in range(num_steps):
            r[s] = model.addVar(vtype=GRB.INTEGER, lb=1, ub=10,
                               name=f"r_{s}")
        
        # Objective: Minimize total rounds (bandwidth cost)
        model.setObjective(
            gp.quicksum(r[s] for s in range(num_steps)),
            GRB.MINIMIZE
        )
        
        # Constraints
        
        # 1. Respect routing order
        for c in range(self.num_chunks):
            route = sorted(routes.get(c, []), key=lambda x: x[0])
            for k in range(len(route) - 1):
                _, i1, j1 = route[k]
                _, i2, j2 = route[k+1]
                if (c,i1,j1) in t and (c,i2,j2) in t:
                    model.addConstr(
                        t[c,i1,j1] < t[c,i2,j2],
                        f"order_{c}_{k}"
                    )
        
        # 2. Bandwidth constraints
        for s in range(num_steps):
            for (i, j) in self.topology.get_links():
                # Count sends on this link at this step
                sends = gp.quicksum(
                    1 for c in range(self.num_chunks)
                    if (c,i,j) in t
                    for _ in [1] if t[c,i,j] == s
                )
                
                bandwidth = self.topology.bandwidth[(i,j)]
                model.addConstr(
                    sends <= bandwidth * r[s],
                    f"bandwidth_{i}_{j}_{s}"
                )
        
        # 3. Ensure postconditions are met in time
        for (c, n) in self.collective.postcondition:
            # Find last send to node n for chunk c
            last_send_time = gp.LinExpr()
            for (chunk, src, dst) in t:
                if chunk == c and dst == n:
                    last_send_time = gp.max_(last_send_time, t[chunk,src,dst])
            
            if last_send_time.size() > 0:
                model.addConstr(
                    last_send_time <= num_steps,
                    f"deadline_{c}_{n}"
                )
        
        # Solve
        model.optimize()
        
        if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
            # Fallback to simple scheduling
            return self._fallback_scheduling(routing_solution)
        
        # Extract solution
        sends = []
        schedule = routing_solution['schedule'].copy()
        rounds = [1] * num_steps  # Default
        
        # Extract send times
        for (c, i, j) in t:
            if t[c,i,j].x >= 0:
                step = int(round(t[c,i,j].x))
                sends.append({
                    'chunk': c,
                    'src': i, 
                    'dst': j,
                    'step': step
                })
        
        # Extract rounds
        for s in range(num_steps):
            if s in r:
                rounds[s] = max(1, int(round(r[s].x)))
        
        return MILPSolution(
            sends=sends,
            rounds=rounds,
            schedule=schedule,
            objective_value=model.objVal if model.status == GRB.OPTIMAL else -1,
            solve_time=model.Runtime
        )
    
    def _is_link_allowed(self, src: int, dst: int) -> bool:
        """Check if a link is allowed by sketch constraints"""
        if not self.sketch_constraints:
            return True
        
        allowed = self.sketch_constraints['allowed_links']
        forbidden = self.sketch_constraints['forbidden_links']
        
        if forbidden and (src, dst) in forbidden:
            return False
        
        if allowed and (src, dst) not in allowed:
            return False
        
        return True
    
    def _fallback_scheduling(self, routing_solution: Dict) -> MILPSolution:
        """Simple fallback scheduling when optimization fails"""
        sends = []
        rounds = []
        
        # Convert routes to sends with simple scheduling
        for c, route in routing_solution['routes'].items():
            for step, (orig_step, src, dst) in enumerate(sorted(route)):
                sends.append({
                    'chunk': c,
                    'src': src,
                    'dst': dst, 
                    'step': step
                })
        
        # Determine rounds needed
        num_steps = routing_solution['num_steps']
        for s in range(num_steps):
            # Count sends at this step
            step_sends = {}
            for send in sends:
                if send['step'] == s:
                    link = (send['src'], send['dst'])
                    step_sends[link] = step_sends.get(link, 0) + 1
            
            # Max rounds needed
            max_rounds = 1
            for link, count in step_sends.items():
                bandwidth = self.topology.bandwidth[link]
                rounds_needed = (count + bandwidth - 1) // bandwidth
                max_rounds = max(max_rounds, rounds_needed)
            
            rounds.append(max_rounds)
        
        return MILPSolution(
            sends=sends,
            rounds=rounds,
            schedule=routing_solution['schedule'],
            objective_value=-1,
            solve_time=0
        )