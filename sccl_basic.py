# sccl_basic_fixed_v2.py
from z3 import *
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
from .topology import *
from .collective import *

class SCCLBasic:
    """Basic SCCL implementation"""
    
    def __init__(self, topology: Topology, collective: Collective, 
                 num_steps: int, num_rounds: int):
        self.topology = topology
        self.collective = collective
        self.num_steps = num_steps
        self.num_rounds = num_rounds
        self.solver = Solver()
        
        # SMT variables
        self.time_vars = {}  # time_{c,n}
        self.send_vars = {}  # snd_{n,c,n'}
        self.round_vars = []  # r_s
        
        self._create_variables()
        self._add_constraints()
    
    def _create_variables(self):
        """Create SMT variables"""
        # time_{c,n}: earliest step when chunk c is available at node n
        for c in range(self.collective.num_chunks):
            for n in range(self.topology.num_nodes):
                self.time_vars[(c, n)] = Int(f'time_{c}_{n}')
        
        # snd_{n,c,n'}: whether node n sends chunk c to n'
        for (src, dst) in self.topology.get_links():
            for c in range(self.collective.num_chunks):
                self.send_vars[(src, c, dst)] = Bool(f'snd_{src}_{c}_{dst}')
        
        # r_s: number of rounds at each step
        for s in range(self.num_steps):
            self.round_vars.append(Int(f'r_{s}'))
    
    def _add_constraints(self):
        """Add SMT constraints"""
        
        # C1: Precondition - nodes that initially have chunks
        for (c, n) in self.collective.precondition:
            self.solver.add(self.time_vars[(c, n)] == 0)
        
        # C2: Postcondition - nodes that must have chunks at the end
        for (c, n) in self.collective.postcondition:
            self.solver.add(self.time_vars[(c, n)] <= self.num_steps)
        
        # C3: To receive a chunk, must receive exactly once
        for c in range(self.collective.num_chunks):
            for n in range(self.topology.num_nodes):
                if (c, n) not in self.collective.precondition:
                    # Sum of all possible incoming sends equals 1
                    incoming_sends = []
                    for (src, dst) in self.topology.get_links():
                        if dst == n:
                            incoming_sends.append(
                                If(self.send_vars[(src, c, n)], 1, 0)
                            )
                    
                    if incoming_sends:
                        self.solver.add(
                            Implies(
                                self.time_vars[(c, n)] <= self.num_steps,
                                Sum(incoming_sends) == 1
                            )
                        )
        
        # C4: To send, must have it first
        for (src, dst) in self.topology.get_links():
            for c in range(self.collective.num_chunks):
                self.solver.add(
                    Implies(
                        self.send_vars[(src, c, dst)],
                        self.time_vars[(c, src)] < self.time_vars[(c, dst)]
                    )
                )
        
        # C5: Bandwidth constraint - limit sends per link at each step
        for s in range(1, self.num_steps + 1):
            for (src, dst) in self.topology.get_links():
                sends_at_step = []
                for c in range(self.collective.num_chunks):
                    sends_at_step.append(
                        If(And(
                            self.send_vars[(src, c, dst)],
                            self.time_vars[(c, dst)] == s
                        ), 1, 0)
                    )
                
                bandwidth = self.topology.bandwidth[(src, dst)]
                self.solver.add(
                    Sum(sends_at_step) <= bandwidth * self.round_vars[s-1]
                )
        
        # C6: Total rounds constraint
        self.solver.add(Sum(self.round_vars) == self.num_rounds)
        
        # Rounds must be non-negative
        for r in self.round_vars:
            self.solver.add(r >= 0)
    
    def _get_value(self, model, var):
        """Safely extract value from Z3 model"""
        val = model.evaluate(var, model_completion=True)
        if is_int_value(val):
            return val.as_long()
        elif is_bool(val):
            return is_true(val)
        else:
            # Try to simplify
            val = simplify(val)
            if is_int_value(val):
                return val.as_long()
            elif is_bool(val):
                return is_true(val)
            else:
                print(f"Warning: Could not extract value for {var}, got {val} of type {type(val)}")
                return None
    
    def solve(self) -> Optional[Dict]:
        """Solve SMT problem"""
        result = self.solver.check()
        
        if result == sat:
            model = self.solver.model()
            
            # Extract solution
            solution = {
                'rounds': [],
                'sends': [],
                'schedule': {},
                'status': 'sat'
            }
            
            # Round information
            for i, r in enumerate(self.round_vars):
                val = self._get_value(model, r)
                if val is not None:
                    solution['rounds'].append(val)
                else:
                    print(f"Warning: Could not get value for round {i}")
                    solution['rounds'].append(0)
            
            # Send information
            for (src, c, dst), var in self.send_vars.items():
                is_sent = self._get_value(model, var)
                if is_sent:
                    time_val = self._get_value(model, self.time_vars[(c, dst)])
                    if time_val is not None:
                        solution['sends'].append({
                            'chunk': c,
                            'src': src,
                            'dst': dst,
                            'step': time_val - 1
                        })
            
            # Schedule information
            for (c, n), var in self.time_vars.items():
                val = self._get_value(model, var)
                if val is not None:
                    solution['schedule'][(c, n)] = val
            
            return solution
        else:
            print(f"Solver result: {result}")
            if result == unsat:
                print(" Problem is unsatisfiable!")
                # Try to get unsat core
                self.solver.set(unsat_core=True)
            return None

def print_solution_details(solution: Dict):
    """Print detailed solution information"""
    if not solution:
        return
    
    print(f"\n=== Solution Details ===")
    print(f"Status: {solution.get('status', 'unknown')}")
    print(f"Rounds per step: {solution['rounds']}")
    print(f"Total sends: {len(solution['sends'])}")
    
    # Group sends by step
    sends_by_step = {}
    for send in solution['sends']:
        step = send['step']
        if step not in sends_by_step:
            sends_by_step[step] = []
        sends_by_step[step].append(send)
    
    # Print sends organized by step
    for step in sorted(sends_by_step.keys()):
        print(f"\nStep {step}:")
        for send in sorted(sends_by_step[step], key=lambda x: (x['src'], x['dst'])):
            print(f"  Node {send['src']} -> Node {send['dst']}: chunk {send['chunk']}")

# Test with simple example first
def test_simple_case():
    """Test with a simple 2-node case first"""
    print("=== Testing Simple 2-node Case ===")
    
    # Create 2-node topology
    bandwidth = {(0, 1): 1, (1, 0): 1}
    topology = Topology(num_nodes=2, bandwidth=bandwidth)
    
    # Create simple collective - node 0 has chunk 0, both need it
    collective = Collective.create_collective(
        name = CollectiveType.BROADCAST,
        num_nodes=2,
        chunks_per_node=1
    )
    
    # This should be solvable in 1 step, 1 round
    sccl = SCCLBasic(topology, collective, num_rounds=1, num_steps=1)
    solution = sccl.solve()
    
    if solution:
        print("Simple test PASSED!")
        print_solution_details(solution)
    else:
        print("Simple test FAILED!")
    
    return solution is not None

# Main test
if __name__ == "__main__":
    # Start with simple test
    if not test_simple_case():
        print("Simple test failed, debugging needed")
        exit(1)
    
    print("\n" + "="*50 + "\n")
    
    # Now test full 4-node ring
    print("=== Testing 4-node Ring All-Gather ===")
    topology = create_ring_topology(4)
    collective = Collective.create_collective(
        collective_type = CollectiveType.ALLGATHER, 
        num_nodes=4, 
        chunks_per_node=2)
    
    print(f"Topology: 4-node ring")
    print(f"Collective: {collective.name}")
    print(f"Total chunks: {collective.num_chunks}")
    print(f"Links: {topology.get_links()}")
    
    # Try with different configurations
    found_solution = False
    
    test_configs = [
        (3, 3),  # Known to work for ring allgather
        (3, 4),
        (4, 4),
        (4, 5),
        (4, 6),
    ]
    
    for steps, rounds in test_configs:
        print(f"\n--- Trying: {steps} steps, {rounds} rounds ---")
        
        sccl = SCCLBasic(topology, collective, steps, rounds)
        solution = sccl.solve()
        
        if solution:
            found_solution = True
            print("Solution found!")
            print_solution_details(solution)
            
            # Verify solution correctness
            print("\nVerifying solution...")
            all_good = True
            
            # Check all postconditions are satisfied
            for (c, n) in collective.postcondition:
                if (c, n) not in solution['schedule']:
                    print(f"  ERROR: Chunk {c} schedule missing for node {n}")
                    all_good = False
                elif solution['schedule'][(c, n)] > steps:
                    print(f"  ERROR: Chunk {c} arrives at node {n} too late")
                    all_good = False
            
            if all_good:
                print("  ✓ Solution verified!")
            
            break  # Found one solution, that's enough for now
    
    if not found_solution:
        print("\nNo solution found! This might indicate a bug in constraints.")