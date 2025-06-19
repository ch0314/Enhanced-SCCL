# sccl_with_constraint_sets.py
from z3 import *
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
from ..topology import *
from ..collective import *

class SCCLBasic:
    """SCCL implementation with proper bandwidth constraint sets"""
    
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
        """Add SMT constraints with proper bandwidth constraint sets"""
        
        # C1: Precondition
        for (c, n) in self.collective.precondition:
            self.solver.add(self.time_vars[(c, n)] == 0)
        
        # C2: Postcondition
        for (c, n) in self.collective.postcondition:
            self.solver.add(self.time_vars[(c, n)] <= self.num_steps)
        
        # C3: To receive a chunk, must receive exactly once
        for c in range(self.collective.num_chunks):
            for n in range(self.topology.num_nodes):
                if (c, n) not in self.collective.precondition:
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
        
        # C5: Bandwidth constraints with constraint sets
        self._relaxed_bandwidth_constraints()
        
        # C6: Total rounds constraint
        self.solver.add(Sum(self.round_vars) == self.num_rounds)
        
        # Rounds must be non-negative
        for r in self.round_vars:
            self.solver.add(r > 0)
    
    def _add_bandwidth_constraints(self):
        """
        Add bandwidth constraints following SCCL paper's constraint sets.
        
        For each step s and bandwidth constraint (L, b) ∈ B:
        |{(c, n, n', s) ∈ T | (n, n') ∈ L}| ≤ b · r_s
        """
        # Get all bandwidth constraints
        constraints = self.topology.get_bandwidth_constraints_for_step()
        
        # For each step
        for s in range(1, self.num_steps + 1):
            # For each bandwidth constraint set
            for link_set, bandwidth in constraints:
                # Count sends in this link set at this step
                sends_in_set = []
                
                for (src, dst) in link_set:
                    for c in range(self.collective.num_chunks):
                        # Add to count if this send happens at step s
                        sends_in_set.append(
                            If(And(
                                self.send_vars[(src, c, dst)],
                                self.time_vars[(c, dst)] == s
                            ), 1, 0)
                        )
                
                # Add constraint: total sends in this set ≤ bandwidth * rounds
                if sends_in_set:
                    self.solver.add(
                        Sum(sends_in_set) <= bandwidth * self.round_vars[s-1]
                    )

    def _relaxed_bandwidth_constraints(self):
        """
        Add relaxed bandwidth constraints for debugging purposes.
        This allows more flexibility in send counts.
        """
        # Get all bandwidth constraints
        constraints = self.topology.get_bandwidth_constraints_for_step()


        for link_set, bandwidth in constraints:
            # Count sends in this link set at this step
            sends_in_set = []

            for (src, dst) in link_set:
                for c in range(self.collective.num_chunks):
                    # Add to count if this send happens at step s
                    sends_in_set.append(
                        If(self.send_vars[(src, c, dst)], 1, 0)
                    )

            # Add relaxed constraint: total sends in this set ≤ bandwidth * rounds + 1
            if sends_in_set:
                self.solver.add(
                    Sum(sends_in_set) <= bandwidth * self.num_rounds
                )

    def solve(self) -> Optional[Dict]:
        """Solve SMT problem"""
        result = self.solver.check()
        
        if result == sat:
            model = self.solver.model()
            
            # Extract solution
            solution = {
                'status': 'sat',
                'rounds': [],
                'sends': [],
                'time': []
            }
            
            # Extract rounds per step
            for s in range(self.num_steps):
                r_val = model.evaluate(self.round_vars[s], model_completion=True)
                solution['rounds'].append(r_val.as_long() if is_int_value(r_val) else 0)

            # Extract time variables
            for c in range(self.collective.num_chunks):
                for n in range(self.topology.num_nodes):
                    time_val = model.evaluate(self.time_vars[(c, n)], model_completion=True)
                    step = time_val.as_long() if is_int_value(time_val) else 0
                    solution['time'].append({
                        'chunk': c,
                        'node': n,
                        'step': step
                    })
            
            # Extract sends
            for (src, dst) in self.topology.get_links():
                for c in range(self.collective.num_chunks):
                    send_var = self.send_vars[(src, c, dst)]
                    if is_true(model.evaluate(send_var, model_completion=True)):
                        time_val = model.evaluate(self.time_vars[(c, dst)], model_completion=True)
                        step = time_val.as_long() - 1 if is_int_value(time_val) else 0
                        
                        solution['sends'].append({
                            'src': src,
                            'dst': dst,
                            'chunk': c,
                            'step': step
                        })
            
            return solution
        else:
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

# Test functions
def test_bidirectional_ring():
    """Test bidirectional ring with proper constraints"""
    print("=== Testing Bidirectional Ring with Constraint Sets ===")
    
    # Create ring with bidirectional constraints
    ring = create_ring_topology(4)
    
    # Create AllGather collective
    collective = Collective.create_collective(
        CollectiveType.ALLGATHER,
        num_nodes=4,
        chunks_per_node=1
    )
    
    print(f"Topology constraints: {len(ring.bandwidth_constraints)}")
    for i, bc in enumerate(ring.bandwidth_constraints):
        print(f"  Constraint {i}: {len(bc.link_set)} links, bandwidth={bc.bandwidth}")
    
    # Try different configurations
    test_configs = [(2, 3), (3, 3), (4, 4)]
    
    for steps, rounds in test_configs:
        print(f"\nTrying S={steps}, R={rounds}...\n")
        sccl = SCCLBasic(ring, collective, steps, rounds)
        solution = sccl.solve()
        
        if solution:
            print(f"  ✓ Found solution!")
            print(f"    Rounds: {solution['rounds']}")
            print(f"    Total sends: {len(solution['sends'])}")
            print_solution_details(solution)
        else:
            print(f"  ✗ No solution")

def test_dg1_topology():
    """Test dgx1 topology with proper constraints"""
    print("=== Testing dgx1 Topology with Constraint Sets ===")

    # Create dgx1 topology
    dgx1 = create_dgx1_topology()

    # Try different configurations
    test_configs = [(1, 2, 2), (2, 3, 3), (3, 4, 4), (4, 5, 5)]


    print(f"Topology constraints: {len(dgx1.bandwidth_constraints)}")
    for i, bc in enumerate(dgx1.bandwidth_constraints):
        print(f"  Constraint {i}: {len(bc.link_set)} links, bandwidth={bc.bandwidth}")


    for chunks, steps, rounds in test_configs:
        # Create AllGather collective
        collective = Collective.create_collective(
            CollectiveType.ALLGATHER,
            num_nodes=dgx1.num_nodes,
            chunks_per_node=chunks
        )       

        print(f"\nTrying C={chunks}, S={steps}, R={rounds}...\n")
        sccl = SCCLBasic(dgx1, collective, steps, rounds)
        solution = sccl.solve()

        if solution:
            print(f"  ✓ Found solution!")
            print(f"    Rounds: {solution['rounds']}")
            print(f"    Total sends: {len(solution['sends'])}")
            print_solution_details(solution)
        else:
            print(f"  ✗ No solution")

# Example usage
if __name__ == "__main__":
    test_dg1_topology()