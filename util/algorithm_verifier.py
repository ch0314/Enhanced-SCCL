# algorithm_verifier.py
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from ..topology import Topology
from ..collective import Collective, CollectiveType

@dataclass
class VerificationResult:
    """Result of algorithm verification"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, any]

class AlgorithmVerifier:
    """Verify correctness of synthesized collective algorithms"""
    
    def __init__(self, topology: Topology, collective: Collective):
        self.topology = topology
        self.collective = collective
        
    def verify_algorithm(self, solution: Dict) -> VerificationResult:
        """
        Comprehensive verification of a synthesized algorithm
        
        Args:
            solution: Dictionary containing 'sends', 'rounds', 'schedule'
        
        Returns:
            VerificationResult with validation status and details
        """
        errors = []
        warnings = []
        statistics = {}
        
        # 1. Verify preconditions
        if not self._verify_preconditions(solution, errors):
            return VerificationResult(False, errors, warnings, statistics)
            
        # # 2. Verify postconditions
        if not self._verify_postconditions(solution, errors):
            return VerificationResult(False, errors, warnings, statistics)
            
        # 3. Verify send validity
        if not self._verify_sends(solution, errors, warnings):
            return VerificationResult(False, errors, warnings, statistics)
            
        # 4. Verify bandwidth constraints
        if not self._verify_bandwidth_constraints(solution, errors, warnings):
            return VerificationResult(False, errors, warnings, statistics)
            
        # 5. Verify chunk routing correctness
        if not self._verify_chunk_routing(solution, errors, warnings):
            return VerificationResult(False, errors, warnings, statistics)
            
        # 6. Compute statistics
        statistics = self._compute_statistics(solution)
        
        return VerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )
    
    def _verify_preconditions(self, solution: Dict, errors: List[str]) -> bool:
        """Verify all precondition chunks start at time 0"""
        time_entries = solution.get('time', [])
        
        # Build a mapping from (chunk, node) to step for easier lookup
        time_map = {}
        for entry in time_entries:
            chunk = entry['chunk']
            node = entry['node']
            step = entry['step']
            time_map[(chunk, node)] = step
        
        for (chunk, node) in self.collective.precondition:
            if (chunk, node) not in time_map:
                errors.append(f"Precondition chunk {chunk} at node {node} missing from time schedule")
                return False
            if time_map[(chunk, node)] != 0:
                errors.append(f"Precondition chunk {chunk} at node {node} should start at time 0, "
                            f"but starts at {time_map[(chunk, node)]}")
                return False
        return True
    
    def _verify_postconditions(self, solution: Dict, errors: List[str]) -> bool:
        """Verify all postcondition chunks are delivered"""
        time_entries = solution.get('time', [])
        num_steps = len(solution.get('rounds', []))
        
        # Build a mapping from (chunk, node) to step for easier lookup
        time_map = {}
        for entry in time_entries:
            chunk = entry['chunk']
            node = entry['node']
            step = entry['step']
            time_map[(chunk, node)] = step
        
        for (chunk, node) in self.collective.postcondition:
            if (chunk, node) not in time_map:
                errors.append(f"Postcondition chunk {chunk} missing at node {node}")
                return False
            if time_map[(chunk, node)] > num_steps:
                errors.append(f"Chunk {chunk} arrives at node {node} after step {num_steps} "
                            f"(arrives at {time_map[(chunk, node)]})")
                return False
        return True
    
    def _verify_sends(self, solution: Dict, errors: List[str], 
                     warnings: List[str]) -> bool:
        """Verify all sends are valid"""
        sends = solution.get('sends', [])
        
        for send in sends:
            chunk = send['chunk']
            src = send['src']
            dst = send['dst']
            step = send['step']
            
            # Check if link exists
            if (src, dst) not in self.topology.bandwidth:
                errors.append(f"Send uses non-existent link ({src}, {dst})")
                return False

                
        return True
    
    def _verify_bandwidth_constraints(self, solution: Dict, errors: List[str], 
                                    warnings: List[str]) -> bool:
        """Verify bandwidth constraints are satisfied"""

        # Get bandwidth constraints
        constraints = self.topology.get_bandwidth_constraints_for_step()

        sends = solution.get('sends', [])
        rounds = solution.get('rounds', [])
        num_steps = len(rounds)

        
        # Group sends by link and step
        sends_by_link_step = {}
        for send in sends:
            key = (send['step'])
            if key not in sends_by_link_step:
                sends_by_link_step[key] = []
            sends_by_link_step[key].append(send)

        # print(sends_by_link_step)
        # exit(0)

        # Track violations for better error reporting
        violations = []
        
        for step in range(num_steps):
            step_sends = sends_by_link_step.get(step, [])
            rounds_at_step = solution['rounds'][step]
            
            # For each bandwidth constraint set
            for constraint_idx, (link_set, bandwidth) in enumerate(constraints):
                # Count sends in this link set at this step
                sends_in_constraint = 0
                affected_links = []
                
                for send in step_sends:
                    src, dst = send['src'], send['dst']
                    if (src, dst) in link_set:
                        sends_in_constraint += 1
                        affected_links.append((src, dst))
                
                # Check constraint: sends_in_constraint <= bandwidth * rounds_at_step
                max_allowed = bandwidth * rounds_at_step
                if sends_in_constraint > max_allowed:
                    violation_info = {
                        'step': step,
                        'constraint_idx': constraint_idx,
                        'sends': sends_in_constraint,
                        'max_allowed': max_allowed,
                        'bandwidth': bandwidth,
                        'rounds': rounds_at_step,
                        'affected_links': affected_links
                    }
                    violations.append(violation_info)
        
        # Report all violations
        if violations:
            print(f"Found {len(violations)} bandwidth constraint violations:")
            for v in violations:
                print(
                    f"  Step {v['step']}: {v['sends']} sends exceed limit of {v['max_allowed']} "
                    f"(bandwidth={v['bandwidth']}, rounds={v['rounds']}) on links: {v['affected_links']}"
                )
            return False
                
        return True
    
    def _verify_chunk_routing(self, solution: Dict, errors: List[str], 
                            warnings: List[str]) -> bool:
        """Verify chunks follow valid routes"""
        sends = solution.get('sends', [])
        
        # Build chunk routes
        chunk_routes = {}  # chunk -> list of (step, src, dst)
        for send in sends:
            chunk = send['chunk']
            if chunk not in chunk_routes:
                chunk_routes[chunk] = []
            chunk_routes[chunk].append((send['step'], send['src'], send['dst']))
        
        # Verify each chunk's route
        for chunk, routes in chunk_routes.items():
            # Sort by step
            routes.sort(key=lambda x: x[0])
            
            # Find initial location
            initial_nodes = {n for (c, n) in self.collective.precondition if c == chunk}
            if not initial_nodes:
                warnings.append(f"Chunk {chunk} has no initial location")
                continue
            
            # Trace the route
            current_locations = initial_nodes.copy()
            
            for step, src, dst in routes:
                if src not in current_locations:
                    errors.append(f"Chunk {chunk}: node {src} sends at step {step} "
                                f"but doesn't have the chunk")
                    return False
                current_locations.add(dst)
            
            # Verify final locations match postconditions
            required_locations = {n for (c, n) in self.collective.postcondition if c == chunk}
            if not required_locations.issubset(current_locations):
                missing = required_locations - current_locations
                errors.append(f"Chunk {chunk} missing from required nodes: {missing}")
                return False
                
        return True
    
    def _compute_statistics(self, solution: Dict) -> Dict[str, any]:
        """Compute algorithm statistics"""
        sends = solution.get('sends', [])
        rounds = solution.get('rounds', [])
        schedule = solution.get('schedule', {})
        
        # Link utilization
        link_usage = {}
        for send in sends:
            link = (send['src'], send['dst'])
            link_usage[link] = link_usage.get(link, 0) + 1
        
        # Step utilization
        sends_per_step = {}
        for send in sends:
            step = send['step']
            sends_per_step[step] = sends_per_step.get(step, 0) + 1
        
        # Completion times
        completion_times = {}
        for (chunk, node), time in schedule.items():
            if (chunk, node) in self.collective.postcondition:
                completion_times[(chunk, node)] = time
        
        return {
            'total_sends': len(sends),
            'total_steps': len(rounds),
            'total_rounds': sum(rounds),
            'link_utilization': link_usage,
            'sends_per_step': sends_per_step,
            'average_link_usage': sum(link_usage.values()) / len(link_usage) if link_usage else 0,
            'max_completion_time': max(completion_times.values()) if completion_times else 0,
            'bandwidth_efficiency': self._compute_bandwidth_efficiency(solution)
        }
    
    def _compute_bandwidth_efficiency(self, solution: Dict) -> float:
        """Compute how efficiently bandwidth is utilized"""
        sends = solution.get('sends', [])
        rounds = solution.get('rounds', [])
        
        # Total available bandwidth across all steps
        total_available = 0
        for step, rounds_at_step in enumerate(rounds):
            for link, bandwidth in self.topology.bandwidth.items():
                total_available += bandwidth * rounds_at_step
        
        # Total used bandwidth
        total_used = len(sends)
        
        return total_used / total_available if total_available > 0 else 0


def visualize_algorithm_execution(solution: Dict, topology: Topology, 
                                collective: Collective):
    """Visualize the execution of an algorithm step by step"""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    sends = solution.get('sends', [])
    num_steps = len(solution.get('rounds', []))
    
    # Group sends by step
    sends_by_step = {}
    for send in sends:
        step = send['step']
        if step not in sends_by_step:
            sends_by_step[step] = []
        sends_by_step[step].append(send)
    
    # Create network graph
    G = nx.DiGraph()
    for i in range(topology.num_nodes):
        G.add_node(i)
    for (src, dst) in topology.get_links():
        G.add_edge(src, dst)
    
    # Layout
    pos = nx.circular_layout(G)
    
    # Create subplots for each step
    fig, axes = plt.subplots(1, num_steps, figsize=(4*num_steps, 4))
    if num_steps == 1:
        axes = [axes]
    
    for step in range(num_steps):
        ax = axes[step]
        ax.set_title(f'Step {step}')
        
        # Draw base network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                             node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                             alpha=0.3, arrows=True)
        
        # Highlight active sends
        if step in sends_by_step:
            active_edges = [(s['src'], s['dst']) for s in sends_by_step[step]]
            nx.draw_networkx_edges(G, pos, edgelist=active_edges, ax=ax,
                                 edge_color='red', width=3, arrows=True)
            
            # Label with chunk numbers
            edge_labels = {}
            for send in sends_by_step[step]:
                edge = (send['src'], send['dst'])
                if edge not in edge_labels:
                    edge_labels[edge] = []
                edge_labels[edge].append(str(send['chunk']))
            
            edge_labels = {e: ','.join(chunks) for e, chunks in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()



def verify_synthesized_algorithms(algorithms: List[Dict], 
                                topology: Topology,
                                collective_type: CollectiveType) -> Dict[int, VerificationResult]:
    """
    Verify all algorithms synthesized by ParetoSynthesizer
    
    Args:
        algorithms: List of algorithms from ParetoSynthesizer
        topology: The topology used for synthesis
        collective_type: The collective type
        
    Returns:
        Dictionary mapping algorithm index to verification result
    """
    verification_results = {}
    
    for idx, algorithm in enumerate(algorithms):
        # Extract the solution from the algorithm
        solution = algorithm.get('solution', {})
        
        # Create collective for this specific algorithm
        chunks_per_node = algorithm.get('chunks_per_node', 1)
        collective = Collective.create_collective(
            collective_type, 
            topology.num_nodes, 
            chunks_per_node
        )
        
        # Create verifier and verify
        verifier = AlgorithmVerifier(topology, collective)
        result = verifier.verify_algorithm(solution)
        
        verification_results[idx] = result
        
    return verification_results

def print_verification_summary(algorithms: List[Dict], 
                             verification_results: Dict[int, VerificationResult]):
    """Print a summary of verification results"""
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total_valid = sum(1 for r in verification_results.values() if r.is_valid)
    print(f"Total algorithms: {len(algorithms)}")
    print(f"Valid algorithms: {total_valid}")
    print(f"Invalid algorithms: {len(algorithms) - total_valid}")
    
    for idx, algorithm in enumerate(algorithms):
        result = verification_results[idx]
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        
        print(f"\n--- Algorithm {idx + 1} ---")
        print(f"Configuration: S={algorithm['steps']}, R={algorithm['rounds']}, "
              f"C={algorithm['chunks_per_node']}")
        print(f"Cost: Latency={algorithm['latency_cost']}, "
              f"Bandwidth={algorithm['bandwidth_cost']:.3f}")
        print(f"Status: {status}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  • {error}")
            if len(result.errors) > 3:
                print(f"  ... and {len(result.errors) - 3} more errors")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings[:2]:
                print(f"  ⚠ {warning}")
                
        if result.is_valid and result.statistics:
            print("Statistics:")
            print(f"  • Total sends: {result.statistics.get('total_sends', 0)}")
            print(f"  • Bandwidth efficiency: "
                  f"{result.statistics.get('bandwidth_efficiency', 0):.2%}")