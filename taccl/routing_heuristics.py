# routing_heuristics.py
"""
Routing Heuristics for TACCL

This module provides various routing heuristics that can guide the synthesizer
towards better algorithms by suggesting efficient routing patterns.
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from ..topology import Topology
from ..collective import Collective, CollectiveType

@dataclass
class Route:
    """Represents a route for a chunk"""
    chunk_id: int
    path: List[int]  # Sequence of nodes
    hops: int
    latency: int  # Estimated latency

class RoutingHeuristic:
    """Base class for routing heuristics"""
    
    def __init__(self, topology: Topology, collective: Collective):
        self.topology = topology
        self.collective = collective
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from topology"""
        G = nx.DiGraph()
        for i in range(self.topology.num_nodes):
            G.add_node(i)
        
        for (src, dst), bandwidth in self.topology.bandwidth.items():
            # Weight inversely proportional to bandwidth for shortest path
            weight = 1.0 / bandwidth if bandwidth > 0 else float('inf')
            G.add_edge(src, dst, weight=weight, bandwidth=bandwidth)
        
        return G
    
    def compute_routes(self) -> Dict[int, List[Route]]:
        """Compute routes for all chunks. To be implemented by subclasses."""
        raise NotImplementedError

class ShortestPathRouting(RoutingHeuristic):
    """Simple shortest path routing"""
    
    def compute_routes(self) -> Dict[int, List[Route]]:
        """Compute shortest path routes for all chunks"""
        routes = {}
        
        for chunk_id in range(self.collective.num_chunks):
            chunk_routes = []
            
            # Find source nodes (where chunk initially resides)
            sources = [n for (c, n) in self.collective.precondition if c == chunk_id]
            
            # Find destination nodes (where chunk needs to go)
            destinations = [n for (c, n) in self.collective.postcondition if c == chunk_id]
            
            for src in sources:
                for dst in destinations:
                    if src != dst:
                        try:
                            path = nx.shortest_path(self.graph, src, dst, weight='weight')
                            route = Route(
                                chunk_id=chunk_id,
                                path=path,
                                hops=len(path) - 1,
                                latency=len(path) - 1
                            )
                            chunk_routes.append(route)
                        except nx.NetworkXNoPath:
                            continue
            
            routes[chunk_id] = chunk_routes
        
        return routes

class RingRouting(RoutingHeuristic):
    """Ring-based routing heuristic"""
    
    def __init__(self, topology: Topology, collective: Collective, ring_order: Optional[List[int]] = None):
        super().__init__(topology, collective)
        self.ring_order = ring_order or self._find_best_ring()
    
    def _find_best_ring(self) -> List[int]:
        """Find a good ring order in the topology"""
        # Try to find Hamiltonian cycle
        if self.topology.num_nodes <= 8:
            # For small topologies, try to find exact Hamiltonian cycle
            try:
                cycle = self._find_hamiltonian_cycle()
                if cycle:
                    return cycle
            except:
                pass
        
        # Fallback: use greedy nearest neighbor
        return self._greedy_ring_construction()
    
    def _find_hamiltonian_cycle(self) -> Optional[List[int]]:
        """Find Hamiltonian cycle if one exists"""
        import itertools
        
        nodes = list(range(self.topology.num_nodes))
        
        # Try all permutations (only feasible for small graphs)
        for perm in itertools.permutations(nodes[1:]):
            path = [nodes[0]] + list(perm)
            
            # Check if this forms a valid cycle
            valid = True
            for i in range(len(path)):
                src = path[i]
                dst = path[(i + 1) % len(path)]
                if not self.graph.has_edge(src, dst):
                    valid = False
                    break
            
            if valid:
                return path
        
        return None
    
    def _greedy_ring_construction(self) -> List[int]:
        """Construct ring greedily"""
        visited = set()
        ring = []
        
        # Start from node 0
        current = 0
        ring.append(current)
        visited.add(current)
        
        while len(visited) < self.topology.num_nodes:
            # Find nearest unvisited neighbor
            best_next = None
            best_weight = float('inf')
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    weight = self.graph[current][neighbor]['weight']
                    if weight < best_weight:
                        best_weight = weight
                        best_next = neighbor
            
            if best_next is not None:
                current = best_next
                ring.append(current)
                visited.add(current)
            else:
                # No unvisited neighbor, try from any unvisited node
                for node in range(self.topology.num_nodes):
                    if node not in visited:
                        current = node
                        ring.append(current)
                        visited.add(current)
                        break
        
        return ring
    
    def compute_routes(self) -> Dict[int, List[Route]]:
        """Compute ring-based routes"""
        routes = {}
        ring_size = len(self.ring_order)
        
        # Create position mapping
        pos_map = {node: i for i, node in enumerate(self.ring_order)}
        
        for chunk_id in range(self.collective.num_chunks):
            chunk_routes = []
            
            # Find source and destination positions
            sources = [n for (c, n) in self.collective.precondition if c == chunk_id]
            destinations = [n for (c, n) in self.collective.postcondition if c == chunk_id]
            
            for src in sources:
                for dst in destinations:
                    if src != dst and src in pos_map and dst in pos_map:
                        src_pos = pos_map[src]
                        dst_pos = pos_map[dst]
                        
                        # Compute clockwise and counter-clockwise distances
                        cw_dist = (dst_pos - src_pos) % ring_size
                        ccw_dist = (src_pos - dst_pos) % ring_size
                        
                        # Choose shorter path
                        if cw_dist <= ccw_dist:
                            # Clockwise path
                            path = []
                            for i in range(cw_dist + 1):
                                pos = (src_pos + i) % ring_size
                                path.append(self.ring_order[pos])
                        else:
                            # Counter-clockwise path
                            path = []
                            for i in range(ccw_dist + 1):
                                pos = (src_pos - i) % ring_size
                                path.append(self.ring_order[pos])
                        
                        route = Route(
                            chunk_id=chunk_id,
                            path=path,
                            hops=len(path) - 1,
                            latency=len(path) - 1
                        )
                        chunk_routes.append(route)
            
            routes[chunk_id] = chunk_routes
        
        return routes

class HierarchicalRouting(RoutingHeuristic):
    """Hierarchical routing for multi-level topologies"""
    
    def __init__(self, topology: Topology, collective: Collective, 
                 node_groups: List[List[int]]):
        super().__init__(topology, collective)
        self.node_groups = node_groups
        self.group_map = self._build_group_map()
    
    def _build_group_map(self) -> Dict[int, int]:
        """Map each node to its group"""
        group_map = {}
        for group_id, nodes in enumerate(self.node_groups):
            for node in nodes:
                group_map[node] = group_id
        return group_map
    
    def compute_routes(self) -> Dict[int, List[Route]]:
        """Compute hierarchical routes"""
        routes = {}
        
        for chunk_id in range(self.collective.num_chunks):
            chunk_routes = []
            
            sources = [n for (c, n) in self.collective.precondition if c == chunk_id]
            destinations = [n for (c, n) in self.collective.postcondition if c == chunk_id]
            
            for src in sources:
                for dst in destinations:
                    if src != dst:
                        route = self._compute_hierarchical_route(src, dst)
                        if route:
                            route.chunk_id = chunk_id
                            chunk_routes.append(route)
            
            routes[chunk_id] = chunk_routes
        
        return routes
    
    def _compute_hierarchical_route(self, src: int, dst: int) -> Optional[Route]:
        """Compute route using hierarchical approach"""
        src_group = self.group_map.get(src)
        dst_group = self.group_map.get(dst)
        
        if src_group is None or dst_group is None:
            return None
        
        path = [src]
        
        if src_group == dst_group:
            # Intra-group routing
            try:
                intra_path = nx.shortest_path(self.graph, src, dst, weight='weight')
                path = intra_path
            except nx.NetworkXNoPath:
                return None
        else:
            # Inter-group routing
            # First, route to gateway node in source group
            src_gateway = self._find_gateway(src_group, dst_group)
            if src_gateway and src_gateway != src:
                try:
                    path_to_gateway = nx.shortest_path(self.graph, src, src_gateway, weight='weight')
                    path.extend(path_to_gateway[1:])
                except nx.NetworkXNoPath:
                    return None
            
            # Route to gateway node in destination group
            dst_gateway = self._find_gateway(dst_group, src_group)
            if dst_gateway and src_gateway:
                try:
                    inter_path = nx.shortest_path(self.graph, src_gateway, dst_gateway, weight='weight')
                    path.extend(inter_path[1:])
                except nx.NetworkXNoPath:
                    return None
            
            # Finally, route from gateway to destination
            if dst_gateway != dst:
                try:
                    path_from_gateway = nx.shortest_path(self.graph, dst_gateway, dst, weight='weight')
                    path.extend(path_from_gateway[1:])
                except nx.NetworkXNoPath:
                    return None
        
        return Route(
            chunk_id=-1,  # Will be set by caller
            path=path,
            hops=len(path) - 1,
            latency=len(path) - 1
        )
    
    def _find_gateway(self, from_group: int, to_group: int) -> Optional[int]:
        """Find gateway node that connects from_group to to_group"""
        from_nodes = self.node_groups[from_group]
        to_nodes = self.node_groups[to_group]
        
        # Find node in from_group with most connections to to_group
        best_gateway = None
        max_connections = 0
        
        for node in from_nodes:
            connections = 0
            for neighbor in self.graph.neighbors(node):
                if neighbor in to_nodes:
                    connections += 1
            
            if connections > max_connections:
                max_connections = connections
                best_gateway = node
        
        return best_gateway

def select_routing_heuristic(topology: Topology, collective: Collective,
                           collective_type: CollectiveType) -> RoutingHeuristic:
    """
    Select appropriate routing heuristic based on topology and collective type.
    """
    # For small topologies or specific patterns, use ring routing
    if topology.num_nodes <= 8 and collective_type in [CollectiveType.ALLGATHER, 
                                                       CollectiveType.ALLREDUCE]:
        return RingRouting(topology, collective)
    
    # For hierarchical topologies (e.g., multi-node with clear groups)
    # This is a simple heuristic to detect groups
    groups = detect_node_groups(topology)
    if len(groups) > 1:
        return HierarchicalRouting(topology, collective, groups)
    
    # Default: shortest path
    return ShortestPathRouting(topology, collective)

def detect_node_groups(topology: Topology) -> List[List[int]]:
    """
    Simple heuristic to detect node groups in topology.
    Groups are detected based on connectivity patterns.
    """
    G = nx.Graph()
    for i in range(topology.num_nodes):
        G.add_node(i)
    
    # Add edges with weights based on bandwidth
    for (src, dst), bandwidth in topology.bandwidth.items():
        if G.has_edge(src, dst):
            G[src][dst]['weight'] = min(G[src][dst]['weight'], 1.0/bandwidth)
        else:
            G.add_edge(src, dst, weight=1.0/bandwidth)
    
    # Use community detection or simple clustering
    # For now, use a simple approach based on min-cut
    if topology.num_nodes <= 4:
        return [list(range(topology.num_nodes))]
    
    # Try to partition into two groups
    partition = nx.minimum_cut(G, 0, topology.num_nodes - 1)[1]
    group1 = list(partition[0])
    group2 = list(partition[1])
    
    # If groups are very unbalanced, treat as single group
    if len(group1) < topology.num_nodes // 4 or len(group2) < topology.num_nodes // 4:
        return [list(range(topology.num_nodes))]
    
    return [group1, group2]