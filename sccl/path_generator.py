# sccl/path_generator.py 

import networkx as nx
from typing import List, Dict, Tuple
from ..topology import *
from ..collective import *

def generate_shortest_paths(topology: Topology, collective: Collective) -> List[Dict]:
    """Generate shortest paths for all chunk transfers"""
    paths = []
    
    # Create NetworkX graph from topology
    G = nx.DiGraph()
    for (src, dst) in topology.get_links():
        G.add_edge(src, dst)
    
    # For each chunk that needs to be transferred
    for (chunk_id, src_node) in collective.precondition:
        for (chunk_id_dst, dst_node) in collective.postcondition:
            if chunk_id == chunk_id_dst and src_node != dst_node:
                try:
                    # Find shortest path
                    node_path = nx.shortest_path(G, src_node, dst_node)
                    edges = [(node_path[i], node_path[i+1]) 
                            for i in range(len(node_path)-1)]
                    
                    paths.append({
                        'chunk': chunk_id,
                        'source': src_node,
                        'destination': dst_node,
                        'edges': edges
                    })
                except nx.NetworkXNoPath:
                    print(f"No path from {src_node} to {dst_node}")
    
    return paths

def generate_ring_paths(topology: Topology, collective: Collective) -> List[Dict]:
    """Generate paths for ring topology"""
    num_nodes = topology.num_nodes
    paths = []
    
    # Create ring order (0 -> 1 -> 2 -> ... -> N-1 -> 0)
    ring_order = list(range(num_nodes))
    
    # Position mapping
    pos_map = {node: i for i, node in enumerate(ring_order)}
    
    # For each chunk transfer
    for (chunk_id, src_node) in collective.precondition:
        for (chunk_id_dst, dst_node) in collective.postcondition:
            if chunk_id == chunk_id_dst and src_node != dst_node:
                src_pos = pos_map[src_node]
                dst_pos = pos_map[dst_node]
                
                # Calculate distances
                cw_dist = (dst_pos - src_pos) % num_nodes  # Clockwise
                ccw_dist = (src_pos - dst_pos) % num_nodes  # Counter-clockwise
                
                # Choose shorter path
                if cw_dist <= ccw_dist:
                    # Clockwise path
                    path_nodes = []
                    for i in range(cw_dist + 1):
                        pos = (src_pos + i) % num_nodes
                        path_nodes.append(ring_order[pos])
                else:
                    # Counter-clockwise path  
                    path_nodes = []
                    for i in range(ccw_dist + 1):
                        pos = (src_pos - i) % num_nodes
                        path_nodes.append(ring_order[pos])
                
                # Convert to edges
                edges = [(path_nodes[i], path_nodes[i+1]) 
                        for i in range(len(path_nodes)-1)]
                
                paths.append({
                    'chunk': chunk_id,
                    'source': src_node,
                    'destination': dst_node,
                    'nodes': path_nodes,
                    'edges': edges,
                    'direction': 'CW' if cw_dist <= ccw_dist else 'CCW'
                })
    
    return paths


def generate_ring_allgather_paths(num_nodes: int, num_chunks: int) -> List[Dict]:
    """
    Generate paths for AllGather on ring topology.
    Each node sends its chunk around the ring.
    """
    paths = []
    
    # Each chunk originates from one node
    for chunk_id in range(num_chunks):
        src_node = chunk_id % num_nodes
        
        # Send to all other nodes in ring order
        for hop in range(1, num_nodes):
            dst_node = (src_node + hop) % num_nodes
            
            # Always go clockwise for AllGather
            path_nodes = [(src_node + i) % num_nodes for i in range(hop + 1)]
            edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
            
            paths.append({
                'chunk': chunk_id,
                'source': src_node,
                'destination': dst_node,
                'nodes': path_nodes,
                'edges': edges,
                'hop_count': hop
            })
    
    return paths

def generate_optimized_ring_paths(topology: Topology, collective: Collective) -> List[Dict]:
    """Generate ring paths optimized for merging"""
    num_nodes = topology.num_nodes
    paths = []
    
    # Group transfers by direction for better merging
    cw_transfers = []  # Clockwise transfers
    ccw_transfers = [] # Counter-clockwise transfers
    
    for (chunk_id, src_node) in collective.precondition:
        for (chunk_id_dst, dst_node) in collective.postcondition:
            if chunk_id == chunk_id_dst and src_node != dst_node:
                # Calculate distances
                cw_dist = (dst_node - src_node) % num_nodes
                ccw_dist = (src_node - dst_node) % num_nodes
                
                if cw_dist <= ccw_dist:
                    cw_transfers.append((chunk_id, src_node, dst_node, cw_dist))
                else:
                    ccw_transfers.append((chunk_id, src_node, dst_node, ccw_dist))
    
    # Generate clockwise paths
    for chunk_id, src, dst, dist in cw_transfers:
        path_nodes = [(src + i) % num_nodes for i in range(dist + 1)]
        edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
        
        paths.append({
            'chunk': chunk_id,
            'source': src,
            'destination': dst,
            'nodes': path_nodes,
            'edges': edges,
            'direction': 'CW',
            'distance': dist
        })
    
    # Generate counter-clockwise paths
    for chunk_id, src, dst, dist in ccw_transfers:
        path_nodes = [(src - i) % num_nodes for i in range(dist + 1)]
        edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
        
        paths.append({
            'chunk': chunk_id,
            'source': src,
            'destination': dst,
            'nodes': path_nodes,
            'edges': edges,
            'direction': 'CCW',
            'distance': dist
        })
    
    return paths