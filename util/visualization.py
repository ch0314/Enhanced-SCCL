# visualization.py
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def visualize_pareto_frontier(algorithms: List[Dict], 
                            save_path: Optional[str] = None,
                            title: Optional[str] = None,
                            show_plot: bool = True,
                            annotate_all: bool = True,
                            highlight_optimal: bool = True) -> Optional[str]:
    """
    Visualize the Pareto frontier of synthesized algorithms
    
    Args:
        algorithms: List of algorithms from ParetoSynthesizer
        save_path: Path to save the plot (if None, auto-generates)
        title: Custom title for the plot
        show_plot: Whether to display the plot
        annotate_all: Whether to annotate all points or just Pareto-optimal ones
        highlight_optimal: Whether to highlight latency/bandwidth optimal points
        
    Returns:
        Path where the plot was saved (if saved)
    """
    if not algorithms:
        print("No algorithms to visualize")
        return None
    
    # Extract costs
    latencies = [alg['latency_cost'] for alg in algorithms]
    bandwidths = [alg['bandwidth_cost'] for alg in algorithms]
    
    # Find Pareto-optimal points
    pareto_indices = _find_pareto_optimal_indices(algorithms)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot all algorithms
    for i, alg in enumerate(algorithms):
        if i in pareto_indices:
            # Pareto-optimal points
            plt.scatter(alg['latency_cost'], alg['bandwidth_cost'], 
                       s=150, alpha=0.9, c='red', 
                       edgecolors='darkred', linewidth=2,
                       marker='*', label='Pareto-optimal' if i == pareto_indices[0] else "")
        else:
            # Non-Pareto points
            plt.scatter(alg['latency_cost'], alg['bandwidth_cost'], 
                       s=100, alpha=0.6, c='lightblue', 
                       edgecolors='blue', linewidth=1,
                       marker='o', label='Non-Pareto' if i == 0 else "")
    
    # Highlight special points if requested
    if highlight_optimal:
        # Find latency-optimal
        min_latency_idx = min(range(len(algorithms)), 
                            key=lambda i: algorithms[i]['latency_cost'])
        # Find bandwidth-optimal  
        min_bandwidth_idx = min(range(len(algorithms)), 
                              key=lambda i: algorithms[i]['bandwidth_cost'])
        
        # Highlight with special markers
        if min_latency_idx in pareto_indices:
            alg = algorithms[min_latency_idx]
            plt.scatter(alg['latency_cost'], alg['bandwidth_cost'], 
                       s=200, c='green', marker='D', 
                       edgecolors='darkgreen', linewidth=3,
                       label='Latency-optimal', zorder=5)
        
        if min_bandwidth_idx in pareto_indices:
            alg = algorithms[min_bandwidth_idx]
            plt.scatter(alg['latency_cost'], alg['bandwidth_cost'], 
                       s=200, c='orange', marker='s', 
                       edgecolors='darkorange', linewidth=3,
                       label='Bandwidth-optimal', zorder=5)
    
    # Annotate points
    for i, alg in enumerate(algorithms):
        if annotate_all or i in pareto_indices:
            # Smart annotation positioning to avoid overlaps
            offset = _get_annotation_offset(i, algorithms)
            plt.annotate(
                f"({alg['steps']},{alg['rounds']},{alg['chunks_per_node']})",
                (alg['latency_cost'], alg['bandwidth_cost']),
                xytext=offset, 
                textcoords='offset points', 
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7) if i in pareto_indices else None,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1')
            )
    
    # Connect Pareto-optimal points
    pareto_algs = [algorithms[i] for i in sorted(pareto_indices, 
                   key=lambda idx: algorithms[idx]['latency_cost'])]
    if len(pareto_algs) > 1:
        pareto_lat = [alg['latency_cost'] for alg in pareto_algs]
        pareto_bw = [alg['bandwidth_cost'] for alg in pareto_algs]
        plt.plot(pareto_lat, pareto_bw, 'r--', alpha=0.7, linewidth=2.5, 
                label='Pareto frontier')
    
    # Styling
    plt.xlabel('Latency Cost (Steps)', fontsize=14, fontweight='bold')
    plt.ylabel('Bandwidth Cost (R/C)', fontsize=14, fontweight='bold')
    
    # Title
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        plt.title('Pareto Frontier for Collective Communication Algorithms', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add text box with explanation
    textstr = 'Annotation format: (steps, rounds, chunks/node)\n'
    textstr += f'Total algorithms: {len(algorithms)}\n'
    textstr += f'Pareto-optimal: {len(pareto_indices)}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', bbox=props)
    
    # Set axis limits with some padding
    x_margin = (max(latencies) - min(latencies)) * 0.1
    y_margin = (max(bandwidths) - min(bandwidths)) * 0.1
    plt.xlim(min(latencies) - x_margin, max(latencies) + x_margin)
    plt.ylim(min(bandwidths) - y_margin, max(bandwidths) + y_margin)
    
    plt.tight_layout()
    
    # Save plot
    saved_path = None
    if save_path is None:
        # Auto-generate filename with timestamp
        os.makedirs('paretocc/plots', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'paretocc/plots/pareto_frontier_{timestamp}.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    saved_path = save_path
    print(f"Plot saved to: {saved_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_path


def _find_pareto_optimal_indices(algorithms: List[Dict]) -> List[int]:
    """Find indices of Pareto-optimal algorithms"""
    pareto_indices = []
    
    for i, alg_i in enumerate(algorithms):
        is_pareto = True
        for j, alg_j in enumerate(algorithms):
            if i != j:
                # Check if alg_j dominates alg_i
                if (alg_j['latency_cost'] <= alg_i['latency_cost'] and 
                    alg_j['bandwidth_cost'] <= alg_i['bandwidth_cost'] and
                    (alg_j['latency_cost'] < alg_i['latency_cost'] or 
                     alg_j['bandwidth_cost'] < alg_i['bandwidth_cost'])):
                    is_pareto = False
                    break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices


def _get_annotation_offset(idx: int, algorithms: List[Dict]) -> Tuple[float, float]:
    """Get smart offset for annotation to avoid overlaps"""
    # Basic offset pattern
    offsets = [
        (10, 10), (-10, 10), (10, -10), (-10, -10),
        (15, 0), (-15, 0), (0, 15), (0, -15)
    ]
    
    # Check for nearby points and adjust
    current = algorithms[idx]
    nearby_count = 0
    
    for i, other in enumerate(algorithms):
        if i != idx:
            lat_diff = abs(current['latency_cost'] - other['latency_cost'])
            bw_diff = abs(current['bandwidth_cost'] - other['bandwidth_cost'])
            
            # If points are close
            if lat_diff < 0.5 and bw_diff < 0.5:
                nearby_count += 1
    
    # Use different offset based on nearby points
    offset_idx = (idx + nearby_count) % len(offsets)
    return offsets[offset_idx]


def compare_topologies(results_dict: Dict[str, List[Dict]], 
                      save_path: Optional[str] = None) -> Optional[str]:
    """
    Compare Pareto frontiers across different topologies
    
    Args:
        results_dict: Dictionary mapping topology name to list of algorithms
        save_path: Path to save the comparison plot
        
    Returns:
        Path where the plot was saved
    """
    plt.figure(figsize=(14, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (topology_name, algorithms) in enumerate(results_dict.items()):
        if not algorithms:
            continue
            
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Extract costs
        latencies = [alg['latency_cost'] for alg in algorithms]
        bandwidths = [alg['bandwidth_cost'] for alg in algorithms]
        
        # Plot points
        plt.scatter(latencies, bandwidths, s=100, alpha=0.7, 
                   c=color, marker=marker, edgecolors='black', 
                   linewidth=1, label=topology_name)
        
        # Connect points for this topology
        sorted_algs = sorted(algorithms, key=lambda x: x['latency_cost'])
        if len(sorted_algs) > 1:
            lat = [alg['latency_cost'] for alg in sorted_algs]
            bw = [alg['bandwidth_cost'] for alg in sorted_algs]
            plt.plot(lat, bw, '--', color=color, alpha=0.5, linewidth=1.5)
    
    plt.xlabel('Latency Cost (Steps)', fontsize=14, fontweight='bold')
    plt.ylabel('Bandwidth Cost (R/C)', fontsize=14, fontweight='bold')
    plt.title('Pareto Frontier Comparison Across Topologies', 
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path is None:
        os.makedirs('sccl/plots', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'sccl/plots/topology_comparison_{timestamp}.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path