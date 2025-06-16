# path_based_synthesizer.py
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from .topology import *


@dataclass
class Path:
    """청크의 경로 정보"""
    chunk_id: int
    source: int
    destination: int
    nodes: List[int]  # 경로상의 노드들
    edges: List[Tuple[int, int]]  # 경로상의 엣지들
    length: int  # 경로 길이 (홉 수)

@dataclass
class MergedTransfer:
    """병합된 전송 정보"""
    chunks: Set[int]  # 함께 전송되는 청크들
    edge: Tuple[int, int]  # 전송 엣지
    time_slot: int  # 전송 시간 슬롯

class PathBasedSynthesizer:
    """경로 기반 집합 통신 합성기"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha  # latency cost
        self.beta = beta    # bandwidth cost
        
    def synthesize_allgather(self, topology: 'Topology') -> Dict:
        """AllGather를 위한 경로 기반 합성"""
        n_nodes = topology.num_nodes
        
        # Step 1: 모든 청크의 최단 경로 계산
        all_paths = self._compute_all_shortest_paths_allgather(topology)
        
        # Step 2: 경로 중첩 분석
        path_overlaps = self._analyze_path_overlaps(all_paths)
        
        # Step 3: 병합 기회를 고려한 스케줄링
        schedule = self._schedule_with_merging(all_paths, path_overlaps, topology)
        
        # Step 4: 비용 계산
        total_cost = self._calculate_cost(schedule)
        
        return {
            'paths': all_paths,
            'schedule': schedule,
            'cost': total_cost,
            'stats': self._get_statistics(schedule)
        }
    
    def _compute_all_shortest_paths_allgather(self, topology: 'Topology') -> List[Path]:
        """AllGather를 위한 모든 청크의 최단 경로 계산"""
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        for (src, dst) in topology.get_links():
            G.add_edge(src, dst)
        
        all_paths = []
        n_nodes = topology.num_nodes
        
        # AllGather: 각 노드의 청크를 모든 다른 노드로 전송
        for source in range(n_nodes):
            chunk_id = source  # 간단히 노드 번호를 청크 ID로 사용
            
            for dest in range(n_nodes):
                if source != dest:
                    try:
                        # 최단 경로 찾기
                        path_nodes = nx.shortest_path(G, source, dest)
                        path_edges = [(path_nodes[i], path_nodes[i+1]) 
                                     for i in range(len(path_nodes)-1)]
                        
                        path = Path(
                            chunk_id=chunk_id,
                            source=source,
                            destination=dest,
                            nodes=path_nodes,
                            edges=path_edges,
                            length=len(path_edges)
                        )
                        all_paths.append(path)
                        
                    except nx.NetworkXNoPath:
                        print(f"경로 없음: {source} -> {dest}")
        
        return all_paths
    
    def _analyze_path_overlaps(self, paths: List[Path]) -> Dict[Tuple[int, int], List[Path]]:
        """경로 중첩 분석 - 같은 엣지를 사용하는 경로들 그룹화"""
        edge_to_paths = defaultdict(list)
        
        for path in paths:
            for edge in path.edges:
                edge_to_paths[edge].append(path)
        
        return dict(edge_to_paths)
    
    def _schedule_with_merging(self, paths: List[Path], 
                              path_overlaps: Dict[Tuple[int, int], List[Path]], 
                              topology: 'Topology') -> List[MergedTransfer]:
        """병합 기회를 고려한 스케줄링"""
        schedule = []
        time_slot = 0
        
        # 각 엣지의 사용 가능 시간 추적
        edge_available_time = defaultdict(int)
        
        # 각 청크-목적지 쌍의 현재 위치 추적
        chunk_location = {}  # (chunk_id, destination) -> current_node
        
        # 초기화: 각 청크는 소스에 위치
        for path in paths:
            chunk_location[(path.chunk_id, path.destination)] = path.source
        
        # 전송해야 할 경로들
        pending_paths = paths.copy()
        
        while pending_paths:
            time_slot += 1
            transfers_this_slot = []
            
            # 이번 시간 슬롯에 가능한 전송들 찾기
            for edge, edge_paths in path_overlaps.items():
                if edge_available_time[edge] >= time_slot:
                    continue  # 이 엣지는 아직 사용 중
                
                # 이 엣지를 사용할 수 있는 청크들 찾기
                ready_chunks = set()
                
                for path in edge_paths:
                    if path not in pending_paths:
                        continue
                    
                    # 청크가 현재 엣지의 소스 노드에 있는지 확인
                    current_loc = chunk_location.get((path.chunk_id, path.destination))
                    if current_loc == edge[0]:
                        # 이 경로에서 다음 엣지가 현재 엣지인지 확인
                        current_idx = path.nodes.index(current_loc)
                        if current_idx < len(path.nodes) - 1:
                            next_node = path.nodes[current_idx + 1]
                            if (current_loc, next_node) == edge:
                                ready_chunks.add((path.chunk_id, path.destination))
                
                if ready_chunks:
                    # 병합 전송 생성
                    transfer = MergedTransfer(
                        chunks={chunk for chunk, _ in ready_chunks},
                        edge=edge,
                        time_slot=time_slot
                    )
                    transfers_this_slot.append(transfer)
                    
                    # 청크 위치 업데이트
                    for chunk_id, dest in ready_chunks:
                        chunk_location[(chunk_id, dest)] = edge[1]
                        
                        # 목적지에 도달했는지 확인
                        if edge[1] == dest:
                            # 해당 경로를 pending에서 제거
                            path_to_remove = None
                            for p in pending_paths:
                                if p.chunk_id == chunk_id and p.destination == dest:
                                    path_to_remove = p
                                    break
                            if path_to_remove:
                                pending_paths.remove(path_to_remove)
                    
                    # 엣지 사용 시간 업데이트
                    edge_available_time[edge] = time_slot + 1
            
            schedule.extend(transfers_this_slot)
        
        return schedule
    
    def _calculate_cost(self, schedule: List[MergedTransfer]) -> Dict[str, float]:
        """스케줄의 비용 계산"""
        if not schedule:
            return {'total': 0, 'latency': 0, 'bandwidth': 0}
        
        # 총 시간 슬롯 수 (latency)
        max_time_slot = max(transfer.time_slot for transfer in schedule)
        
        # 총 전송 수 (bandwidth 관련)
        total_transfers = len(schedule)
        
        # 병합으로 인한 절약 계산
        total_chunks_moved = sum(len(transfer.chunks) for transfer in schedule)
        savings = total_chunks_moved - total_transfers
        
        # 비용 계산 (간단한 모델)
        latency_cost = self.alpha * max_time_slot
        bandwidth_cost = self.beta * total_transfers
        
        return {
            'total': latency_cost + bandwidth_cost,
            'latency': latency_cost,
            'bandwidth': bandwidth_cost,
            'time_slots': max_time_slot,
            'transfers': total_transfers,
            'savings': savings
        }
    
    def _get_statistics(self, schedule: List[MergedTransfer]) -> Dict:
        """스케줄 통계"""
        if not schedule:
            return {}
        
        # 병합 통계
        merge_sizes = [len(transfer.chunks) for transfer in schedule]
        
        # 엣지별 사용 횟수
        edge_usage = defaultdict(int)
        for transfer in schedule:
            edge_usage[transfer.edge] += 1
        
        return {
            'total_transfers': len(schedule),
            'average_merge_size': sum(merge_sizes) / len(merge_sizes),
            'max_merge_size': max(merge_sizes),
            'edge_usage': dict(edge_usage),
            'total_time_slots': max(t.time_slot for t in schedule)
        }
    
    def visualize_schedule(self, schedule: List[MergedTransfer], topology: 'Topology'):
        """스케줄 시각화"""
        # 시간 슬롯별로 그룹화
        slots = defaultdict(list)
        for transfer in schedule:
            slots[transfer.time_slot].append(transfer)
        
        max_slot = max(slots.keys()) if slots else 0
        
        # 각 시간 슬롯에 대한 서브플롯 생성
        fig, axes = plt.subplots(1, max_slot, figsize=(4*max_slot, 4))
        if max_slot == 1:
            axes = [axes]
        
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        for i in range(topology.num_nodes):
            G.add_node(i)
        for (src, dst) in topology.get_links():
            G.add_edge(src, dst)
        
        pos = nx.circular_layout(G)
        
        for slot in range(1, max_slot + 1):
            ax = axes[slot-1]
            ax.set_title(f'Time Slot {slot}')
            
            # 기본 네트워크 그리기
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
            nx.draw_networkx_labels(G, pos, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.3)
            
            # 활성 전송 표시
            if slot in slots:
                active_edges = [t.edge for t in slots[slot]]
                edge_labels = {}
                
                for transfer in slots[slot]:
                    chunks_str = ','.join(map(str, sorted(transfer.chunks)))
                    edge_labels[transfer.edge] = f"C{{{chunks_str}}}"
                
                nx.draw_networkx_edges(G, pos, edgelist=active_edges, ax=ax,
                                     edge_color='red', width=3, arrows=True)
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# 테스트 코드
if __name__ == "__main__":
    
    # 4-node ring topology 생성
    ring_4 = create_ring_topology(4)
    
    # 경로 기반 합성기 생성
    synthesizer = PathBasedSynthesizer(alpha=1.0, beta=0.5)
    
    # AllGather 합성
    print("=== 4-Node Ring AllGather 합성 ===")
    result = synthesizer.synthesize_allgather(ring_4)
    
    # 결과 출력
    print(f"\n총 비용: {result['cost']['total']:.2f}")
    print(f"  - Latency 비용: {result['cost']['latency']:.2f} (시간 슬롯: {result['cost']['time_slots']})")
    print(f"  - Bandwidth 비용: {result['cost']['bandwidth']:.2f} (전송 수: {result['cost']['transfers']})")
    print(f"  - 병합으로 인한 절약: {result['cost']['savings']} 전송")
    
    print(f"\n통계:")
    stats = result['stats']
    print(f"  - 총 전송 수: {stats['total_transfers']}")
    print(f"  - 평균 병합 크기: {stats['average_merge_size']:.2f}")
    print(f"  - 최대 병합 크기: {stats['max_merge_size']}")
    print(f"  - 총 시간 슬롯: {stats['total_time_slots']}")
    
    print(f"\n엣지별 사용 횟수:")
    for edge, count in stats['edge_usage'].items():
        print(f"  {edge}: {count}회")
    
    # 스케줄 시각화
    print("\n스케줄 시각화:")
    synthesizer.visualize_schedule(result['schedule'], ring_4)
    
    # SCCL과 비교를 위한 추가 분석
    print("\n=== SCCL과 비교 ===")
    # Ring에서 AllGather의 이론적 최적값
    # - Latency optimal: 2 steps (diameter)
    # - Bandwidth optimal: 3 steps, 3 rounds
    print(f"경로 기반 접근: {result['cost']['time_slots']} 시간 슬롯")
    print("SCCL latency optimal: 2 steps")
    print("SCCL bandwidth optimal: 3 steps")