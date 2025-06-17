# enhanced_synthesizer.py
import itertools
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import heapq
import random
from .path_based_synthesizer import *
from ..topology import create_dgx1_topology

class LookaheadMergingStrategy:
    """미래의 병합 기회를 고려한 병합 전략"""
    
    def __init__(self, lookahead_depth: int = 2):
        self.lookahead_depth = lookahead_depth
    
    def score_merge_decision(self, 
                           current_edge: Tuple[int, int],
                           chunks_to_merge: Set[Tuple[int, int]],  # (chunk_id, destination)
                           paths_dict: Dict[Tuple[int, int], Path],
                           path_overlaps: Dict) -> float:
        """병합 결정의 점수를 계산"""
        score = 0.0
        
        # 1. 즉각적인 이득 (병합된 청크 수)
        immediate_gain = len(chunks_to_merge)
        score += immediate_gain
        
        # 2. 미래 병합 기회 평가
        for chunk_key in chunks_to_merge:
            if chunk_key not in paths_dict:
                continue
                
            path = paths_dict[chunk_key]
            
            # 현재 엣지의 위치 찾기
            try:
                current_pos = path.edges.index(current_edge)
            except ValueError:
                continue
            
            # 앞으로 lookahead_depth 만큼의 엣지 확인
            for i in range(1, min(self.lookahead_depth + 1, 
                                len(path.edges) - current_pos)):
                future_edge = path.edges[current_pos + i]
                
                # 이 미래 엣지에서 만날 수 있는 다른 청크들
                future_companions = set()
                for other_chunk_key in chunks_to_merge:
                    if other_chunk_key != chunk_key and other_chunk_key in paths_dict:
                        other_path = paths_dict[other_chunk_key]
                        if future_edge in other_path.edges:
                            future_companions.add(other_chunk_key)
                
                # 미래 병합 가능성에 따른 보너스
                if future_companions:
                    score += len(future_companions) * (0.5 ** i)  # 거리에 따라 감소
        
        return score


class EnhancedPathBasedSynthesizer(PathBasedSynthesizer):
    """향상된 병합 전략을 포함한 경로 기반 합성기"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                 merge_strategy: str = 'greedy'):
        super().__init__(alpha, beta)
        self.merge_strategy = merge_strategy
        self.lookahead_strategy = LookaheadMergingStrategy(lookahead_depth=2)
    
    def synthesize_allgather(self, topology: 'Topology') -> Dict:
        """선택된 병합 전략으로 AllGather 합성"""
        n_nodes = topology.num_nodes
        
        # Step 1: 모든 청크의 최단 경로 계산
        all_paths = self._compute_all_shortest_paths_allgather(topology)
        
        # Step 2: 경로 중첩 분석
        path_overlaps = self._analyze_path_overlaps(all_paths)
        
        # Step 3: 선택된 전략으로 스케줄링
        if self.merge_strategy == 'lookahead':
            schedule = self._schedule_with_lookahead(all_paths, path_overlaps, topology)
        elif self.merge_strategy == 'dp':
            schedule = self._schedule_with_dp(all_paths, topology)
        else:
            schedule = self._schedule_with_merging(all_paths, path_overlaps, topology)
        
        # Step 4: 비용 계산
        total_cost = self._calculate_cost(schedule)
        
        return {
            'paths': all_paths,
            'schedule': schedule,
            'cost': total_cost,
            'stats': self._get_statistics(schedule),
            'strategy': self.merge_strategy
        }
    
    def _schedule_with_lookahead(self, paths: List[Path], 
                                path_overlaps: Dict[Tuple[int, int], List[Path]], 
                                topology: 'Topology') -> List[MergedTransfer]:
        """Lookahead 전략을 사용한 스케줄링"""
        schedule = []
        time_slot = 0
        
        # 경로를 (chunk_id, destination)으로 인덱싱
        paths_dict = {}
        for path in paths:
            key = (path.chunk_id, path.destination)
            paths_dict[key] = path
        
        # 각 엣지의 사용 가능 시간
        edge_available_time = defaultdict(int)
        
        # 각 청크-목적지 쌍의 현재 위치
        chunk_location = {}
        for path in paths:
            chunk_location[(path.chunk_id, path.destination)] = path.source
        
        # pending_paths를 키로 관리 (Path 객체 대신)
        pending_keys = set(paths_dict.keys())
        
        while pending_keys:
            time_slot += 1
            transfers_this_slot = []
            
            # 모든 가능한 병합 옵션 평가
            merge_options = []
            
            for edge, edge_paths in path_overlaps.items():
                if edge_available_time[edge] >= time_slot:
                    continue
                
                # 이 엣지에서 전송 가능한 청크들
                ready_chunks = []
                for path in edge_paths:
                    key = (path.chunk_id, path.destination)
                    
                    if key not in pending_keys:
                        continue
                    
                    current_loc = chunk_location.get(key)
                    
                    if current_loc == edge[0]:
                        # 다음 노드가 맞는지 확인
                        try:
                            current_idx = path.nodes.index(current_loc)
                            if current_idx < len(path.nodes) - 1:
                                next_node = path.nodes[current_idx + 1]
                                if next_node == edge[1]:
                                    ready_chunks.append(key)
                        except ValueError:
                            continue
                
                if ready_chunks:
                    # 가능한 모든 병합 조합에 대해 점수 계산
                    max_merge_size = min(len(ready_chunks), 4)  # 최대 4개까지 병합
                    
                    for r in range(1, max_merge_size + 1):
                        for chunk_subset in itertools.combinations(ready_chunks, r):
                            chunk_subset_set = set(chunk_subset)
                            
                            # Lookahead 점수 계산
                            score = self.lookahead_strategy.score_merge_decision(
                                edge, chunk_subset_set, paths_dict, path_overlaps
                            )
                            
                            merge_options.append((score, edge, chunk_subset))
            
            # 가장 높은 점수의 병합 선택
            if merge_options:
                merge_options.sort(reverse=True, key=lambda x: x[0])
                
                # 상위 옵션 중에서 선택 (충돌 방지)
                used_edges = set()
                
                for score, edge, chunk_subset in merge_options:
                    if edge not in used_edges:
                        # 이 병합 수행
                        chunks_ids = {chunk_id for chunk_id, _ in chunk_subset}
                        
                        transfer = MergedTransfer(
                            chunks=chunks_ids,
                            edge=edge,
                            time_slot=time_slot
                        )
                        transfers_this_slot.append(transfer)
                        used_edges.add(edge)
                        
                        # 상태 업데이트
                        for chunk_id, dest in chunk_subset:
                            chunk_location[(chunk_id, dest)] = edge[1]
                            
                            if edge[1] == dest:
                                # 목적지 도달 - 해당 키 제거
                                pending_keys.discard((chunk_id, dest))
                        
                        edge_available_time[edge] = time_slot + 1
            
            schedule.extend(transfers_this_slot)
            
            # 무한 루프 방지
            if time_slot > 100:
                print(f"경고: 시간 제한 초과. 남은 경로: {len(pending_keys)}")
                break
        
        return schedule

    def _schedule_with_dp(self, paths: List[Path], 
                         topology: 'Topology') -> List[MergedTransfer]:
        """동적 프로그래밍을 사용한 최적 스케줄링"""
        n_paths = len(paths)
        
        # if n_paths > 10:
        #     print(f"경고: DP는 {n_paths}개 경로에 너무 느립니다. Greedy로 전환합니다.")
        #     return self._schedule_with_merging(paths, 
        #                                      self._analyze_path_overlaps(paths), 
        #                                      topology)
        
        print(f"DP 스케줄링 시작: {n_paths}개 경로")
        
        # 각 경로를 (chunk_id, dest)로 인덱싱
        path_keys = [(p.chunk_id, p.destination) for p in paths]
        
        # 상태: 각 경로의 현재 노드 위치
        initial_state = tuple(p.source for p in paths)
        goal_state = tuple(p.destination for p in paths)
        
        # DP 테이블: state -> (cost, parent_state, transfers)
        from collections import deque
        import heapq
        
        # Priority queue: (cost, state)
        pq = [(0, initial_state)]
        best_cost = {initial_state: 0}
        parent = {}
        
        while pq:
            current_cost, current_state = heapq.heappop(pq)
            
            if current_state == goal_state:
                # 해를 찾음
                return self._reconstruct_dp_path(parent, current_state, paths)
            
            if current_cost > best_cost.get(current_state, float('inf')):
                continue
            
            # 각 엣지에 대해 가능한 전송 찾기
            for edge in topology.get_links():
                src, dst = edge
                
                # 이 엣지를 사용할 수 있는 경로들
                can_use = []
                for i, (path, current_pos) in enumerate(zip(paths, current_state)):
                    if current_pos == src and dst in path.nodes:
                        # 목적지로 가는 경로상에 있는지 확인
                        try:
                            idx = path.nodes.index(current_pos)
                            if idx < len(path.nodes) - 1 and path.nodes[idx + 1] == dst:
                                can_use.append(i)
                        except ValueError:
                            pass
                
                if not can_use:
                    continue
                
                # 가능한 부분집합들 (병합)
                for r in range(1, min(len(can_use) + 1, 4)):
                    for subset in itertools.combinations(can_use, r):
                        # 새 상태 생성
                        new_state = list(current_state)
                        for idx in subset:
                            new_state[idx] = dst
                        new_state = tuple(new_state)
                        
                        # 비용 계산
                        new_cost = current_cost + self.alpha + self.beta
                        
                        # 더 좋은 경로면 업데이트
                        if new_cost < best_cost.get(new_state, float('inf')):
                            best_cost[new_state] = new_cost
                            parent[new_state] = (current_state, edge, set(subset))
                            heapq.heappush(pq, (new_cost, new_state))
        
        print("경고: DP가 해를 찾지 못했습니다.")
        return []
    
    def _reconstruct_dp_path(self, parent: Dict, goal_state: Tuple, 
                           paths: List[Path]) -> List[MergedTransfer]:
        """DP 경로 재구성"""
        schedule = []
        current = goal_state
        
        # 역순으로 경로 추적
        transitions = []
        while current in parent:
            prev_state, edge, path_indices = parent[current]
            transitions.append((edge, path_indices))
            current = prev_state
        
        # 정방향으로 변환
        transitions.reverse()
        
        # 스케줄 생성
        time_slot = 0
        for edge, path_indices in transitions:
            time_slot += 1
            
            # 청크 ID들 수집
            chunks = {paths[idx].chunk_id for idx in path_indices}
            
            transfer = MergedTransfer(
                chunks=chunks,
                edge=edge,
                time_slot=time_slot
            )
            schedule.append(transfer)
        
        return schedule


# 테스트 코드
if __name__ == "__main__":
    
    # 4-node ring topology
    dgx_1 = create_dgx1_topology()  # DGX-1 토폴로지
    
    print("=== 병합 전략 비교 ===\n")
    
    # 1. Greedy (기본) 전략
    synthesizer_greedy = EnhancedPathBasedSynthesizer(alpha=1.0, beta=0.5, 
                                                     merge_strategy='greedy')
    result_greedy = synthesizer_greedy.synthesize_allgather(dgx_1)
    print(f"Greedy 전략:")
    print(f"  총 비용: {result_greedy['cost']['total']:.2f}")
    print(f"  시간 슬롯: {result_greedy['cost']['time_slots']}")
    print(f"  전송 수: {result_greedy['cost']['transfers']}")
    print(f"  병합 절약: {result_greedy['cost']['savings']}\n")
    
    # 2. Lookahead 전략
    synthesizer_lookahead = EnhancedPathBasedSynthesizer(alpha=1.0, beta=0.5,
                                                        merge_strategy='lookahead')
    result_lookahead = synthesizer_lookahead.synthesize_allgather(dgx_1)
    print(f"Lookahead 전략:")
    print(f"  총 비용: {result_lookahead['cost']['total']:.2f}")
    print(f"  시간 슬롯: {result_lookahead['cost']['time_slots']}")
    print(f"  전송 수: {result_lookahead['cost']['transfers']}")
    print(f"  병합 절약: {result_lookahead['cost']['savings']}\n")
    
    # 전략 비교
    if result_lookahead['cost']['total'] < result_greedy['cost']['total']:
        improvement = (result_greedy['cost']['total'] - result_lookahead['cost']['total']) / result_greedy['cost']['total'] * 100
        print(f"Lookahead가 Greedy보다 {improvement:.1f}% 개선됨")
    
    # 3. DP 전략 (작은 문제에만)
    dgx_1 = create_dgx1_topology()  # DGX-1 토폴로지
    synthesizer_dp = EnhancedPathBasedSynthesizer(alpha=1.0, beta=0.5,
                                                 merge_strategy='dp')
    result_dp = synthesizer_dp.synthesize_allgather(dgx_1)
    print(f"\nDP 전략 (3-node ring):")
    print(f"  총 비용: {result_dp['cost']['total']:.2f}")
    print(f"  시간 슬롯: {result_dp['cost']['time_slots']}")
    print(f"  전송 수: {result_dp['cost']['transfers']}")
    print(f"  병합 절약: {result_dp['cost']['savings']}\n")
    
    # 시각화
    print("Lookahead 전략 시각화 (4-node ring):")
    synthesizer_lookahead.visualize_schedule(result_lookahead['schedule'], dgx_1)