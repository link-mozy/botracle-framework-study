"""
BOTracle 3단계 - Website Traversal Graph (WT Graph)

이 파일은 BOTracle 논문 Section 3.3의 WT Graph를 구현합니다.
머신러닝 초급자를 위해 모든 개념에 상세한 주석을 달았습니다.

═══════════════════════════════════════════════════════════════
  WT Graph가 뭔가요? (비유로 설명)
═══════════════════════════════════════════════════════════════

  쇼핑몰에서 사람들의 동선을 추적하는 CCTV를 떠올려보세요:

  - 노드(Node) = 쇼핑몰의 각 매장 (웹페이지)
  - 엣지(Edge) = 매장 사이의 이동 경로 (페이지 간 네비게이션)
  - 엣지 가중치 = 그 경로를 지나간 횟수

  사람 고객:
    입구 → 마음에 드는 매장 2~3곳만 방문 → 되돌아가서 비교 → 구매

  봇(자동 방문자):
    입구 → 모든 매장을 순서대로 방문 → 빠짐없이 체계적으로 순회

  이 "동선 지도"가 바로 WT Graph이고,
  동선 패턴의 차이를 분석해 봇인지 사람인지 판별합니다.


═══════════════════════════════════════════════════════════════
  WT Graph 구조 (논문 Table 1)
═══════════════════════════════════════════════════════════════

  예시: 사용자가 홈 → 카테고리 → 상품A → 카테고리 → 상품B 순서로 탐색

        ┌─────┐    1     ┌──────────┐    1     ┌────────┐
        │ 홈  │ ──────▶ │ 카테고리  │ ──────▶ │ 상품A  │
        └─────┘         └──────────┘         └────────┘
                              │  ▲
                            1 │  │ 1  (뒤로가기)
                              ▼  │
                         ┌────────┐
                         │ 상품B  │
                         └────────┘

  - 노드: 홈, 카테고리, 상품A, 상품B (4개)
  - 엣지: 홈→카테고리(1), 카테고리→상품A(1), 상품A→카테고리(1),
          카테고리→상품B(1) (4개)
  - 카테고리 노드의 hit_count = 2 (2번 방문)


═══════════════════════════════════════════════════════════════
  추출하는 메트릭 (논문 기반)
═══════════════════════════════════════════════════════════════

  1. Node Degree       : 노드에 연결된 엣지 수
  2. Node Count        : 전체 노드 수
  3. Edge Count        : 전체 엣지 수
  4. Page Type Dist.   : 페이지 유형별 비율
  5. Session Topics    : RAKE로 추출한 키워드
  6. Number of Hits    : 총 방문 횟수
  7. Hits per Sub Page : 페이지별 방문 횟수
  8. Degree Centrality : 노드의 연결 중심성
  9. Betweenness Cent. : 노드의 매개 중심성
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 데이터 클래스 정의
# ──────────────────────────────────────────────

@dataclass
class WTNode:
    """
    WT Graph의 노드 (= 웹페이지 하나)

    비유: 쇼핑몰의 매장 하나

    Attributes:
        id:              페이지 이름 (예: "category/electronics")
        page_type:       페이지 유형 (예: "product", "category")
        timestamps:      이 페이지를 방문한 시각 목록
        hit_count:       이 페이지의 총 방문 횟수
        benchmark_label: 벤치마크 라벨 (학습 데이터용)
        page_title:      페이지 제목 (RAKE 키워드 추출용)
    """
    id: str
    page_type: str = "other"
    timestamps: List[float] = field(default_factory=list)
    hit_count: int = 0
    benchmark_label: str = ""
    page_title: str = ""


@dataclass
class WTEdge:
    """
    WT Graph의 엣지 (= 페이지 간 이동 경로)

    비유: 쇼핑몰에서 매장 A에서 매장 B로 가는 통로

    Attributes:
        source: 출발 페이지 (이전 페이지)
        target: 도착 페이지 (현재 페이지)
        weight: 이 경로를 지나간 횟수 (많을수록 자주 이동한 경로)
    """
    source: str
    target: str
    weight: int = 1


# ──────────────────────────────────────────────
# 페이지 유형 상수
# ──────────────────────────────────────────────
# 쇼핑몰 기준 페이지 유형 분류
# DGCNN 노드 특징의 One-Hot 인코딩에 사용됩니다
PAGE_TYPES = [
    "home",       # 메인 페이지
    "product",    # 상품 상세
    "category",   # 카테고리 목록
    "search",     # 검색 결과
    "cart",       # 장바구니
    "checkout",   # 결제
    "account",    # 계정/마이페이지
    "other",      # 기타
]


class WTGraph:
    """
    Website Traversal Graph (WT Graph)

    사용자의 웹사이트 탐색 경로를 방향 그래프(directed graph)로 표현합니다.
    논문 Section 3.3 기반 구현.

    사용법:
        graph = WTGraph()
        graph.add_hit({
            "detailedPagename": "home",
            "previousPagename": None,
            "timestamp": 1707580800000,
            "pageType": "home",
            "pageTitle": "Welcome to TechShop",
        })
        graph.add_hit({
            "detailedPagename": "category/electronics",
            "previousPagename": "home",
            "timestamp": 1707580810000,
            "pageType": "category",
            "pageTitle": "Electronics & Gadgets",
        })
        metrics = graph.extract_metrics()
    """

    def __init__(self):
        # ── 그래프 저장소 ──
        # nodes: 페이지 이름 → WTNode 객체
        # edges: "출발→도착" 문자열 → WTEdge 객체
        self.nodes: Dict[str, WTNode] = {}
        self.edges: Dict[str, WTEdge] = {}

        # 세션의 첫 번째 페이지 (세션 시작점)
        self.first_hit_pagename: Optional[str] = None

    def add_hit(self, hit_data: dict) -> None:
        """
        새로운 hit(페이지 방문)을 WT Graph에 추가합니다.

        논문: "incoming data points, termed as hits, incrementally
               expand the existing session graph"

        비유: CCTV에 새로운 이동 기록이 찍힐 때마다
              동선 지도에 점과 화살표를 추가하는 것

        Args:
            hit_data: 히트 정보 딕셔너리
                - detailedPagename: 현재 방문한 페이지 이름
                - previousPagename: 이전 페이지 이름 (첫 방문이면 None)
                - timestamp:        방문 시각 (밀리초 단위)
                - pageType:         페이지 유형 (home, product 등)
                - pageTitle:        페이지 제목 (RAKE 키워드 추출용)
        """
        page_name = hit_data.get("detailedPagename", "")
        prev_page = hit_data.get("previousPagename")
        timestamp = hit_data.get("timestamp", 0)
        page_type = hit_data.get("pageType", "other")
        page_title = hit_data.get("pageTitle", "")

        if not page_name:
            return

        # 세션의 첫 번째 페이지 기록
        if self.first_hit_pagename is None:
            self.first_hit_pagename = page_name

        # ── 1단계: 노드 생성 또는 업데이트 ──
        # 같은 페이지를 다시 방문하면 기존 노드의 hit_count를 증가시킵니다
        # (새 노드를 만드는 게 아니라, 기존 노드에 "집계"합니다)
        if page_name not in self.nodes:
            self.nodes[page_name] = WTNode(
                id=page_name,
                page_type=page_type,
                timestamps=[timestamp],
                hit_count=1,
                page_title=page_title,
            )
        else:
            node = self.nodes[page_name]
            node.hit_count += 1
            node.timestamps.append(timestamp)

        # ── 2단계: 엣지 생성 또는 가중치 증가 ──
        # 이전 페이지가 있으면 "이전 → 현재" 방향으로 엣지를 연결합니다
        if prev_page:
            # 이전 페이지 노드가 아직 없으면 빈 노드를 생성
            # (이전 페이지 정보가 불완전할 수 있음)
            if prev_page not in self.nodes:
                self.nodes[prev_page] = WTNode(
                    id=prev_page,
                    page_type="other",
                    timestamps=[],
                    hit_count=0,
                )

            # 엣지 키: "출발페이지->도착페이지" 형식의 문자열
            edge_key = f"{prev_page}->{page_name}"

            if edge_key not in self.edges:
                # 새 경로 → 엣지 생성 (가중치 = 1)
                self.edges[edge_key] = WTEdge(
                    source=prev_page,
                    target=page_name,
                    weight=1,
                )
            else:
                # 같은 경로를 다시 지남 → 가중치 증가
                self.edges[edge_key].weight += 1

    # ══════════════════════════════════════════════
    # 메트릭 추출
    # ══════════════════════════════════════════════

    def extract_metrics(self) -> dict:
        """
        WT Graph에서 모든 메트릭을 추출합니다.
        이 메트릭들이 DGCNN의 입력 특징으로 사용됩니다.

        Returns:
            dict: 9개 메트릭이 담긴 딕셔너리
        """
        node_count = len(self.nodes)
        edge_count = len(self.edges)

        # 1. 총 히트 수 (모든 페이지 방문 횟수의 합)
        total_hits = sum(n.hit_count for n in self.nodes.values())

        # 2. 노드별 히트 수
        hits_per_sub_page = {
            node_id: node.hit_count
            for node_id, node in self.nodes.items()
        }

        # 3. Node Degree (각 노드에 연결된 엣지 수)
        node_degrees = self._compute_node_degrees()

        # 4. Page Type Distribution (페이지 유형별 비율)
        page_type_distribution = self._compute_page_type_distribution(
            total_hits
        )

        # 5. Degree Centrality (연결 중심성)
        degree_centrality = self._compute_degree_centrality(node_degrees)

        # 6. Betweenness Centrality (매개 중심성)
        betweenness_centrality = self._compute_betweenness_centrality()

        # 7. Session Topics (RAKE 키워드 추출)
        session_topics = self._extract_session_topics()

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "total_hits": total_hits,
            "hits_per_sub_page": hits_per_sub_page,
            "node_degrees": node_degrees,
            "page_type_distribution": page_type_distribution,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "session_topics": session_topics,
        }

    def _compute_node_degrees(self) -> Dict[str, int]:
        """
        각 노드의 차수(degree)를 계산합니다.

        차수 = 해당 노드에 연결된 엣지의 총 수
        (들어오는 엣지 + 나가는 엣지)

        비유: 한 매장에 연결된 통로의 수
              통로가 많은 매장 = 허브(중심지)

        예시:
             A → B → C
             │       ▲
             └───────┘

          A의 degree = 2 (A→B, A→C)
          B의 degree = 2 (A→B, B→C)
          C의 degree = 2 (B→C, A→C)
        """
        degrees: Dict[str, int] = {node_id: 0 for node_id in self.nodes}

        for edge in self.edges.values():
            # 나가는 엣지: source의 degree +1
            if edge.source in degrees:
                degrees[edge.source] += 1
            # 들어오는 엣지: target의 degree +1
            if edge.target in degrees:
                degrees[edge.target] += 1

        return degrees

    def _compute_page_type_distribution(
        self, total_hits: int
    ) -> Dict[str, float]:
        """
        페이지 유형별 방문 비율을 계산합니다.

        비유: 쇼핑몰에서 식당가, 의류매장, 전자제품매장 등
              각 구역을 얼마나 자주 방문했는지의 비율

        봇 vs 사람 차이:
          - 봇: 모든 유형을 균일하게 방문 (비율이 비슷)
          - 사람: 관심 있는 유형에 집중 (product가 높고 checkout은 낮음)
        """
        distribution: Dict[str, float] = {}

        for node in self.nodes.values():
            ptype = node.page_type
            distribution[ptype] = distribution.get(ptype, 0) + node.hit_count

        # 비율로 변환 (합이 1이 되도록)
        if total_hits > 0:
            for key in distribution:
                distribution[key] /= total_hits

        return distribution

    def _compute_degree_centrality(
        self, node_degrees: Dict[str, int]
    ) -> Dict[str, float]:
        """
        각 노드의 연결 중심성(Degree Centrality)을 계산합니다.

        수식: DC(v) = degree(v) / (N - 1)
          - degree(v) = 노드 v에 연결된 엣지 수
          - N = 전체 노드 수

        비유: 쇼핑몰에서 한 매장이 다른 매장들과 얼마나 잘 연결되어 있는지

        값 범위: 0.0 (고립된 노드) ~ 1.0 (모든 노드와 연결)

        예시 (노드 4개: A, B, C, D):
          A가 B, C, D 모두와 연결 → degree=3, DC = 3/(4-1) = 1.0
          D가 A만과 연결           → degree=1, DC = 1/(4-1) = 0.33
        """
        n = len(self.nodes)
        centrality: Dict[str, float] = {}

        for node_id, degree in node_degrees.items():
            if n > 1:
                centrality[node_id] = degree / (n - 1)
            else:
                centrality[node_id] = 0.0

        return centrality

    def _compute_betweenness_centrality(self) -> Dict[str, float]:
        """
        각 노드의 매개 중심성(Betweenness Centrality)을 계산합니다.

        ═══════════════════════════════════════════════
        매개 중심성이란?
        ═══════════════════════════════════════════════

        "다른 모든 노드 쌍 사이의 최단 경로 중,
         이 노드를 지나가는 경로의 비율"

        비유: 쇼핑몰에서 어떤 매장이 "교차로" 역할을 하는지
              A매장에서 C매장으로 가려면 반드시 B매장을 지나야 한다면,
              B매장의 betweenness centrality가 높습니다.

        수식: BC(v) = Σ_{s≠v≠t} (σ_st(v) / σ_st)
          - σ_st     = s에서 t로 가는 최단 경로의 총 수
          - σ_st(v)  = 그 중 v를 경유하는 경로의 수

        값 범위: 0.0 (아무도 경유하지 않음) ~ 1.0 (모든 경로가 이 노드를 경유)

        예시:
              A → B → C → D
                       ↗
              E ──────┘

          C의 BC가 높음: A→D, B→D, E→D 모두 C를 경유해야 함

        ═══════════════════════════════════════════════
        Brandes 알고리즘 (효율적 계산 방법)
        ═══════════════════════════════════════════════

        모든 노드 쌍의 최단 경로를 일일이 구하면 너무 느립니다.
        Brandes 알고리즘은 BFS(너비 우선 탐색)를 사용하여
        각 출발 노드에서 한 번의 탐색으로 모든 경유 정보를 계산합니다.

        복잡도: O(V × E) (V=노드 수, E=엣지 수)
        """
        node_ids = list(self.nodes.keys())
        n = len(node_ids)

        # 결과 저장: 각 노드의 매개 중심성
        bc: Dict[str, float] = {nid: 0.0 for nid in node_ids}

        if n <= 2:
            return bc

        # ── 인접 리스트 구성 (그래프 탐색용) ──
        # 각 노드에서 갈 수 있는 이웃 노드 목록
        adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for edge in self.edges.values():
            if edge.source in adjacency:
                adjacency[edge.source].append(edge.target)

        # ── 각 출발 노드(s)에 대해 BFS 수행 ──
        for s in node_ids:
            # ── BFS 준비 ──
            # stack: BFS에서 방문한 노드를 역순으로 처리하기 위한 스택
            stack: List[str] = []

            # predecessors: 각 노드의 최단 경로에서의 직전 노드 목록
            # (최단 경로가 여러 개일 수 있으므로 리스트)
            predecessors: Dict[str, List[str]] = {
                nid: [] for nid in node_ids
            }

            # sigma: s에서 각 노드까지의 최단 경로 수
            sigma: Dict[str, int] = {nid: 0 for nid in node_ids}
            sigma[s] = 1  # 자기 자신까지의 경로는 1개

            # dist: s에서 각 노드까지의 최단 거리 (-1 = 아직 미방문)
            dist: Dict[str, int] = {nid: -1 for nid in node_ids}
            dist[s] = 0

            # ── BFS 실행 ──
            # BFS = 가까운 노드부터 차례대로 방문하는 탐색 방법
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)

                # v에서 갈 수 있는 모든 이웃 노드 탐색
                for w in adjacency.get(v, []):
                    # 처음 방문하는 노드
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)

                    # w가 v를 통해 최단 경로에 있는 경우
                    if dist[w] == dist[v] + 1:
                        # s→v→w 경로 수 = s→v 경로 수
                        sigma[w] += sigma[v]
                        # v는 w의 직전 노드(predecessor)
                        predecessors[w].append(v)

            # ── 역방향 누적 (Back-propagation of dependencies) ──
            # BFS에서 먼 노드부터 가까운 노드 순으로 처리합니다.
            #
            # delta[v] = v를 경유하는 경로의 비율 누적값
            delta: Dict[str, float] = {nid: 0.0 for nid in node_ids}

            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    # "s→v→w 경로 비율" = sigma[v] / sigma[w]
                    # 이 비율에 w 이후의 기여분(delta[w])을 더해서 v에 누적
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])

                # s 자신은 제외하고 중심성에 더함
                if w != s:
                    bc[w] += delta[w]

        # ── 정규화 ──
        # 방향 그래프에서의 정규화: 1 / ((N-1)(N-2))
        # 값을 0~1 범위로 변환합니다
        norm = (n - 1) * (n - 2) if n > 2 else 1.0
        for nid in bc:
            bc[nid] /= norm

        return bc

    def _extract_session_topics(self) -> List[str]:
        """
        RAKE 알고리즘으로 페이지 제목에서 키워드를 추출합니다.

        ═══════════════════════════════════════════════
        RAKE (Rapid Automatic Keyword Extraction)란?
        ═══════════════════════════════════════════════

        텍스트에서 중요한 키워드를 자동으로 뽑아내는 알고리즘입니다.

        동작 원리:
          1. 텍스트를 단어로 분리
          2. 불용어(stopwords: the, is, at 등 의미 없는 단어) 제거
          3. 남은 단어들의 동시 출현 빈도로 점수 계산
          4. 점수 ≥ 1인 키워드만 선택

        논문에서의 활용:
          봇과 사람의 세션에서 추출된 키워드를 비교하여
          탐색 주제의 차이를 분석합니다.
          - 사람: 특정 주제 키워드에 집중 ("laptop", "gaming")
          - 봇: 매우 다양하거나 무관한 키워드 분포

        Returns:
            score ≥ 1인 키워드 리스트
        """
        # 모든 페이지 제목을 하나의 텍스트로 합치기
        titles = [
            node.page_title
            for node in self.nodes.values()
            if node.page_title
        ]

        if not titles:
            return []

        combined_text = " ".join(titles)

        try:
            # rake-nltk 라이브러리 사용
            from rake_nltk import Rake

            rake = Rake()
            rake.extract_keywords_from_text(combined_text)

            # score ≥ 1인 키워드만 필터링
            # get_ranked_phrases_with_scores() → [(score, phrase), ...]
            scored_phrases = rake.get_ranked_phrases_with_scores()
            keywords = [
                phrase for score, phrase in scored_phrases if score >= 1.0
            ]
            return keywords

        except ImportError:
            # rake-nltk가 설치되지 않은 경우 간단한 폴백
            # 단어 빈도 기반으로 키워드 추출
            words = combined_text.lower().split()
            # 불용어 제거 (간단한 영어 불용어)
            stopwords = {
                "the", "a", "an", "is", "are", "was", "were", "to", "of",
                "in", "for", "on", "and", "or", "at", "by", "with", "&",
            }
            keywords = [w for w in words if w not in stopwords and len(w) > 2]
            # 중복 제거
            return list(dict.fromkeys(keywords))

    # ══════════════════════════════════════════════
    # 인접 행렬 변환 (DGCNN 입력용)
    # ══════════════════════════════════════════════

    def to_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        WT Graph를 인접 행렬(adjacency matrix)로 변환합니다.

        ═══════════════════════════════════════════════
        인접 행렬이란?
        ═══════════════════════════════════════════════

        그래프의 연결 관계를 2차원 표(행렬)로 표현한 것입니다.

        예시: A → B (가중치 2), B → C (가중치 1)

              A  B  C
          A [ 0, 2, 0 ]    A→B 연결(가중치 2)
          B [ 0, 0, 1 ]    B→C 연결(가중치 1)
          C [ 0, 0, 0 ]    C에서 나가는 엣지 없음

        행(row) = 출발 노드
        열(col) = 도착 노드
        값      = 엣지 가중치 (0이면 연결 없음)

        Returns:
            (adj_matrix, node_list)
            - adj_matrix: N x N numpy 배열 (N = 노드 수)
            - node_list: 노드 ID 리스트 (행렬의 인덱스 순서)
        """
        node_list = list(self.nodes.keys())
        n = len(node_list)

        # 노드 ID → 행렬 인덱스 매핑 (빠른 조회용)
        node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}

        # N x N 영행렬 생성
        adj = np.zeros((n, n), dtype=np.float32)

        # 각 엣지를 행렬에 기록
        for edge in self.edges.values():
            i = node_to_idx.get(edge.source)
            j = node_to_idx.get(edge.target)
            if i is not None and j is not None:
                adj[i][j] = edge.weight

        return adj, node_list

    # ══════════════════════════════════════════════
    # 유틸리티 메서드
    # ══════════════════════════════════════════════

    def get_summary(self) -> dict:
        """그래프의 간단한 요약 정보를 반환합니다."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "total_hits": sum(n.hit_count for n in self.nodes.values()),
            "first_page": self.first_hit_pagename,
        }

    def to_dict(self) -> dict:
        """
        그래프 상태를 딕셔너리로 직렬화합니다.
        API 응답이나 시각화에 사용됩니다.
        """
        return {
            "nodes": [
                {
                    "id": node.id,
                    "pageType": node.page_type,
                    "hitCount": node.hit_count,
                    "pageTitle": node.page_title,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                }
                for edge in self.edges.values()
            ],
            "summary": self.get_summary(),
        }
