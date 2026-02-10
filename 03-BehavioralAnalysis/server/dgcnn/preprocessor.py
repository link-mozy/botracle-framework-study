"""
BOTracle 3단계 - 그래프 전처리기 (Graph Preprocessor)

WT Graph를 DGCNN 모델이 이해할 수 있는 텐서(인접 행렬 + 노드 특징 행렬)로 변환합니다.

2단계의 전처리기(preprocessor.py)가 원시 JSON → 11차원 수치 벡터를 만들었다면,
3단계의 전처리기는 WT Graph → (인접 행렬, 노드 특징 행렬)을 만듭니다.

═══════════════════════════════════════════════════════════════
  변환 과정
═══════════════════════════════════════════════════════════════

  WT Graph (방향 그래프)
       │
       ▼
  1. 인접 행렬 생성 (N × N)
       - 방향 그래프 → 비방향 그래프로 대칭화
       - 엣지 가중치를 0~1 범위로 정규화
       - 자기 자신과의 연결(self-loop) 추가
       │
       ▼
  2. 노드 특징 행렬 생성 (N × 12)
       - 각 노드에서 추출한 메트릭을 특징 벡터로 조합
       │
       ▼
  DGCNN 모델 입력
    - adj: (N × N) 인접 행렬
    - features: (N × 12) 노드 특징 행렬

═══════════════════════════════════════════════════════════════
  노드 특징 벡터 구조 (12차원)
═══════════════════════════════════════════════════════════════

  인덱스   특징               범위      설명
  ──────  ─────────────────  ────────  ──────────────────────
  [0]     hit_count          [0, 1]   이 페이지 방문 횟수 (정규화)
  [1]     node_degree        [0, 1]   연결된 엣지 수 (정규화)
  [2]     degree_centrality  [0, 1]   연결 중심성
  [3]     betweenness_cent.  [0, 1]   매개 중심성
  [4]     page_type: home    {0, 1}   ─┐
  [5]     page_type: product {0, 1}    │
  [6]     page_type: category{0, 1}    │ 페이지 유형
  [7]     page_type: search  {0, 1}    │ One-Hot
  [8]     page_type: cart    {0, 1}    │ 인코딩
  [9]     page_type: checkout{0, 1}    │ (8차원)
  [10]    page_type: account {0, 1}    │
  [11]    page_type: other   {0, 1}   ─┘
"""

import numpy as np

from dgcnn.wt_graph import WTGraph, PAGE_TYPES


class GraphPreprocessor:
    """
    WT Graph를 DGCNN 입력 텐서로 변환하는 전처리기.

    사용법:
        preprocessor = GraphPreprocessor()
        adj, features = preprocessor.transform(wt_graph)
        # adj: numpy array (N, N)
        # features: numpy array (N, 12)
    """

    # 노드 특징 벡터의 차원 수
    # 4 (수치 메트릭) + 8 (페이지 유형 One-Hot) = 12
    FEATURE_DIM = 4 + len(PAGE_TYPES)  # = 12

    def __init__(self):
        self.feature_dim = self.FEATURE_DIM

    def transform(self, wt_graph: WTGraph):
        """
        WT Graph를 DGCNN 입력 형식으로 변환합니다.

        Args:
            wt_graph: WTGraph 객체

        Returns:
            (adj_matrix, node_features)
            - adj_matrix:    정규화된 비방향 인접 행렬 (numpy, shape: [N, N])
            - node_features: 노드 특징 행렬 (numpy, shape: [N, 12])
        """
        # ── 1. 메트릭 추출 ──
        metrics = wt_graph.extract_metrics()

        # ── 2. 인접 행렬 생성 + 전처리 ──
        adj_matrix = self._build_adjacency_matrix(wt_graph)

        # ── 3. 노드 특징 행렬 생성 ──
        node_features = self._build_node_features(wt_graph, metrics)

        return adj_matrix, node_features

    def _build_adjacency_matrix(self, wt_graph: WTGraph) -> np.ndarray:
        """
        인접 행렬을 생성하고 전처리합니다.

        3가지 전처리 단계:
          1. 비방향 대칭화: 방향 그래프 → 비방향 그래프
          2. 가중치 정규화: 값을 0~1 범위로 변환
          3. Self-loop 추가: 자기 자신과의 연결 추가

        ═══════════════════════════════════════════════
        왜 비방향 그래프로 변환하나요?
        ═══════════════════════════════════════════════

        논문 Section 4.2: "DGCNN operates on undirected graphs"
        DGCNN은 비방향 그래프에서 동작하도록 설계되었습니다.

        변환 방법:
          방향 그래프에서 A→B 엣지가 있으면
          비방향 그래프에서 A-B 양쪽 연결로 변환합니다.

          원래:      대칭화 후:
          A → B      A ↔ B
          (adj[0][1]=1, adj[1][0]=0)  →  (adj[0][1]=1, adj[1][0]=1)

        ═══════════════════════════════════════════════
        왜 Self-loop를 추가하나요?
        ═══════════════════════════════════════════════

        Self-loop = 자기 자신에게 연결 (adj[i][i] = 1)

        GCN에서 이웃 정보를 집계할 때, self-loop가 없으면
        자기 자신의 정보가 빠져버립니다.

        비유: 회의에서 다른 사람 의견만 듣고 자기 의견은 반영 안 하는 것
              → self-loop = "자기 의견도 포함시키기"
        """
        # 방향 그래프의 인접 행렬 가져오기
        adj, node_list = wt_graph.to_adjacency_matrix()
        n = len(node_list)

        if n == 0:
            return np.zeros((1, 1), dtype=np.float32)

        # ── 1단계: 비방향 대칭화 ──
        # adj[i][j]와 adj[j][i] 중 큰 값을 양쪽에 적용
        adj_symmetric = np.maximum(adj, adj.T)

        # ── 2단계: 가중치 정규화 ──
        # 값을 0~1 범위로 변환 (최대값으로 나누기)
        max_weight = adj_symmetric.max()
        if max_weight > 0:
            adj_symmetric = adj_symmetric / max_weight

        # ── 3단계: Self-loop 추가 ──
        # 대각선에 1 추가 (자기 자신과의 연결)
        np.fill_diagonal(adj_symmetric, 1.0)

        return adj_symmetric

    def _build_node_features(
        self, wt_graph: WTGraph, metrics: dict
    ) -> np.ndarray:
        """
        각 노드의 특징 벡터를 생성합니다.

        각 노드에 대해 12차원 특징 벡터를 만듭니다:
          [hit_count, degree, degree_cent, betweenness_cent,
           home, product, category, search, cart, checkout, account, other]

        모든 수치값은 0~1 범위로 정규화됩니다.
        """
        node_ids = list(wt_graph.nodes.keys())
        n = len(node_ids)

        if n == 0:
            return np.zeros((1, self.feature_dim), dtype=np.float32)

        features = np.zeros((n, self.feature_dim), dtype=np.float32)

        # 정규화를 위한 최대값 계산
        total_hits = metrics["total_hits"] or 1
        node_degrees = metrics["node_degrees"]
        max_degree = max(node_degrees.values()) if node_degrees else 1

        degree_centrality = metrics["degree_centrality"]
        betweenness_centrality = metrics["betweenness_centrality"]

        for i, node_id in enumerate(node_ids):
            node = wt_graph.nodes[node_id]

            # ── 수치 메트릭 (인덱스 0~3) ──

            # [0] 히트 수 (전체 대비 비율로 정규화)
            features[i, 0] = node.hit_count / total_hits

            # [1] 노드 차수 (최대 차수 대비 비율로 정규화)
            features[i, 1] = node_degrees.get(node_id, 0) / max(
                max_degree, 1
            )

            # [2] 연결 중심성 (이미 0~1 범위)
            features[i, 2] = degree_centrality.get(node_id, 0.0)

            # [3] 매개 중심성 (이미 0~1 범위)
            features[i, 3] = betweenness_centrality.get(node_id, 0.0)

            # ── 페이지 유형 One-Hot 인코딩 (인덱스 4~11) ──
            # 해당 페이지 유형의 인덱스만 1, 나머지 0
            #
            # 예: page_type = "product"
            #     PAGE_TYPES = ["home","product","category",...]
            #     "product"는 인덱스 1 → features[i, 4+1] = 1.0
            page_type = node.page_type
            if page_type in PAGE_TYPES:
                type_idx = PAGE_TYPES.index(page_type)
            else:
                type_idx = PAGE_TYPES.index("other")  # 기본값: other

            features[i, 4 + type_idx] = 1.0

        return features

    def transform_batch(self, wt_graphs: list):
        """
        여러 WT Graph를 한 번에 변환합니다.

        Args:
            wt_graphs: WTGraph 객체 리스트

        Returns:
            (adj_list, features_list)
            - adj_list:      인접 행렬 리스트
            - features_list: 노드 특징 행렬 리스트
        """
        adj_list = []
        features_list = []

        for graph in wt_graphs:
            adj, feat = self.transform(graph)
            adj_list.append(adj)
            features_list.append(feat)

        return adj_list, features_list
