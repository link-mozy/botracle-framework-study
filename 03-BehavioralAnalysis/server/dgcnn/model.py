"""
BOTracle 3단계 - DGCNN 모델 (Deep Graph Convolutional Neural Network)

이 파일은 BOTracle 논문 Section 4.2의 DGCNN 아키텍처를 구현합니다.
머신러닝 초급자를 위해 모든 개념에 상세한 주석을 달았습니다.

═══════════════════════════════════════════════════════════════
  DGCNN이 뭔가요? (비유로 설명)
═══════════════════════════════════════════════════════════════

  DGCNN은 "그래프 데이터"를 분류하는 딥러닝 모델입니다.

  비유: 도시 지도를 보고 그 도시의 특성을 판단하는 전문가

  1단계 - GCN (Graph Convolution):
    "마을 주민들이 이웃과 정보를 교환하는 과정"

    각 노드(주민)가 이웃 노드의 정보를 모아서 자신의 정보를 업데이트합니다.
    이것을 4번 반복하면 마을 전체의 분위기를 각 주민이 파악하게 됩니다.

    → 결과: 각 노드가 주변 구조를 반영한 특징 벡터를 갖게 됩니다.

  2단계 - SortPooling:
    "가장 중요한 주민 35명을 선발하는 과정"

    그래프 크기가 제각각이므로, 일정한 크기로 통일해야 합니다.
    GCN 점수가 높은 상위 35개 노드만 선택합니다.

    → 결과: 모든 그래프가 동일한 크기(35×97)로 변환됩니다.

  3단계 - 1D-CNN:
    "선발된 주민들의 정보를 스캔하여 패턴을 찾는 과정"

    SortPooling으로 정렬된 특징을 1차원 신호로 보고,
    컨볼루션 필터로 패턴을 감지합니다.

    → 결과: "봇일 확률" 0~1 사이의 값


═══════════════════════════════════════════════════════════════
  전체 구조 (논문 Section 4.2)
═══════════════════════════════════════════════════════════════

  WT Graph (인접 행렬 + 노드 특징)
       │
       ▼
  ┌─────────────────────────────┐
  │  GCN Layer 1: input → 32   │  tanh 활성화
  │  GCN Layer 2: 32 → 32      │  tanh 활성화
  │  GCN Layer 3: 32 → 32      │  tanh 활성화
  │  GCN Layer 4: 32 → 1       │  tanh 활성화
  └──────────┬──────────────────┘
             │ 모든 레이어 출력 연결 → 노드당 32+32+32+1 = 97차원
             ▼
  ┌─────────────────────────────┐
  │  SortPooling (k=35)         │  상위 35개 노드 선택
  │  출력 크기: 35 × 97 = 3395 │
  └──────────┬──────────────────┘
             │ 1차원으로 펼침 → (1, 3395)
             ▼
  ┌─────────────────────────────┐
  │  1D-Conv1: 16 filters       │  kernel=97, stride=97 (노드 단위 스캔)
  │  1D-Conv2: 32 filters       │  kernel=5, stride=1
  │  MaxPool: pool_size=2       │
  └──────────┬──────────────────┘
             │ 평탄화(flatten)
             ▼
  ┌─────────────────────────────┐
  │  Dense: 128 units, ReLU     │
  │  Dropout: p=0.5             │
  │  Output: 1 unit, Sigmoid    │
  └─────────────────────────────┘
             │
             ▼
       봇 확률 (0~1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    그래프 컨볼루션 레이어 (Graph Convolution Layer)

    ═══════════════════════════════════════════════
    이 레이어가 하는 일
    ═══════════════════════════════════════════════

    각 노드가 "이웃 노드들의 정보를 모아서" 자신의 정보를 갱신합니다.

    비유: 반(class)에서 정보를 수집하는 학생

      1. 주변 친구들(이웃 노드)에게 정보를 물어봄 → 집계 (aggregate)
      2. 수집한 정보를 정리하여 자기만의 의견으로 변환 → 선형 변환 (linear)
      3. 최종 의견에 판단 기준 적용 → 활성화 함수 (tanh)

    수식:
      h_v^(l+1) = tanh( W^(l) × Σ_{u ∈ N(v)} h_u^(l) )

      설명:
        h_v^(l)  = l번째 레이어에서 노드 v의 특징 벡터
        N(v)     = 노드 v의 이웃 노드 집합
        Σ        = 이웃들의 특징을 모두 합산 (집계)
        W^(l)    = l번째 레이어의 학습 가능한 가중치 행렬
        tanh     = 활성화 함수 (출력을 -1~1 범위로 조정)

    행렬 연산으로 표현:
      H^(l+1) = tanh( A × H^(l) × W^(l) )
      여기서 A = 인접 행렬 (어떤 노드가 이웃인지 나타내는 행렬)

    Args:
        in_features:  입력 특징 차원 수
        out_features: 출력 특징 차원 수
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # 선형 변환 레이어 (학습 가능한 가중치 W)
        # 수집한 이웃 정보를 새로운 차원으로 변환합니다
        #
        # 예: in_features=12 → out_features=32
        #     12차원 정보를 32차원으로 확장 (더 풍부한 표현)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass)

        Args:
            x:   노드 특징 행렬 (batch × N × in_features)
                 batch = 한 번에 처리하는 그래프 수
                 N     = 노드 수
            adj: 인접 행렬 (batch × N × N)
                 adj[i][j] > 0 이면 노드 i → j 연결됨

        Returns:
            새로운 노드 특징 행렬 (batch × N × out_features)

        연산 과정:
          1. adj × x = 이웃 노드의 특징을 합산 (집계)
             행렬 곱으로 한 번에 모든 노드의 이웃 정보를 모읍니다.

             예시 (3노드, A→B, A→C):
               adj = [[0,1,1],    x = [[0.5, 0.3],  # A의 특징
                      [0,0,0],         [0.8, 0.1],  # B의 특징
                      [0,0,0]]         [0.2, 0.9]]  # C의 특징

               adj × x = [[1.0, 1.0],  # A: B+C의 특징 합산
                          [0.0, 0.0],  # B: 이웃 없음
                          [0.0, 0.0]]  # C: 이웃 없음

          2. linear(집계 결과) = 가중치 W를 곱하여 차원 변환
        """
        # 이웃 노드 특징 집계: 인접 행렬 × 노드 특징 행렬
        # torch.matmul은 배치 차원을 자동으로 처리합니다
        h = torch.matmul(adj, x)  # (batch, N, in_features)

        # 선형 변환: 집계된 정보를 새로운 표현으로 변환
        return self.linear(h)     # (batch, N, out_features)


class DGCNN(nn.Module):
    """
    Deep Graph Convolutional Neural Network

    논문 Section 4.2 기반 구현.
    WT Graph를 입력받아 봇/사람을 이진 분류합니다.

    구조: GCN (4 layers) → SortPooling (k=35) → 1D-CNN → Dense → Sigmoid
    """

    # ══════════════════════════════════════════════
    # 논문 하이퍼파라미터 (Section 4.2)
    # ══════════════════════════════════════════════
    # 이 값들은 논문에서 실험적으로 최적화된 값입니다.
    # 변경하면 성능이 달라질 수 있습니다.

    GCN_HIDDEN_UNITS = [32, 32, 32, 1]  # 4개 GCN 레이어의 출력 차원
    SORT_POOL_K = 35                     # SortPooling에서 선택할 상위 노드 수
    LEARNING_RATE = 0.0001               # Adam 옵티마이저 학습률
    DROPOUT_RATE = 0.5                   # Dropout 비율

    def __init__(self, input_dim: int, k: int = 35):
        """
        Args:
            input_dim: 노드 특징 벡터의 차원 수
                       (preprocessor에서 생성하는 특징 벡터의 크기)
            k:         SortPooling에서 선택할 상위 노드 수 (논문: 35)
        """
        super().__init__()
        self.k = k

        # ══════════════════════════════════════════════
        # 1단계: GCN 레이어 (Graph Convolution Network)
        # ══════════════════════════════════════════════
        #
        # 4개의 GCN 레이어를 순서대로 통과합니다.
        # 각 레이어는 이웃 노드 정보를 더 넓은 범위에서 수집합니다.
        #
        # 비유: 소문이 퍼져나가는 과정
        #   1번째 레이어: 바로 옆 이웃의 정보만 수집 (1-hop)
        #   2번째 레이어: 이웃의 이웃 정보까지 수집 (2-hop)
        #   3번째 레이어: 3단계 떨어진 노드 정보까지 (3-hop)
        #   4번째 레이어: 4단계 떨어진 노드 정보까지 (4-hop)
        #
        # hidden units: 32, 32, 32, 1
        #   - 처음 3개 레이어: 32차원으로 풍부한 특징 학습
        #   - 마지막 레이어: 1차원으로 압축 (SortPooling 정렬 기준)

        self.conv1 = GraphConvLayer(input_dim, 32)  # 입력 → 32차원
        self.conv2 = GraphConvLayer(32, 32)          # 32 → 32차원
        self.conv3 = GraphConvLayer(32, 32)          # 32 → 32차원
        self.conv4 = GraphConvLayer(32, 1)           # 32 → 1차원

        # ══════════════════════════════════════════════
        # GCN 전체 hidden units 합계
        # ══════════════════════════════════════════════
        # SortPooling 후 각 노드의 특징은 모든 GCN 레이어의 출력을 연결(concat)합니다.
        # 따라서 각 노드의 최종 특징 차원 = 32 + 32 + 32 + 1 = 97
        total_gcn_units = sum(self.GCN_HIDDEN_UNITS)  # = 97

        # ══════════════════════════════════════════════
        # 3단계: 1D-CNN (1차원 컨볼루션 신경망)
        # ══════════════════════════════════════════════
        #
        # SortPooling의 출력을 1차원 신호로 보고 패턴을 감지합니다.
        #
        # 비유: 바코드 스캐너
        #   SortPooling 결과 = 중요한 노드 35개의 특징을 일렬로 나열한 "바코드"
        #   1D-CNN = 이 바코드를 스캔하여 봇/사람 패턴을 감지
        #
        # ── 1D-Conv Layer 1 ──
        # 16개 필터, 커널 크기 = 97, 스트라이드 = 97
        #
        # 커널 크기가 97인 이유:
        #   각 노드의 특징이 97차원이므로, 커널이 정확히 "한 노드"를 한 번에 처리합니다.
        #   stride=97이므로 한 칸 이동할 때마다 다음 노드로 넘어갑니다.
        #
        #   시각화:
        #     [노드1의 97차원][노드2의 97차원][노드3의 97차원]...
        #     ├── 커널 ──┤  (97칸을 한 번에 스캔)
        #                ├── 커널 ──┤  (stride=97으로 다음 노드로 이동)

        self.conv1d_1 = nn.Conv1d(
            in_channels=1,               # 입력 채널 = 1 (1차원 신호)
            out_channels=16,              # 16개 필터 (16가지 패턴 감지)
            kernel_size=total_gcn_units,  # 97 (한 노드의 전체 특징)
            stride=total_gcn_units,       # 97 (노드 단위로 이동)
        )

        # ── 1D-Conv Layer 2 ──
        # 32개 필터, 커널 크기 = 5, 스트라이드 = 1
        #
        # 이전 레이어의 출력(16채널)에서 더 넓은 범위의 패턴을 감지합니다.
        # 커널 크기 5 = 연속된 5개 노드의 패턴을 한 번에 확인
        self.conv1d_2 = nn.Conv1d(
            in_channels=16,   # 이전 레이어의 출력 채널 수
            out_channels=32,  # 32개 필터
            kernel_size=5,    # 5개 위치를 한 번에 확인
            stride=1,         # 1칸씩 이동
        )

        # ── MaxPooling ──
        # 연속된 2개 값 중 큰 값만 선택하여 크기를 절반으로 줄입니다.
        #
        # 비유: 시험 답안지에서 2문제씩 묶어 높은 점수만 남기기
        #   → 중요한 정보는 유지하면서 데이터 크기를 줄임 (차원 축소)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # ══════════════════════════════════════════════
        # Dense 레이어 입력 크기 계산
        # ══════════════════════════════════════════════
        # 각 레이어를 통과할 때마다 데이터 크기가 어떻게 변하는지 추적합니다.
        #
        # SortPooling 출력: k × 97 = 35 × 97 = 3395
        # Conv1d_1 후: (3395 - 97) / 97 + 1 = 35  (노드 수와 동일!)
        # Conv1d_2 후: (35 - 5) / 1 + 1 = 31
        # MaxPool 후: 31 // 2 = 15
        # Flatten 후: 32 × 15 = 480
        conv1d_1_out = (k * total_gcn_units - total_gcn_units) // total_gcn_units + 1
        conv1d_2_out = conv1d_1_out - 5 + 1
        maxpool_out = conv1d_2_out // 2
        dense_input = 32 * maxpool_out

        # ── Dense Layer ──
        # 128개 뉴런, ReLU 활성화
        #
        # Dense(완전연결층) = 모든 입력을 모든 뉴런에 연결
        # 여기서 최종적인 봇/사람 판단을 위한 고수준 특징을 학습합니다
        self.fc = nn.Linear(dense_input, 128)

        # ── Dropout ──
        # 학습 중 50%의 뉴런을 무작위로 끔 → 과적합 방지
        #
        # 과적합(overfitting)이란?
        #   학습 데이터는 잘 맞추는데 새로운 데이터는 못 맞추는 현상
        #   비유: 시험 기출문제만 외워서 시험은 잘 보는데, 응용문제는 못 푸는 것
        #
        # Dropout의 효과:
        #   매번 다른 뉴런 조합으로 학습하므로
        #   특정 뉴런에 과도하게 의존하지 않게 됩니다
        self.dropout = nn.Dropout(p=self.DROPOUT_RATE)

        # ── Output Layer ──
        # 1개 뉴런, Sigmoid 활성화
        #
        # Sigmoid: 어떤 값이든 0~1 사이로 변환
        #   sigmoid(x) = 1 / (1 + e^(-x))
        #   → 0에 가까우면 "사람", 1에 가까우면 "봇"
        self.output = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass): 그래프를 입력받아 봇 확률을 출력합니다.

        Args:
            x:   노드 특징 행렬 (batch × N × input_dim)
                 batch     = 한 번에 처리하는 그래프 수
                 N         = 노드 수 (그래프마다 다를 수 있지만, 배치 내에서는 패딩으로 통일)
                 input_dim = 노드 특징 차원 수
            adj: 인접 행렬 (batch × N × N)

        Returns:
            봇 확률 (batch × 1), 범위 [0, 1]
              0에 가까우면 → 사람
              1에 가까우면 → 봇
        """
        batch_size = x.size(0)

        # ═════════════════════════════════════════
        # 1단계: GCN - 이웃 정보 수집 (4 레이어)
        # ═════════════════════════════════════════
        #
        # 각 레이어에서:
        #   1. 인접 행렬을 사용해 이웃 특징 집계
        #   2. 선형 변환으로 새로운 표현 생성
        #   3. tanh 활성화 적용
        #
        # tanh(x)는 출력을 -1 ~ +1 범위로 조정합니다.
        # 양수 = "이 특성이 있다", 음수 = "이 특성이 없다"

        h1 = torch.tanh(self.conv1(x, adj))    # (batch, N, 32)
        h2 = torch.tanh(self.conv2(h1, adj))   # (batch, N, 32)
        h3 = torch.tanh(self.conv3(h2, adj))   # (batch, N, 32)
        h4 = torch.tanh(self.conv4(h3, adj))   # (batch, N, 1)

        # 모든 GCN 레이어의 출력을 연결 (concatenate)
        # 각 노드가 "1-hop 정보 + 2-hop 정보 + 3-hop 정보 + 4-hop 정보"를
        # 모두 가진 97차원 벡터가 됩니다
        h = torch.cat([h1, h2, h3, h4], dim=-1)  # (batch, N, 97)

        # ═════════════════════════════════════════
        # 2단계: SortPooling
        # ═════════════════════════════════════════
        #
        # 문제: 그래프마다 노드 수(N)가 다름 → 통일된 크기 필요
        # 해결: 가장 "중요한" 상위 k개 노드만 선택
        #
        # "중요도" 기준: 마지막 GCN 레이어(h4)의 출력값
        #   → h4의 값이 클수록 그래프에서 더 중요한 노드
        #
        # 비유: 반에서 시험 점수(h4) 기준으로 상위 35명을 선발하는 것

        # h4의 값을 기준으로 내림차순 정렬
        sort_scores = h4.squeeze(-1)  # (batch, N) - 각 노드의 점수

        # 노드 수가 k보다 적은 경우를 처리
        # 부족한 만큼 0으로 채워진 가짜 노드를 추가 (제로 패딩)
        current_n = sort_scores.size(1)
        if current_n < self.k:
            # 노드 수가 k보다 적으면 제로 패딩
            pad_size = self.k - current_n
            # 점수에 매우 작은 값으로 패딩 (정렬 시 뒤로 밀림)
            score_pad = torch.full(
                (batch_size, pad_size), -1e9, device=x.device
            )
            sort_scores = torch.cat([sort_scores, score_pad], dim=1)

            # h(특징)에도 0으로 패딩
            h_pad = torch.zeros(
                batch_size, pad_size, h.size(-1), device=x.device
            )
            h = torch.cat([h, h_pad], dim=1)

        # 점수 기준 내림차순 정렬
        _, sort_idx = sort_scores.sort(dim=-1, descending=True)

        # 상위 k개 노드만 선택
        sort_idx = sort_idx[:, :self.k]  # (batch, k)

        # 선택된 노드의 특징을 가져오기
        sort_idx_expanded = sort_idx.unsqueeze(-1).expand(
            -1, -1, h.size(-1)
        )
        h = torch.gather(h, 1, sort_idx_expanded)  # (batch, k, 97)

        # ═════════════════════════════════════════
        # 3단계: 1D-CNN
        # ═════════════════════════════════════════
        #
        # SortPooling 결과를 1차원 신호로 변환하여 CNN 처리

        # (batch, k, 97) → (batch, 1, k*97) : 2차원을 1차원으로 펼침
        h = h.reshape(batch_size, 1, -1)  # (batch, 1, 3395)

        # 1D-Conv Layer 1: 노드 단위로 패턴 감지
        h = F.relu(self.conv1d_1(h))  # (batch, 16, k)

        # 1D-Conv Layer 2: 인접 노드 그룹의 패턴 감지
        h = F.relu(self.conv1d_2(h))  # (batch, 32, k-4)

        # MaxPooling: 크기를 절반으로 축소
        h = self.maxpool(h)           # (batch, 32, (k-4)//2)

        # ═════════════════════════════════════════
        # Dense + Sigmoid: 최종 분류
        # ═════════════════════════════════════════

        # 평탄화 (Flatten): 다차원 → 1차원
        h = h.reshape(batch_size, -1)  # (batch, 480)

        # Dense 레이어: 고수준 특징 학습
        h = F.relu(self.fc(h))         # (batch, 128)

        # Dropout: 과적합 방지 (학습 시에만 활성화)
        h = self.dropout(h)

        # Sigmoid 출력: 봇 확률
        out = torch.sigmoid(self.output(h))  # (batch, 1)

        return out


# ══════════════════════════════════════════════════════════════
# DGCNN 래퍼 클래스 (학습·예측·저장·로드를 한 곳에서 관리)
# ══════════════════════════════════════════════════════════════

class DGCNNClassifier:
    """
    DGCNN 모델을 감싸서 학습, 예측, 저장, 로드를 편리하게 해주는 클래스.

    2단계의 SGAN 클래스와 동일한 역할입니다.

    사용법:
        classifier = DGCNNClassifier(input_dim=12)
        classifier.train(adj_list, features_list, labels, epochs=100)
        prediction, confidence = classifier.predict(adj, features)
    """

    def __init__(self, input_dim: int = 12, k: int = 35):
        """
        Args:
            input_dim: 노드 특징 벡터의 차원 수 (기본: 12)
            k:         SortPooling에서 선택할 상위 노드 수 (기본: 35)
        """
        self.input_dim = input_dim
        self.k = k

        # DGCNN 모델 생성
        self.model = DGCNN(input_dim=input_dim, k=k)

        # ══════════════════════════════════════════════
        # 옵티마이저 (Optimizer)
        # ══════════════════════════════════════════════
        # "학습 중 가중치를 어떻게 조정할지"를 결정하는 알고리즘
        #
        # Adam: 현재 가장 널리 쓰이는 옵티마이저
        #   learning_rate = 0.0001 (매우 조심스럽게 조정)
        #   논문 Section 4.2에서 이 값을 사용합니다
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=DGCNN.LEARNING_RATE,
        )

        # ══════════════════════════════════════════════
        # 손실 함수 (Loss Function)
        # ══════════════════════════════════════════════
        # "예측이 정답과 얼마나 다른지"를 측정하는 함수
        #
        # Binary Cross-Entropy (이진 교차 엔트로피):
        #   이진 분류(봇/사람) 문제에 사용
        #   - 정답=1(봇)인데 0.9로 예측 → 손실 작음 (잘 맞춤)
        #   - 정답=1(봇)인데 0.1로 예측 → 손실 큼 (크게 틀림)
        self.criterion = nn.BCELoss()

    def train(
        self,
        adj_list: list,
        features_list: list,
        labels,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> dict:
        """
        DGCNN 모델을 학습시킵니다.

        ═══════════════════════════════════════════════
        학습 과정 요약
        ═══════════════════════════════════════════════

        매 에폭(epoch)마다:
          1. 배치 단위로 데이터를 나눔
          2. 각 배치에 대해:
             a. 모델에 입력 → 봇 확률 예측
             b. 예측과 정답의 차이(손실) 계산
             c. 손실을 줄이는 방향으로 가중치 조정 (역전파)
          3. 전체 손실 기록

        Args:
            adj_list:       인접 행렬 리스트 (각 원소: numpy array [N, N])
            features_list:  노드 특징 행렬 리스트 (각 원소: numpy array [N, F])
            labels:         라벨 배열 (0=사람, 1=봇)
            epochs:         전체 데이터를 몇 번 반복 학습할지
            batch_size:     한 번에 처리할 그래프 수

        Returns:
            학습 이력 (dict): epoch별 손실값 기록
        """
        import numpy as np

        self.model.train()  # 학습 모드 활성화 (Dropout 활성화)

        n_samples = len(labels)
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            # 매 에폭마다 데이터 순서를 셔플 (무작위 섞기)
            # → 학습이 특정 순서에 치우치지 않도록
            indices = np.random.permutation(n_samples)

            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0

            # 배치 단위로 학습
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                # ── 배치 데이터 준비 ──
                # 그래프마다 노드 수가 다르므로, 배치 내 최대 노드 수에 맞춰
                # 제로 패딩(zero padding)을 적용합니다.
                batch_adj, batch_feat, batch_labels = self._collate_batch(
                    [adj_list[i] for i in batch_idx],
                    [features_list[i] for i in batch_idx],
                    [labels[i] for i in batch_idx],
                )

                # ── 순전파 (Forward) ──
                # 모델에 입력하여 봇 확률 예측
                predictions = self.model(batch_feat, batch_adj)  # (batch, 1)

                # ── 손실 계산 ──
                loss = self.criterion(predictions, batch_labels)

                # ── 역전파 (Backward) + 가중치 업데이트 ──
                # 1. 이전 기울기 초기화 (안 하면 기울기가 누적됨)
                self.optimizer.zero_grad()
                # 2. 손실로부터 각 가중치의 기울기(gradient) 계산
                loss.backward()
                # 3. 기울기 방향으로 가중치 조정
                self.optimizer.step()

                # 통계 기록
                epoch_loss += loss.item()
                predicted_labels = (predictions > 0.5).float()
                epoch_correct += (
                    predicted_labels == batch_labels
                ).sum().item()
                n_batches += 1

            # 에폭 결과 기록
            avg_loss = epoch_loss / max(n_batches, 1)
            accuracy = epoch_correct / n_samples
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            # 10 에폭마다 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Accuracy: {accuracy:.4f}"
                )

        return history

    def _collate_batch(
        self, adj_list: list, features_list: list, labels_list: list
    ):
        """
        가변 크기 그래프들을 하나의 배치로 합칩니다.

        문제: 각 그래프의 노드 수가 다름 (예: 5개, 10개, 3개)
        해결: 배치 내 최대 노드 수에 맞춰 제로 패딩

        비유: 크기가 다른 사진들을 같은 크기의 액자에 넣기
              작은 사진은 빈 공간을 검은색(0)으로 채움

        예시:
          그래프A: 5노드, 그래프B: 3노드 → 최대 5노드로 통일
          그래프B의 인접 행렬: 3×3 → 5×5로 확장 (나머지 0)
          그래프B의 특징 행렬: 3×F → 5×F로 확장 (나머지 0)
        """
        import numpy as np

        batch_size = len(adj_list)

        # 배치 내 최대 노드 수 찾기
        max_n = max(adj.shape[0] for adj in adj_list)
        # 최소 k개는 되어야 SortPooling이 동작
        max_n = max(max_n, self.k)

        feature_dim = features_list[0].shape[1]

        # 패딩된 배치 텐서 생성
        padded_adj = np.zeros((batch_size, max_n, max_n), dtype=np.float32)
        padded_feat = np.zeros(
            (batch_size, max_n, feature_dim), dtype=np.float32
        )

        for i in range(batch_size):
            n = adj_list[i].shape[0]
            padded_adj[i, :n, :n] = adj_list[i]
            padded_feat[i, :n, :] = features_list[i]

        # numpy → PyTorch 텐서 변환
        batch_adj = torch.tensor(padded_adj, dtype=torch.float32)
        batch_feat = torch.tensor(padded_feat, dtype=torch.float32)
        batch_labels = torch.tensor(
            [[float(l)] for l in labels_list], dtype=torch.float32
        )

        return batch_adj, batch_feat, batch_labels

    def predict(self, adj, features) -> tuple:
        """
        학습된 모델로 봇/사람을 예측합니다.

        Args:
            adj:      인접 행렬 (numpy array, shape: [N, N])
            features: 노드 특징 행렬 (numpy array, shape: [N, F])

        Returns:
            (prediction, confidence)
            prediction: "bot" 또는 "human"
            confidence: 예측 신뢰도 (0.0 ~ 1.0)
        """
        self.model.eval()  # 예측 모드 (Dropout 비활성화)

        with torch.no_grad():  # 기울기 계산 비활성화 (메모리 절약)
            # 단일 그래프를 배치 형태로 변환 (배치 크기 = 1)
            batch_adj, batch_feat, _ = self._collate_batch(
                [adj], [features], [0]
            )

            # 모델 추론
            prob = self.model(batch_feat, batch_adj)  # (1, 1)
            bot_prob = prob.item()  # 0~1 사이 봇 확률

        # 0.5 기준으로 봇/사람 판정
        if bot_prob >= 0.5:
            return "bot", bot_prob
        else:
            return "human", 1.0 - bot_prob

    def save_weights(self, path: str):
        """모델 가중치를 파일로 저장합니다."""
        import os
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "dgcnn_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"모델 가중치 저장 완료: {save_path}")

    def load_weights(self, path: str):
        """저장된 가중치를 불러옵니다."""
        import os
        load_path = os.path.join(path, "dgcnn_model.pth")
        self.model.load_state_dict(
            torch.load(load_path, weights_only=True)
        )
        print(f"모델 가중치 로드 완료: {load_path}")


# ══════════════════════════════════════════════════════════════
# 사용 예시
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 하이퍼파라미터
    input_dim = 12   # 노드 특징 차원 (메트릭 수)
    k = 35           # SortPooling에서 선택할 상위 노드 수
    batch_size = 4
    num_nodes = 50   # 예시 그래프의 노드 수

    model = DGCNN(input_dim=input_dim, k=k)

    # 더미 입력 생성
    x = torch.randn(batch_size, num_nodes, input_dim)       # 노드 특징
    adj = torch.randint(
        0, 2, (batch_size, num_nodes, num_nodes)
    ).float()  # 인접 행렬

    # 추론
    prob = model(x, adj)
    print(f"봇 확률: {prob.squeeze()}")
    # 예: tensor([0.4821, 0.5103, 0.4997, 0.5234])
    # λ와 비교하여 최종 판정
