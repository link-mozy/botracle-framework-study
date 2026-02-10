# Behavioral Analysis (행위 기반 분석)

## 프로젝트 개요

BOTracle 프레임워크의 3단계인 **행위 기반 봇 탐지** (DGCNN) 프로젝트입니다.
BOTracle 파이프라인에서 SGAN으로 분류되지 않은 요청(hit)을 세션 단위로 묶어 WT Graph를 생성하고, DGCNN 모델로 봇/사람을 판별합니다.

### 주요 기능

- 세션 단위 Website Traversal Graph (WT Graph) 구성
- WT Graph에서 그래프 특징(메트릭) 추출
- DGCNN을 통한 봇/사람 이진 분류

### 기술 스택

- 언어: TypeScript (클라이언트 WT Graph 데이터 수집), Python (서버, WT Graph 분석, DGCNN 분석)
- API 방식: RESTful API
- ML 프레임워크: PyTorch (DGCNN), RAKE (키워드 추출)

## BOTracle 파이프라인에서의 위치

```plaintext
  요청(Hit) 수신
       │
       ▼
┌──────────────┐    분류됨
│  휴리스틱      │──────────────▶ 봇/사람 판정
│  (1단계)      │
└──────┬───────┘
       │ 미분류
       ▼
┌──────────────┐    신뢰도 ≥ λ
│  SGAN        │──────────────▶ 봇/사람 판정
│  (2단계)      │
└──────┬───────┘
       │ 신뢰도 < λ
       ▼
┌──────────────┐    신뢰도 ≥ λ
│  DGCNN       │──────────────▶ 봇/사람 판정   ◀── 이 프로젝트
│  (3단계)      │
└──────────────┘
```

2단계 SGAN에서 신뢰도가 임계값 λ 미만인 hit들은 클라이언트 세션별로 누적됩니다. 이 세션 hit들을 WT Graph로 변환하고, 학습된 DGCNN에 입력합니다. DGCNN도 확률값을 λ와 비교하여:
- λ 이상이면 → 봇/사람 판정 완료
- λ 미만이면 → 추가 hit이 세션에 쌓일 때까지 대기 후 재분석

**핵심 차별점**: 3단계는 IP 주소, user agent, 윈도우 크기 등 기술적(비행위) 특징을 **전혀 사용하지 않고**, 오직 웹사이트 탐색 행동 패턴만으로 분류합니다. 봇이 기술적 특징을 위조하더라도 행동 패턴까지 완벽히 모사하기 어렵다는 가설에 기반합니다.

---

## WT Graph (Website Traversal Graph) 상세

### 개념

WT Graph는 사용자의 웹사이트 탐색 경로를 방향 그래프(directed graph)로 표현한 것입니다.

- **노드(Node)**: 웹사이트의 서브페이지(sub-page)
- **엣지(Edge)**: 한 페이지에서 다른 페이지로의 탐색 경로 (네비게이션 링크)

엣지에는 방문 빈도에 따른 가중치가 부여되고, 노드에는 관련 속성이 라벨링됩니다. 이렇게 구축된 그래프는 봇과 사람의 탐색 패턴 차이를 분석하기 위한 종합적 데이터 소스가 됩니다.

### 핵심 가설

자동화된 웹 봇은 사람과 구별되는 고유한 탐색 패턴을 보입니다:
- **봇**: 집중적인 검색 전략(체계적 크롤링), 특정 페이지 갱신 빈도
- **사람**: 목적 지향적이고 선택적인 탐색, 자연스러운 뒤로가기/재방문 패턴

### WT Graph 구성 요소 (논문 Table 1)

| Attribute | Description | Component |
|-----------|-------------|-----------|
| First Hit Pagename | 세션의 첫 번째 페이지 | Node |
| Detailed Pagename | hit에 해당하는 특정 페이지 | Node, Edge |
| Previous Pagename | hit의 이전 페이지 | Node, Edge |
| Timestamp | 페이지 방문 시간 | Node Label |
| Page Type | 방문 페이지의 카테고리 | Node Label |
| Benchmark Label | 해당 hit의 벤치마크 라벨 | Node Label |

### WT Graph 구성 절차

1. **세션 식별**: SGAN에서 신뢰도 < λ인 hit들을 클라이언트 세션별로 그룹핑
2. **노드 생성**: 각 hit의 `Detailed Pagename`을 노드로 생성 (같은 페이지 재방문 시 기존 노드에 집계)
3. **엣지 생성**: `Previous Pagename` → `Detailed Pagename` 방향으로 엣지 연결
4. **노드 라벨링**: 각 노드에 `Timestamp`, `Page Type`, `Benchmark Label` 부여
5. **엣지 가중치**: 동일 경로의 방문 빈도에 따라 가중치 설정
6. **동적 확장**: 새로운 hit이 도착하면 기존 세션 그래프에 점진적으로 추가 (그래프가 동적으로 성장)

### WT Graph에서 추출하는 메트릭

WT Graph로부터 다음 메트릭들을 추출하여 DGCNN의 노드 특징(node features)으로 사용합니다:

| 메트릭 | 설명 |
|--------|------|
| **Node Degree** | 노드에 연결된 엣지 수 |
| **Node Count** | 그래프 내 전체 노드 수 |
| **Edge Count** | 그래프 내 전체 엣지 수 |
| **Page Type Distribution** | 각 페이지 유형의 상대적 빈도 (hit 수 기준) |
| **Session Topics** | RAKE 알고리즘으로 페이지 제목에서 추출한 키워드 (score ≥ 1) |
| **Number of Hits** | 그래프 내 모든 페이지의 총 방문 수 |
| **Hits per Sub Page** | 각 서브페이지의 방문 횟수 |
| **Degree Centrality** | 연결된 엣지 수 기반 노드 중심성 |
| **Betweenness Centrality** | 다른 모든 노드 쌍 간 최단 경로 중 해당 노드를 경유하는 비율 |

**Session Topics 상세**: RAKE (Rapid Automatic Keyword Extraction) 알고리즘을 사용하여 WT Graph 내 페이지 제목에서 키워드를 추출합니다. Score 1 이상인 키워드만 사용하며, 이미 알려진 봇/사람 클라이언트의 세션 토픽 집합과 비교하여 미확인 클라이언트를 분류하는 데 활용합니다.

### WT Graph 구성 예시 코드

```typescript
interface PageHit {
  detailedPagename: string;   // 현재 페이지 이름
  previousPagename: string;   // 이전 페이지 이름
  firstHitPagename: string;   // 세션 첫 페이지
  timestamp: number;          // 방문 시간
  pageType: string;           // 페이지 카테고리
  benchmarkLabel: string;     // 벤치마크 라벨
}

interface WTNode {
  id: string;                 // 페이지 이름 (정규화)
  pageType: string;
  timestamps: number[];       // 방문 시간 목록
  hitCount: number;           // 해당 페이지 방문 횟수
  benchmarkLabel: string;
}

interface WTEdge {
  source: string;             // 이전 페이지
  target: string;             // 현재 페이지
  weight: number;             // 해당 경로 방문 빈도
}

class WebsiteTraversalGraph {
  nodes: Map<string, WTNode>;
  edges: Map<string, WTEdge>;  // key: "source->target"

  constructor() {
    this.nodes = new Map();
    this.edges = new Map();
  }

  /**
   * 새로운 hit을 WT Graph에 추가 (동적 확장)
   * 논문: "incoming data points, termed as hits, incrementally
   *        expand the existing session graph"
   */
  addHit(hit: PageHit): void {
    // 1. 노드 생성 또는 업데이트 (같은 페이지 재방문 시 집계)
    if (!this.nodes.has(hit.detailedPagename)) {
      this.nodes.set(hit.detailedPagename, {
        id: hit.detailedPagename,
        pageType: hit.pageType,
        timestamps: [hit.timestamp],
        hitCount: 1,
        benchmarkLabel: hit.benchmarkLabel,
      });
    } else {
      const node = this.nodes.get(hit.detailedPagename)!;
      node.hitCount++;
      node.timestamps.push(hit.timestamp);
    }

    // 2. 이전 페이지가 있으면 엣지 생성 또는 가중치 증가
    if (hit.previousPagename) {
      // 이전 페이지 노드가 없으면 생성
      if (!this.nodes.has(hit.previousPagename)) {
        this.nodes.set(hit.previousPagename, {
          id: hit.previousPagename,
          pageType: 'unknown',
          timestamps: [],
          hitCount: 0,
          benchmarkLabel: '',
        });
      }

      const edgeKey = `${hit.previousPagename}->${hit.detailedPagename}`;
      if (!this.edges.has(edgeKey)) {
        this.edges.set(edgeKey, {
          source: hit.previousPagename,
          target: hit.detailedPagename,
          weight: 1,
        });
      } else {
        this.edges.get(edgeKey)!.weight++;
      }
    }
  }

  /** 그래프 메트릭 추출 (DGCNN 입력용) */
  extractMetrics() {
    const nodeCount = this.nodes.size;
    const edgeCount = this.edges.size;

    // Node Degree 계산
    const nodeDegrees = new Map<string, number>();
    this.nodes.forEach((_, id) => nodeDegrees.set(id, 0));
    this.edges.forEach((edge) => {
      nodeDegrees.set(edge.source, (nodeDegrees.get(edge.source) || 0) + 1);
      nodeDegrees.set(edge.target, (nodeDegrees.get(edge.target) || 0) + 1);
    });

    // Page Type Distribution
    const pageTypeDistribution: Record<string, number> = {};
    let totalHits = 0;
    this.nodes.forEach((node) => {
      pageTypeDistribution[node.pageType] =
        (pageTypeDistribution[node.pageType] || 0) + node.hitCount;
      totalHits += node.hitCount;
    });
    Object.keys(pageTypeDistribution).forEach((key) => {
      pageTypeDistribution[key] /= totalHits || 1;
    });

    // Hits per Sub Page
    const hitsPerSubPage = new Map<string, number>();
    this.nodes.forEach((node, id) => {
      hitsPerSubPage.set(id, node.hitCount);
    });

    // Degree Centrality: degree / (N-1)
    const degreeCentrality = new Map<string, number>();
    nodeDegrees.forEach((degree, id) => {
      degreeCentrality.set(id, nodeCount > 1 ? degree / (nodeCount - 1) : 0);
    });

    return {
      nodeCount,
      edgeCount,
      totalHits,
      nodeDegrees,
      pageTypeDistribution,
      hitsPerSubPage,
      degreeCentrality,
      // betweennessCentrality: 별도 그래프 알고리즘으로 계산 필요
      // sessionTopics: RAKE 알고리즘으로 페이지 제목에서 추출 필요
    };
  }

  /**
   * DGCNN 입력을 위한 인접 행렬 + 노드 특징 행렬 생성
   * - 인접 행렬: N x N (방향 그래프, 가중치 포함)
   * - 노드 특징 행렬: N x F (각 노드의 특징 벡터)
   */
  toDGCNNInput() {
    const nodeList = Array.from(this.nodes.keys());
    const n = nodeList.length;

    // 인접 행렬 (가중치 포함)
    const adjMatrix: number[][] = Array(n)
      .fill(null)
      .map(() => Array(n).fill(0));

    for (const edge of this.edges.values()) {
      const i = nodeList.indexOf(edge.source);
      const j = nodeList.indexOf(edge.target);
      if (i >= 0 && j >= 0) {
        adjMatrix[i][j] = edge.weight;
      }
    }

    return { adjMatrix, nodeList, nodeCount: n };
  }
}
```

---

## DGCNN (Deep Graph Convolutional Neural Network) 상세

### 개요

DGCNN은 Zhang et al. [42]의 프레임워크를 기반으로 하며, 비방향 그래프(undirected graph)에 노드 특징을 인코딩하여 그래프 분류를 수행하는 딥러닝 아키텍처입니다. 논문에서는 WT Graph를 입력으로 받아 봇/사람을 이진 분류합니다.

### DGCNN 3단계 처리 파이프라인

DGCNN은 3개의 연속된 레이어 그룹으로 구성됩니다:

```plaintext
WT Graph (인접 행렬 + 노드 특징)
       │
       ▼
┌─────────────────────────────┐
│  1단계: GCN (Graph           │  ← 그래프에서 구조적 특징 추출
│         Convolution Layers) │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  2단계: SortPooling          │  ← 가변 크기 → 고정 크기 텐서 변환
│                             │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  3단계: 1D-CNN               │  ← 정렬된 특징에서 패턴 학습 → 분류
│         + Dense + Sigmoid   │
└─────────────────────────────┘
           │
           ▼
     봇/사람 확률 (0~1)
```

### 1단계: GCN (Graph Convolution Network)

그래프 구조에서 이웃 노드의 특징을 집계(aggregate)하여 노드 임베딩을 학습합니다.

**구조 (논문 Section 4.2)**:
- **4개의 그래프 컨볼루션 레이어**: hidden units = `32, 32, 32, 1`
- **활성화 함수**: 각 레이어에서 `tanh` 사용

**동작 원리**:
각 GCN 레이어에서 노드 `v`의 새로운 표현은:
1. 인접 행렬을 사용하여 이웃 노드의 특징을 집계
2. 학습 가능한 가중치 행렬과 곱한 후 tanh 활성화 적용
3. 이를 반복하여 더 넓은 범위의 이웃 정보를 캡처

```
h_v^(l+1) = tanh(W^(l) · Σ_{u∈N(v)} h_u^(l))
```

4개 레이어를 거치면 각 노드는 4-hop 이웃까지의 구조 정보를 반영한 임베딩을 갖게 됩니다.

### 2단계: SortPooling

GCN의 출력은 그래프 크기에 따라 가변적(노드 수가 다름)이므로, 고정 크기 텐서로 변환해야 1D-CNN에 입력할 수 있습니다.

**동작 원리**:
1. 각 노드의 마지막 GCN 레이어 출력값을 기준으로 **내림차순 정렬**
2. 상위 **k개** 노드만 선택 (k = 35)
3. 노드가 k개 미만인 그래프는 0으로 패딩, 초과하면 잘라냄
4. 결과: 고정 크기 텐서 (k × GCN 출력 차원)

**k = 35의 의미**: 그래프에서 가장 "중요한" 상위 35개 노드의 특징만 사용하여 분류합니다. 이는 WT Graph의 핵심 탐색 패턴을 요약하는 효과가 있습니다.

### 3단계: 1D-CNN + Dense Layers

SortPooling의 고정 크기 출력을 1차원 컨볼루션으로 처리하여 최종 분류합니다.

**구조 (논문 Section 4.2)**:

| Layer | 상세 설정 |
|-------|----------|
| 1D-Conv Layer 1 | 16 filters, kernel size = `sum of all GCN hidden units` (= 32+32+32+1 = 97), stride = 97 |
| 1D-Conv Layer 2 | 32 filters, kernel size = 5, stride = 1 |
| MaxPooling | pool size = 2 |
| Dense Layer | 128 hidden units, ReLU 활성화 |
| Dropout | p = 0.5 |
| Output Layer | 1 unit, Sigmoid 활성화 (이진 분류) |

**1D-Conv Layer 1의 커널 크기가 GCN hidden units 합계인 이유**:
SortPooling 후 각 노드는 모든 GCN 레이어의 출력을 연결(concatenate)한 벡터를 가집니다. 커널 크기를 이 벡터의 차원과 동일하게 설정하여, 한 번의 컨볼루션으로 **한 노드의 전체 GCN 특징**을 처리합니다. stride도 동일하게 설정하여 노드 단위로 이동합니다.

### 학습 설정

| 항목 | 값 |
|------|-----|
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |
| Learning Rate | α = 0.0001 |
| 출력 활성화 | Sigmoid (1 unit) |
| SortPooling k | 35 |

### DGCNN 모델 구조 예시

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    그래프 컨볼루션 레이어
    인접 행렬을 통해 이웃 노드 특징을 집계한 후 선형 변환
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: 인접 행렬 (N x N), x: 노드 특징 (N x in_features)
        # 이웃 노드 특징 집계 후 선형 변환
        h = torch.matmul(adj, x)
        return self.linear(h)


class DGCNN(nn.Module):
    """
    Deep Graph Convolutional Neural Network
    논문 Section 4.2 기반 구현

    구조: GCN (4 layers) → SortPooling (k=35) → 1D-CNN (2 layers) → Dense → Sigmoid
    """
    def __init__(self, input_dim: int, k: int = 35):
        super().__init__()

        # ===== 1단계: GCN 레이어 (hidden units: 32, 32, 32, 1) =====
        self.conv1 = GraphConvLayer(input_dim, 32)
        self.conv2 = GraphConvLayer(32, 32)
        self.conv3 = GraphConvLayer(32, 32)
        self.conv4 = GraphConvLayer(32, 1)

        # SortPooling 파라미터
        self.k = k  # 상위 k개 노드 선택 (논문: k=35)

        # GCN 전체 hidden units 합계 = 32 + 32 + 32 + 1 = 97
        total_gcn_units = 32 + 32 + 32 + 1  # = 97

        # ===== 3단계: 1D-CNN =====
        # Conv1: 16 filters, kernel_size = total_gcn_units, stride = total_gcn_units
        self.conv1d_1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=total_gcn_units,
            stride=total_gcn_units
        )
        # Conv2: 32 filters, kernel_size = 5, stride = 1
        self.conv1d_2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1
        )
        # MaxPooling: pool_size = 2
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # Dense layer 입력 크기 계산
        # SortPooling 출력: k * total_gcn_units = 35 * 97 = 3395
        # Conv1d_1 출력: (3395 - 97) / 97 + 1 = 35
        # Conv1d_2 출력: (35 - 5) / 1 + 1 = 31
        # MaxPool 출력: 31 // 2 = 15
        # Flatten: 32 * 15 = 480
        conv1d_1_out = (k * total_gcn_units - total_gcn_units) // total_gcn_units + 1
        conv1d_2_out = conv1d_1_out - 5 + 1
        maxpool_out = conv1d_2_out // 2
        dense_input = 32 * maxpool_out

        self.fc = nn.Linear(dense_input, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: 노드 특징 행렬 (batch x N x input_dim)
        adj: 인접 행렬 (batch x N x N)
        Returns: 봇 확률 (batch x 1), 범위 [0, 1]
        """
        batch_size = x.size(0)

        # ===== 1단계: GCN - tanh 활성화 =====
        h1 = torch.tanh(self.conv1(x, adj))   # (batch, N, 32)
        h2 = torch.tanh(self.conv2(h1, adj))   # (batch, N, 32)
        h3 = torch.tanh(self.conv3(h2, adj))   # (batch, N, 32)
        h4 = torch.tanh(self.conv4(h3, adj))   # (batch, N, 1)

        # 모든 GCN 레이어 출력 연결 (SortPooling 입력)
        h = torch.cat([h1, h2, h3, h4], dim=-1)  # (batch, N, 97)

        # ===== 2단계: SortPooling =====
        # 마지막 GCN 레이어 출력 기준 내림차순 정렬
        sort_scores = h4.squeeze(-1)               # (batch, N)
        _, sort_idx = sort_scores.sort(dim=-1, descending=True)

        # 상위 k개 노드 선택
        sort_idx = sort_idx[:, :self.k]            # (batch, k)
        sort_idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))
        h = torch.gather(h, 1, sort_idx_expanded)  # (batch, k, 97)

        # ===== 3단계: 1D-CNN =====
        h = h.view(batch_size, 1, -1)             # (batch, 1, k*97)
        h = F.relu(self.conv1d_1(h))               # (batch, 16, k)
        h = F.relu(self.conv1d_2(h))               # (batch, 32, k-4)
        h = self.maxpool(h)                         # (batch, 32, (k-4)//2)

        # Dense + Sigmoid
        h = h.view(batch_size, -1)                 # (batch, 32 * (k-4)//2)
        h = F.relu(self.fc(h))                     # (batch, 128)
        h = self.dropout(h)
        out = torch.sigmoid(self.output(h))         # (batch, 1)

        return out


# 사용 예시
if __name__ == "__main__":
    # 하이퍼파라미터
    input_dim = 10   # 노드 특징 차원 (메트릭 수)
    k = 35           # SortPooling에서 선택할 상위 노드 수
    batch_size = 4
    num_nodes = 50   # 예시 그래프의 노드 수

    model = DGCNN(input_dim=input_dim, k=k)

    # 더미 입력 생성
    x = torch.randn(batch_size, num_nodes, input_dim)       # 노드 특징
    adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()  # 인접 행렬

    # 추론
    prob = model(x, adj)
    print(f"봇 확률: {prob.squeeze()}")
    # 예: tensor([0.4821, 0.5103, 0.4997, 0.5234])
    # λ와 비교하여 최종 판정
```

---

## 성능 (논문 Section 5.2)

### 모델 비교 (Table 3)

| Model | Accuracy | Recall | Precision | F1-Score | AUROC |
|-------|----------|--------|-----------|----------|-------|
| **SGAN** | 0.9895 | 0.9875 | 0.9189 | 0.9519 | 0.9886 |
| **DGCNN** | 0.9845 | 0.9833 | 0.9791 | 0.9812 | 0.9892 |
| Botcha-MAM | 0.9364 | 0.8383 | 1.0 | 0.9120 | 0.9437 |
| Botcha-RAM | 0.9952 | 0.9663 | 0.9807 | 0.9735 | 0.9996 |

DGCNN은 기술적 특징 없이 행위 특징만으로 Accuracy 98.45%, F1-Score 98.12%를 달성합니다. SGAN보다 Precision이 높으며(0.9791 vs 0.9189), F1-Score도 우수합니다.

### WT Graph 크기별 분류 성능 (Table 5)

| Nodes | # Graphs | ACC | Recall | Precision | F1-Score |
|-------|----------|-----|--------|-----------|----------|
| 1 | 26,137 | 0.998 | 0.981 | 0.998 | 0.99 |
| 2 | 17,066 | 0.973 | 1.0 | 0.974 | 0.986 |
| 3 | 3,533 | 1.0 | 1.0 | 1.0 | 1.0 |
| 4 | 371 | 0.998 | 0.999 | 0.999 | 0.999 |
| 5 | 251 | 0.998 | 1.0 | 0.998 | 0.999 |
| 6 | 101 | 0.998 | 1.0 | 0.998 | 0.999 |
| 7 | 526 | 0.997 | 1.0 | 0.997 | 0.998 |
| 8 | 1,579 | 1.0 | 1.0 | 1.0 | 1.0 |
| 9 | 1,175 | 1.0 | 1.0 | 1.0 | 1.0 |
| 10 | 108 | 1.0 | 1.0 | 1.0 | 1.0 |

**핵심 발견**: 그래프 크기(노드 수)가 증가할수록 분류 성능이 향상됩니다. 노드가 1개뿐인 최소 그래프에서도 ACC 99.8%를 달성하는데, 이는 WT Graph가 같은 웹페이지에 대한 여러 상호작용을 단일 노드에 집계하기 때문입니다. 노드가 8개 이상이면 모든 메트릭에서 100%를 달성합니다.
