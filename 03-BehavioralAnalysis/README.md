# BOTracle 3단계 - 행위 기반 분석 (Behavioral Analysis)

BOTracle 파이프라인의 3단계로, **DGCNN (Deep Graph Convolutional Neural Network)** 을 사용하여
웹사이트 탐색 행동 패턴만으로 봇과 사람을 분류합니다.

기술적 특징(IP, User Agent 등)을 전혀 사용하지 않고, 오직 **탐색 행동**만으로 판별합니다.

## 프로젝트 구조

```
03-BehavioralAnalysis/
├── CLAUDE.md                  # 논문 기반 상세 문서
├── README.md                  # 이 파일
├── .python-version            # Python 3.10.2
├── requirements.txt           # PyTorch, Flask 등 의존성
├── client/
│   ├── index.html             # 쇼핑몰 탐색 시뮬레이터 데모 페이지
│   └── wt_graph_collector.js  # 브라우저 네비게이션 데이터 수집기
└── server/
    ├── app.py                 # Flask REST API 서버 (포트 5001)
    ├── train.py               # DGCNN 학습 스크립트
    └── dgcnn/
        ├── model.py           # DGCNN 모델 (GCN + SortPooling + 1D-CNN)
        ├── wt_graph.py        # WT Graph 구성 + 메트릭 추출
        ├── preprocessor.py    # 그래프 → DGCNN 입력 텐서 변환
        ├── predictor.py       # 모델 추론 + 규칙 기반 폴백
        └── sample_data.py     # 봇/사람 탐색 패턴 시뮬레이션
```

## 구성 요소

### Client (`client/`)
- **wt_graph_collector.js**: 브라우저에서 페이지 네비게이션을 추적하고 서버로 hit 데이터 전송
- **index.html**: 가상 쇼핑몰 탐색 시뮬레이터 (사람 탐색 + 봇 시뮬레이션)

### Server (`server/`)
- **app.py**: Flask REST API - hit 수신, 세션별 WT Graph 관리, DGCNN 분석
- **train.py**: 샘플 데이터 생성 → DGCNN 학습 → 성능 평가
- **dgcnn/**: DGCNN 모델 및 WT Graph 모듈

## 실행 방법

### 1. 의존성 설치

```bash
cd 03-BehavioralAnalysis
pip install -r requirements.txt
```

### 2. 모델 학습 (선택)

```bash
python server/train.py
```

학습하지 않아도 규칙 기반 폴백으로 동작합니다.

### 3. 서버 실행

```bash
python server/app.py
```

서버가 `http://localhost:5001`에서 시작됩니다.

### 4. 데모 페이지 열기

`client/index.html`을 브라우저에서 열어 쇼핑몰 탐색 시뮬레이터를 사용합니다.

## API 엔드포인트

| Method | URL | 설명 |
|--------|-----|------|
| POST | `/api/hit` | 페이지 방문 데이터 수신 |
| POST | `/api/analyze` | 세션의 WT Graph를 DGCNN으로 분석 |
| GET | `/api/session/<id>` | 세션 그래프 상태 조회 |
| GET | `/api/health` | 서버 상태 확인 |

## 기술 스택

- **ML 프레임워크**: PyTorch (DGCNN)
- **API 서버**: Flask
- **키워드 추출**: RAKE (rake-nltk)
- **클라이언트**: JavaScript (ES6)
