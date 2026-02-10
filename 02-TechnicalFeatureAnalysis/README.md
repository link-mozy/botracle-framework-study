# 02 - Technical Feature Analysis (기술적 특징 분석)

BOTracle 파이프라인 2단계: SGAN 기반 기술적 특징 분석

## 프로젝트 구조

```
02-TechnicalFeatureAnalysis/
├── CLAUDE.md              # 상세 논문 분석 및 아키텍처 설명
├── requirements.txt       # Python 패키지 의존성
├── client/
│   ├── index.html         # 데모 웹 페이지
│   └── feature_extractor.js  # 브라우저 특징 수집기
└── server/
    ├── app.py             # Flask API 서버
    ├── train.py           # SGAN 학습 스크립트
    └── sgan/
        ├── model.py       # SGAN 모델 (Generator + Discriminator)
        ├── preprocessor.py # 특징 전처리기
        ├── predictor.py   # 예측기 (모델 로드 + 추론)
        └── sample_data.py # 샘플 데이터 생성기
```

## 구조 설명

### Client (브라우저)
- feature_extractor.js - 브라우저에서 기술적 특징을 수집하는 클래스. 논문의 Feature Importance 순위를 주석으로 표기
- index.html - 데모 웹 페이지. 수집된 특징 확인 + 서버 분석 요청 UI

### Server (Python)
- app.py - Flask REST API 서버. /api/analyze 엔드포인트에서 특징을 받아 SGAN으로 판별
- train.py - 학습 스크립트. 데이터 생성 → 학습 → 평가 → 모델 저장

### SGAN 모듈
- model.py - 핵심 파일. SGAN 모델 전체 구현. 초급자를 위해 모든 개념을 비유와 함께 상세 주석 (위조지폐 비유, 레이어별 역할, 수식 의미, 학습 과정 등)
- preprocessor.py - 원시 JSON → 11차원 수치 벡터 변환 (Flagging, 정규화, One-Hot 인코딩)
- predictor.py - 학습된 모델 로드 + 추론. 모델 없으면 규칙 기반 폴백
- sample_data.py - 사람/봇 트래픽 시뮬레이션 데이터 생성 (봇 3단계: 단순/중급/고급)

## 실행 방법

```bash
cd 02-TechnicalFeatureAnalysis

# 1. 의존성 설치
pip install -r requirements.txt

# 2. SGAN 모델 학습
python server/train.py

# 3. API 서버 실행
python server/app.py

# 4. 브라우저에서 데모 페이지 열기
open client/index.html
```
