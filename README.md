# BOTracle 프레임워크 기반 봇 탐지

## BOTracle 개요

출처:
- 논문: [BOTracle: A framework for Discriminating Bots and Humans (arXiv)](https://arxiv.org/abs/2412.02266)
- 학회: ESORICS 2024 International Workshops, Springer LNCS vol 15264

저자:
- Jan Kadel, August See, Ritwik Sinha, Mathias Fischer (2025)

## 핵심 특징

BOTracle은 세 가지 탐지 방법을 단계적으로 적용하는 다단계 파이프라인 프레임워크

| 단계 | 방법 | 기술 | 목적 |
|--|--|--|--|
| 1단계 | [휴리스틱]() | 규칙 기반 | 명백한 봇 빠른 탐지 |
| 2단계 | [기술적 특징 분석]() | SGAN | 기술적 지문 기분 탐지 |
| 3단계 | [행위 분석]() | DGCNN | 탐색 패턴 기반 탐지 |