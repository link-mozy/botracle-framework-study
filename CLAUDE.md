# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BOTracle 논문(arXiv:2412.02266, ESORICS 2024) 기반의 봇 탐지 프레임워크 학습 및 구현 레포지토리. 논문의 3단계 파이프라인을 각 디렉토리로 분리하여 분석·구현합니다.

**참고 논문**: `doc/BOTracle-A_framework_for_Discriminating_Bots_and_Humans.pdf`

## Repository Structure

```
01-heuristic/              # 1단계: 휴리스틱 기반 봇 탐지 (규칙 기반, 빠른 필터링)
02-TechnicalFeatureAnalysis/  # 2단계: SGAN 기반 기술적 특징 분석 (Python 3.10.2)
03-BehavioralAyalysis/     # 3단계: DGCNN 기반 웹사이트 탐색 그래프 분석
doc/                       # 원본 논문 PDF
```

각 단계 디렉토리의 `CLAUDE.md`에 해당 단계의 상세 설명, 아키텍처, 구현 예시가 포함됩니다.

## BOTracle Pipeline (논문 핵심)

요청(Hit) → **1단계 휴리스틱** (명백한 봇 필터) → **2단계 SGAN** (기술적 특징으로 개별 요청 분류) → **3단계 DGCNN** (세션 단위 행동 패턴 분류). 각 단계에서 신뢰도 임계값 λ를 초과하면 판정 종료, 미달하면 다음 단계로 전달.

## Tech Stack

- **Client**: JavaScript (브라우저 특징 수집)
- **Server/ML**: Python 3.10.2, TensorFlow/Keras
- **2단계 SGAN**: Semi-Supervised GAN - 판별자(공유 히든 레이어 + 판별/분류 듀얼 헤드), 생성자(100차원 latent → 특징 벡터)
- **3단계 DGCNN**: Deep Graph Convolutional Neural Network + Website Traversal 그래프

## Writing Conventions

- 각 단계 디렉토리의 `CLAUDE.md`는 논문 내용을 정확히 반영해야 함 (수치, 아키텍처, 수식 등)
- 논문에 없는 내용(Siamese Network 등)을 혼합하지 않을 것
- 구현 예시 코드는 논문의 Section 4 (Implementation)을 기반으로 작성
- 한국어로 문서 작성, 기술 용어는 영어 병기
