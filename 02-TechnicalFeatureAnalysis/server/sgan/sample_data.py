"""
BOTracle 2단계 - 샘플 데이터 생성기

실제 웹 트래픽 데이터가 없으므로, 학습용 샘플 데이터를 생성합니다.
논문의 데이터 특성과 Feature Importance 분석 결과를 반영하여
사람 트래픽과 봇 트래픽의 특징 차이를 시뮬레이션합니다.

논문 데이터 규모 (Section 5.1, Table 2):
    Bot:     65,018 hits (휴리스틱 보강 후)
    Human:   7,630 hits
    Unknown: 710,023 hits
"""

import numpy as np

# 재현 가능한 결과를 위한 랜덤 시드
np.random.seed(42)


def generate_human_features(n_samples: int) -> np.ndarray:
    """
    사람 사용자의 기술적 특징을 시뮬레이션합니다.

    사람의 특성:
      - 다양한 브라우저 창 크기 (일반적인 해상도 범위)
      - 표준 User Agent (Mozilla/5.0)
      - Java 미지원 (현대 브라우저)
      - 자동화 도구 미탐지
      - 쿠키 활성화

    Args:
        n_samples: 생성할 샘플 수

    Returns:
        numpy array (shape: [n_samples, 11])
        벡터 구조: [height, width, java, webdriver, auto_count,
                    cookie, ua_mozilla, ua_python, ua_curl, ua_bot, ua_other]
    """
    features = np.zeros((n_samples, 11), dtype=np.float32)

    # 브라우저 창 높이 (정규화): 일반적으로 600~1200px
    # → 정규화: 600/2160=0.28 ~ 1200/2160=0.56
    features[:, 0] = np.random.uniform(0.25, 0.60, n_samples)

    # 브라우저 창 너비 (정규화): 일반적으로 1024~1920px
    # → 정규화: 1024/3840=0.27 ~ 1920/3840=0.50
    features[:, 1] = np.random.uniform(0.25, 0.55, n_samples)

    # Java 지원: 현대 브라우저는 대부분 미지원 (5%만 지원)
    features[:, 2] = (np.random.random(n_samples) < 0.05).astype(np.float32)

    # WebDriver: 사람은 자동화 도구를 사용하지 않음 (0%)
    features[:, 3] = 0.0

    # 자동화 플래그 수: 사람은 0개
    features[:, 4] = 0.0

    # 쿠키: 사람은 대부분 활성화 (95%)
    features[:, 5] = (np.random.random(n_samples) < 0.95).astype(np.float32)

    # User Agent: 대부분 표준 브라우저 (Mozilla/5.0 = 인덱스 0)
    # 95% Mozilla, 5% 기타
    ua_rand = np.random.random(n_samples)
    features[:, 6] = (ua_rand < 0.95).astype(np.float32)  # Mozilla/5.0
    features[:, 10] = (ua_rand >= 0.95).astype(np.float32)  # Other

    return features


def generate_bot_features(n_samples: int) -> np.ndarray:
    """
    봇의 기술적 특징을 시뮬레이션합니다.

    논문의 Feature Importance (Table 4) 기반 봇 특성:
      - 비현실적으로 작은 브라우저 창 크기 (R²=0.542, 0.287 - 가장 중요)
      - 비표준 User Agent 또는 자동화 라이브러리 UA
      - Java 지원됨으로 보고 (오래된/위조된 UA)
      - WebDriver 플래그 활성화
      - 자동화 도구 흔적

    봇의 종류를 3가지로 시뮬레이션:
      Type A (40%): 단순 봇 - 자동화 라이브러리 직접 사용
      Type B (40%): 중급 봇 - 실제 브라우저 사용, 일부 위장
      Type C (20%): 고급 봇 - 사람처럼 위장 (탐지 어려움)
    """
    features = np.zeros((n_samples, 11), dtype=np.float32)

    # 봇 타입별 샘플 수
    n_a = int(n_samples * 0.4)  # 단순 봇
    n_b = int(n_samples * 0.4)  # 중급 봇
    n_c = n_samples - n_a - n_b  # 고급 봇

    # ── Type A: 단순 봇 (40%) ──
    # 최소 창 크기, 자동화 도구 그대로 노출
    a_end = n_a
    features[:a_end, 0] = np.random.uniform(0.0, 0.05, n_a)     # 매우 작은 높이
    features[:a_end, 1] = np.random.uniform(0.0, 0.05, n_a)     # 매우 작은 너비
    features[:a_end, 2] = 0.0                                    # Java 미지원
    features[:a_end, 3] = 1.0                                    # WebDriver 활성화
    features[:a_end, 4] = np.random.uniform(0.3, 1.0, n_a)      # 자동화 플래그 다수
    features[:a_end, 5] = 0.0                                    # 쿠키 비활성화
    features[:a_end, 7] = 1.0                                    # python UA

    # ── Type B: 중급 봇 (40%) ──
    # 어느 정도 위장하지만 완벽하지 않음
    b_end = a_end + n_b
    features[a_end:b_end, 0] = np.random.uniform(0.05, 0.20, n_b)  # 작은 높이
    features[a_end:b_end, 1] = np.random.uniform(0.05, 0.25, n_b)  # 작은 너비
    features[a_end:b_end, 2] = 1.0                                   # Java 지원 (위조)
    features[a_end:b_end, 3] = 0.0                                   # WebDriver 숨김
    features[a_end:b_end, 4] = np.random.uniform(0.0, 0.15, n_b)    # 일부 플래그
    features[a_end:b_end, 5] = 1.0                                   # 쿠키 활성화
    features[a_end:b_end, 6] = 1.0                                   # Mozilla UA (위조)

    # ── Type C: 고급 봇 (20%) ──
    # 사람과 매우 유사 → SGAN이 탐지하기 어려움
    features[b_end:, 0] = np.random.uniform(0.20, 0.55, n_c)    # 정상 범위 높이
    features[b_end:, 1] = np.random.uniform(0.25, 0.50, n_c)    # 정상 범위 너비
    features[b_end:, 2] = 0.0                                    # Java 미지원 (정상)
    features[b_end:, 3] = 0.0                                    # WebDriver 숨김
    features[b_end:, 4] = 0.0                                    # 플래그 없음
    features[b_end:, 5] = 1.0                                    # 쿠키 활성화
    features[b_end:, 6] = 1.0                                    # Mozilla UA

    return features


def generate_unlabeled_features(n_samples: int) -> np.ndarray:
    """
    레이블이 없는 데이터를 시뮬레이션합니다.

    실제 환경에서는 대부분의 트래픽이 봇인지 사람인지 알 수 없습니다.
    논문에서 전체 782,671 hits 중 710,023(90.7%)이 Unknown입니다.

    여기서는 사람 70% + 봇 30% 비율로 섞되, 레이블은 제공하지 않습니다.
    """
    n_human = int(n_samples * 0.7)
    n_bot = n_samples - n_human

    human = generate_human_features(n_human)
    bot = generate_bot_features(n_bot)

    # 섞기
    mixed = np.concatenate([human, bot], axis=0)
    np.random.shuffle(mixed)

    return mixed


def generate_dataset(n_labeled_human=500, n_labeled_bot=500, n_unlabeled=5000):
    """
    학습용 전체 데이터셋을 생성합니다.

    Args:
        n_labeled_human: 레이블이 있는 사람 데이터 수
        n_labeled_bot:   레이블이 있는 봇 데이터 수
        n_unlabeled:     레이블이 없는 데이터 수

    Returns:
        dict:
            labeled_features: 레이블이 있는 특징 (shape: [N, 11])
            labels:           해당 레이블 (shape: [N], 0=사람, 1=봇)
            unlabeled_features: 레이블이 없는 특징 (shape: [M, 11])
    """
    # 레이블 있는 데이터 생성
    human_features = generate_human_features(n_labeled_human)
    bot_features = generate_bot_features(n_labeled_bot)

    # 합치고 셔플
    labeled_features = np.concatenate([human_features, bot_features], axis=0)
    labels = np.concatenate([
        np.zeros(n_labeled_human, dtype=np.int32),  # 0 = 사람
        np.ones(n_labeled_bot, dtype=np.int32),      # 1 = 봇
    ])

    shuffle_idx = np.random.permutation(len(labels))
    labeled_features = labeled_features[shuffle_idx]
    labels = labels[shuffle_idx]

    # 레이블 없는 데이터 생성
    unlabeled_features = generate_unlabeled_features(n_unlabeled)

    print(f"데이터셋 생성 완료:")
    print(f"  레이블 있는 데이터: {len(labels)} (사람: {n_labeled_human}, 봇: {n_labeled_bot})")
    print(f"  레이블 없는 데이터: {len(unlabeled_features)}")
    print(f"  특징 벡터 차원: {labeled_features.shape[1]}")

    return {
        "labeled_features": labeled_features,
        "labels": labels,
        "unlabeled_features": unlabeled_features,
    }
