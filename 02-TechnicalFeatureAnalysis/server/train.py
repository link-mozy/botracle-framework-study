"""
BOTracle 2단계 - SGAN 학습 스크립트

샘플 데이터로 SGAN 모델을 학습시키고, 결과를 평가합니다.

실행 방법:
    cd 02-TechnicalFeatureAnalysis
    python server/train.py

학습 완료 후:
    - 모델 가중치: server/trained_model/
    - 학습 결과가 콘솔에 출력됩니다
"""

import os
import sys
import numpy as np

# sgan 패키지 임포트를 위한 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sgan.sample_data import generate_dataset
from sgan.model import SGAN


def evaluate(sgan, test_features, test_labels):
    """
    학습된 모델의 성능을 평가합니다.

    평가 지표 설명 (논문 Table 3에서 사용):

      Accuracy (정확도):
        전체 중 맞춘 비율
        예: 100개 중 95개 맞춤 → 0.95

      Precision (정밀도):
        "봇이라고 판정한 것" 중 실제 봇의 비율
        → 높을수록 오탐(False Positive)이 적음
        → 사람을 봇으로 잘못 판정하면 고객 불만 발생

      Recall (재현율):
        "실제 봇" 중 탐지에 성공한 비율
        → 높을수록 놓치는 봇이 적음
        → 봇을 놓치면 보안 위험 발생

      F1-Score:
        Precision과 Recall의 조화 평균
        → 둘 다 높아야 F1도 높음

    논문 결과 (SGAN):
      Accuracy=0.9895, Recall=0.9875, Precision=0.9189, F1=0.9519
    """
    # 예측 수행
    predictions = []
    for feature in test_features:
        pred, _ = sgan.predict(feature)
        predictions.append(1 if pred == "bot" else 0)

    predictions = np.array(predictions)

    # 지표 계산
    correct = (predictions == test_labels).sum()
    accuracy = correct / len(test_labels)

    # 봇(=1) 기준으로 Precision, Recall 계산
    true_positive = ((predictions == 1) & (test_labels == 1)).sum()
    false_positive = ((predictions == 1) & (test_labels == 0)).sum()
    false_negative = ((predictions == 0) & (test_labels == 1)).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total": len(test_labels),
        "correct": correct,
    }


def main():
    print("=" * 60)
    print("BOTracle 2단계 - SGAN 학습")
    print("=" * 60)

    # ── 1. 데이터 생성 ──
    print("\n[1/4] 샘플 데이터 생성 중...")
    dataset = generate_dataset(
        n_labeled_human=500,
        n_labeled_bot=500,
        n_unlabeled=3000,
    )

    # 학습/테스트 분리 (80% 학습, 20% 테스트)
    n_total = len(dataset["labels"])
    n_train = int(n_total * 0.8)

    idx = np.random.permutation(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_features = dataset["labeled_features"][train_idx]
    train_labels = dataset["labels"][train_idx]
    test_features = dataset["labeled_features"][test_idx]
    test_labels = dataset["labels"][test_idx]

    print(f"  학습 데이터: {len(train_labels)} (사람: {(train_labels == 0).sum()}, 봇: {(train_labels == 1).sum()})")
    print(f"  테스트 데이터: {len(test_labels)} (사람: {(test_labels == 0).sum()}, 봇: {(test_labels == 1).sum()})")
    print(f"  비레이블 데이터: {len(dataset['unlabeled_features'])}")

    # ── 2. 모델 생성 ──
    print("\n[2/4] SGAN 모델 생성 중...")
    sgan = SGAN(feature_dim=11, num_classes=2)
    print(f"  생성자: 100차원 노이즈 → 200 → 11차원 출력")
    print(f"  판별자: 11차원 입력 → 100×3 공유 레이어 → 판별 헤드 + 분류 헤드")
    print(f"  옵티마이저: Adam (lr={sgan.LEARNING_RATE}, β₁={sgan.BETA_1})")

    # ── 3. 학습 ──
    print("\n[3/4] SGAN 학습 시작...")
    print("-" * 60)
    history = sgan.train(
        labeled_features=train_features,
        labels=train_labels,
        unlabeled_features=dataset["unlabeled_features"],
        epochs=100,
        batch_size=32,
    )
    print("-" * 60)

    # ── 4. 평가 ──
    print("\n[4/4] 모델 평가 중...")
    metrics = evaluate(sgan, test_features, test_labels)

    print(f"\n{'=' * 60}")
    print(f"  평가 결과")
    print(f"{'=' * 60}")
    print(f"  Accuracy  (정확도): {metrics['accuracy']:.4f}")
    print(f"  Precision (정밀도): {metrics['precision']:.4f}")
    print(f"  Recall    (재현율): {metrics['recall']:.4f}")
    print(f"  F1-Score          : {metrics['f1_score']:.4f}")
    print(f"  ({metrics['correct']}/{metrics['total']} 맞춤)")
    print(f"\n  참고 - 논문 SGAN 결과:")
    print(f"  Accuracy=0.9895, Precision=0.9189, Recall=0.9875, F1=0.9519")

    # ── 모델 저장 ──
    save_dir = os.path.join(os.path.dirname(__file__), "trained_model")
    os.makedirs(save_dir, exist_ok=True)
    sgan.save_weights(save_dir)

    print(f"\n학습 완료! 서버를 시작하려면:")
    print(f"  python server/app.py")


if __name__ == "__main__":
    main()
