"""
BOTracle 3단계 - DGCNN 학습 스크립트

샘플 데이터를 생성하고, DGCNN 모델을 학습시키고, 성능을 평가합니다.

실행 방법:
    cd 03-BehavioralAnalysis
    pip install -r requirements.txt
    python server/train.py

학습 과정:
    [1/4] 샘플 데이터 생성
    [2/4] DGCNN 모델 생성
    [3/4] DGCNN 학습 (100 epochs)
    [4/4] 모델 평가 (Accuracy, Precision, Recall, F1-Score)
"""

import os
import sys
import numpy as np

# NLTK 데이터 다운로드 (RAKE 키워드 추출에 필요)
# 최초 실행 시 한 번만 다운로드됩니다
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except ImportError:
    print("nltk 미설치 (RAKE 키워드 추출이 간단 모드로 동작합니다)")

from dgcnn.sample_data import generate_dataset
from dgcnn.preprocessor import GraphPreprocessor
from dgcnn.model import DGCNNClassifier


def main():
    print("=" * 60)
    print("BOTracle 3단계 - DGCNN 모델 학습")
    print("행위 기반 봇 탐지 (Website Traversal Graph + DGCNN)")
    print("=" * 60)

    # ══════════════════════════════════════════════
    # [1/4] 샘플 데이터 생성
    # ══════════════════════════════════════════════
    print("\n[1/4] 샘플 데이터 생성 중...")
    dataset = generate_dataset(n_human=300, n_bot=300)
    graphs = dataset["graphs"]
    labels = dataset["labels"]

    # ── 전처리: WT Graph → 텐서 ──
    print("\n  그래프를 DGCNN 입력 형식으로 변환 중...")
    preprocessor = GraphPreprocessor()
    adj_list, features_list = preprocessor.transform_batch(graphs)

    print(f"  노드 특징 차원: {preprocessor.feature_dim}")

    # ── 학습/테스트 데이터 분리 (80% / 20%) ──
    n_total = len(labels)
    n_train = int(n_total * 0.8)

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_adj = [adj_list[i] for i in train_idx]
    train_feat = [features_list[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_adj = [adj_list[i] for i in test_idx]
    test_feat = [features_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"  학습 데이터: {len(train_labels)}개")
    print(f"  테스트 데이터: {len(test_labels)}개")

    # ══════════════════════════════════════════════
    # [2/4] DGCNN 모델 생성
    # ══════════════════════════════════════════════
    print("\n[2/4] DGCNN 모델 생성 중...")
    classifier = DGCNNClassifier(
        input_dim=preprocessor.feature_dim,  # 12
        k=35,                                  # SortPooling k값
    )

    print(f"  입력 차원: {preprocessor.feature_dim}")
    print(f"  SortPooling k: 35")
    print(f"  GCN 레이어: 4개 (32, 32, 32, 1)")
    print(f"  학습률: {classifier.model.LEARNING_RATE}")

    # ══════════════════════════════════════════════
    # [3/4] DGCNN 학습
    # ══════════════════════════════════════════════
    print("\n[3/4] DGCNN 학습 시작...")
    history = classifier.train(
        adj_list=train_adj,
        features_list=train_feat,
        labels=train_labels,
        epochs=100,
        batch_size=32,
    )

    # ══════════════════════════════════════════════
    # [4/4] 모델 평가
    # ══════════════════════════════════════════════
    print("\n[4/4] 모델 평가 중...")

    # 테스트 데이터로 예측
    correct = 0
    tp = 0  # True Positive (봇을 봇으로 정확히 예측)
    fp = 0  # False Positive (사람을 봇으로 잘못 예측)
    fn = 0  # False Negative (봇을 사람으로 잘못 예측)

    for i in range(len(test_labels)):
        prediction, confidence = classifier.predict(
            test_adj[i], test_feat[i]
        )
        predicted_label = 1 if prediction == "bot" else 0
        true_label = test_labels[i]

        if predicted_label == true_label:
            correct += 1

        if predicted_label == 1 and true_label == 1:
            tp += 1
        elif predicted_label == 1 and true_label == 0:
            fp += 1
        elif predicted_label == 0 and true_label == 1:
            fn += 1

    # 평가 지표 계산
    accuracy = correct / len(test_labels)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_score = (
        2 * precision * recall / max(precision + recall, 1e-8)
    )

    print(f"\n{'─' * 40}")
    print(f"  테스트 결과 ({len(test_labels)}개 샘플)")
    print(f"{'─' * 40}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1_score:.4f}")
    print(f"{'─' * 40}")

    # 논문 결과와 비교
    print(f"\n  논문 결과 (참고):")
    print(f"    Accuracy:  0.9845")
    print(f"    Precision: 0.9791")
    print(f"    Recall:    0.9833")
    print(f"    F1-Score:  0.9812")
    print(f"    (실제 데이터 78만건 기준, 샘플 데이터와 차이가 있을 수 있음)")

    # ── 모델 저장 ──
    model_dir = os.path.join(os.path.dirname(__file__), "trained_model")
    classifier.save_weights(model_dir)

    print(f"\n학습 완료! 모델이 {model_dir}에 저장되었습니다.")
    print("서버 실행: python server/app.py")


if __name__ == "__main__":
    main()
