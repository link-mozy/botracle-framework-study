"""
BOTracle 3단계 - DGCNN 예측기 (Predictor)

학습된 DGCNN 모델을 사용하여 WT Graph를 분류합니다.
모델 파일이 없으면 규칙 기반 폴백으로 동작합니다.

2단계의 SGANPredictor와 동일한 역할입니다.
"""

import os
import numpy as np

from dgcnn.model import DGCNNClassifier
from dgcnn.preprocessor import GraphPreprocessor
from dgcnn.wt_graph import WTGraph

# 학습된 모델 가중치 저장 경로
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_model")


class DGCNNPredictor:
    """
    DGCNN 모델을 로드하고 예측을 수행하는 클래스.

    학습된 모델이 있으면 DGCNN을 사용하고,
    없으면 규칙 기반 폴백으로 동작합니다.
    """

    def __init__(self, input_dim: int = 12):
        self.input_dim = input_dim
        self.preprocessor = GraphPreprocessor()
        self.classifier = None
        self._loaded = False

        # 학습된 모델 로드 시도
        if os.path.exists(MODEL_DIR):
            try:
                self.classifier = DGCNNClassifier(input_dim=input_dim)
                self.classifier.load_weights(MODEL_DIR)
                self._loaded = True
            except Exception as e:
                print(f"모델 로드 실패 (규칙 기반 모드로 동작): {e}")
                self.classifier = None
                self._loaded = False

    def is_loaded(self) -> bool:
        """학습된 DGCNN 모델이 로드되었는지 확인"""
        return self._loaded

    def predict(self, wt_graph: WTGraph) -> tuple:
        """
        WT Graph에서 봇/사람을 예측합니다.

        Args:
            wt_graph: WTGraph 객체

        Returns:
            (prediction, confidence)
            prediction: "bot" 또는 "human"
            confidence: 신뢰도 (0.0 ~ 1.0)
        """
        if self._loaded and self.classifier is not None:
            # 학습된 DGCNN 모델 사용
            adj, features = self.preprocessor.transform(wt_graph)
            return self.classifier.predict(adj, features)
        else:
            # 규칙 기반 폴백
            return self._rule_based_predict(wt_graph)

    def _rule_based_predict(self, wt_graph: WTGraph) -> tuple:
        """
        규칙 기반 폴백 예측.

        DGCNN 모델이 없을 때 WT Graph 메트릭을 기반으로
        간단한 규칙으로 봇/사람을 판별합니다.

        ═══════════════════════════════════════════════
        봇 탐지 규칙 (논문 Section 3.3 기반)
        ═══════════════════════════════════════════════

        봇의 행동 패턴:
          1. 체계적 크롤링 → 많은 노드, 균일한 방문 분포
          2. 빠른 페이지 전환 → 짧은 타임스탬프 간격
          3. 높은 edge/node 비율 → 모든 경로를 탐색
          4. 균일한 페이지별 방문 횟수 → 각 페이지를 1번씩 방문
          5. 낮은 centrality 편차 → 구조가 평평함
        """
        metrics = wt_graph.extract_metrics()
        bot_score = 0.0

        node_count = metrics["node_count"]
        edge_count = metrics["edge_count"]
        total_hits = metrics["total_hits"]

        if node_count == 0:
            return "human", 0.5

        # ── 규칙 1: 빠른 페이지 전환 (짧은 타임스탬프 간격) ──
        # 사람: 평균 2~30초, 봇: 0.1~0.5초
        all_timestamps = []
        for node in wt_graph.nodes.values():
            all_timestamps.extend(node.timestamps)
        all_timestamps.sort()

        if len(all_timestamps) >= 2:
            intervals = [
                all_timestamps[i + 1] - all_timestamps[i]
                for i in range(len(all_timestamps) - 1)
            ]
            avg_interval = np.mean(intervals) if intervals else 10000

            # 평균 간격이 1초 미만이면 봇 의심
            if avg_interval < 1000:
                bot_score += 0.3
            elif avg_interval < 3000:
                bot_score += 0.1

        # ── 규칙 2: 높은 노드 커버리지 ──
        # 사이트의 많은 페이지를 방문했으면 봇 의심
        # (사람은 보통 전체 페이지의 일부만 방문)
        if node_count >= 10:
            bot_score += 0.2

        # ── 규칙 3: 균일한 페이지별 방문 횟수 ──
        # 봇은 각 페이지를 정확히 1번씩 방문하는 경향
        # 사람은 특정 페이지를 여러 번 방문
        hits_per_page = list(metrics["hits_per_sub_page"].values())
        if hits_per_page:
            hit_std = np.std(hits_per_page) if len(hits_per_page) > 1 else 0
            hit_mean = np.mean(hits_per_page)

            # 변동이 거의 없으면 (모든 페이지 방문 횟수가 비슷) 봇 의심
            if hit_mean > 0 and hit_std / max(hit_mean, 1) < 0.3:
                bot_score += 0.2

        # ── 규칙 4: 높은 edge/node 비율 ──
        # 봇은 모든 경로를 탐색하므로 edge가 많음
        if node_count > 1:
            edge_ratio = edge_count / (node_count * (node_count - 1))
            if edge_ratio > 0.5:
                bot_score += 0.2

        # ── 규칙 5: 시간 간격의 규칙성 ──
        # 봇은 일정한 간격, 사람은 불규칙한 간격
        if len(all_timestamps) >= 3:
            intervals = [
                all_timestamps[i + 1] - all_timestamps[i]
                for i in range(len(all_timestamps) - 1)
            ]
            if intervals:
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                # 변동계수(CV)가 작으면 규칙적 → 봇 의심
                if interval_mean > 0 and interval_std / interval_mean < 0.2:
                    bot_score += 0.2

        bot_score = min(bot_score, 1.0)

        if bot_score >= 0.5:
            return "bot", bot_score
        else:
            return "human", 1.0 - bot_score
