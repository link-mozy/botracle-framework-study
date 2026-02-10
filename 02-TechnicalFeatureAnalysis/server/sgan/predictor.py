"""
BOTracle 2단계 - SGAN 예측기 (Predictor)

학습된 SGAN 모델을 사용하여 새로운 데이터를 예측합니다.
모델 파일이 없으면 간단한 규칙 기반 폴백을 사용합니다.
"""

import os
import numpy as np
import tensorflow as tf

from sgan.model import SGAN

# 학습된 모델 가중치 저장 경로
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_model")


class SGANPredictor:
    """
    SGAN 모델을 로드하고 예측을 수행하는 클래스.

    학습된 모델이 있으면 SGAN을 사용하고,
    없으면 규칙 기반 폴백으로 동작합니다.
    """

    def __init__(self, feature_dim=11):
        self.feature_dim = feature_dim
        self.sgan = None
        self._loaded = False

        # 학습된 모델 로드 시도
        if os.path.exists(MODEL_DIR):
            try:
                self.sgan = SGAN(feature_dim=feature_dim)
                self.sgan.load_weights(MODEL_DIR)
                self._loaded = True
            except Exception as e:
                print(f"모델 로드 실패 (규칙 기반 모드로 동작): {e}")
                self.sgan = None
                self._loaded = False

    def is_loaded(self) -> bool:
        """학습된 SGAN 모델이 로드되었는지 확인"""
        return self._loaded

    def predict(self, feature_vector: np.ndarray) -> tuple:
        """
        봇/사람을 예측합니다.

        Args:
            feature_vector: 전처리된 특징 벡터 (shape: [11])

        Returns:
            (prediction, confidence)
            prediction: "bot" 또는 "human"
            confidence: 신뢰도 (0.0 ~ 1.0)
        """
        if self._loaded and self.sgan is not None:
            return self.sgan.predict(feature_vector)
        else:
            return self._rule_based_predict(feature_vector)

    def _rule_based_predict(self, feature_vector: np.ndarray) -> tuple:
        """
        규칙 기반 폴백 예측.

        SGAN 모델이 없을 때 논문의 휴리스틱과 Feature Importance를
        기반으로 간단한 규칙으로 판별합니다.

        규칙 (논문 Table 4 기반):
          1. 브라우저 창 크기가 매우 작으면 → 봇
          2. WebDriver 플래그가 활성화되면 → 봇
          3. 자동화 도구가 탐지되면 → 봇
          4. python/curl/bot UA면 → 봇
        """
        v = feature_vector
        if len(v.shape) > 1:
            v = v[0]

        bot_score = 0.0

        # 브라우저 창 높이가 매우 작음 (50px 미만 → 정규화값 0.023)
        if v[0] < 0.025:
            bot_score += 0.4

        # 브라우저 창 너비가 매우 작음
        if v[1] < 0.025:
            bot_score += 0.3

        # WebDriver 활성화
        if v[3] > 0.5:
            bot_score += 0.3

        # 자동화 플래그 존재
        if v[4] > 0.0:
            bot_score += 0.2

        # 쿠키 비활성화
        if v[5] < 0.5:
            bot_score += 0.1

        # 비표준 UA (python=index 7, curl=8, bot=9)
        if v[7] > 0.5 or v[8] > 0.5 or v[9] > 0.5:
            bot_score += 0.2

        bot_score = min(bot_score, 1.0)

        if bot_score >= 0.5:
            return "bot", bot_score
        else:
            return "human", 1.0 - bot_score
