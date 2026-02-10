"""
BOTracle 2단계 - 특징 전처리기 (Feature Preprocessor)

클라이언트에서 수집한 원시 데이터를 SGAN 모델이 이해할 수 있는
1차원 수치 벡터로 변환합니다.

논문 Section 3.2:
    "preprocessing involves discarding irrelevant features and
     transforming the data into a one-dimensional numerical vector
     using flagging, integer, and one-hot encoding"

전처리 과정:
    원시 JSON 데이터
         │
         ▼
    불필요한 필드 제거 (timestamp 등)
         │
         ▼
    인코딩 변환:
      - Flagging: 이진값 (True/False → 1/0)
      - 정규화:   연속값을 0~1 범위로 변환
      - One-Hot:  범주형 데이터를 이진 벡터로 변환
         │
         ▼
    수치 벡터 (numpy array)
"""

import numpy as np


# ──────────────────────────────────────────────
# 논문 Table 4의 주요 User Agent 카테고리
# One-Hot 인코딩에 사용합니다.
# ──────────────────────────────────────────────
USER_AGENT_CATEGORIES = [
    "Mozilla/5.0",  # 표준 브라우저 (Chrome, Firefox, Safari 등)
    "python",       # Python 요청 라이브러리 (python-requests 등)
    "curl",         # curl 커맨드라인 도구
    "bot",          # 이름에 'bot'이 포함된 UA (Googlebot 등)
    "other",        # 위에 해당하지 않는 경우
]

# 브라우저 창 크기 정규화 기준 (일반적인 최대 해상도)
MAX_BROWSER_HEIGHT = 2160  # 4K 세로
MAX_BROWSER_WIDTH = 3840   # 4K 가로


class FeaturePreprocessor:
    """
    원시 특징 데이터를 SGAN 입력 벡터로 변환하는 전처리기.

    변환 결과 벡터 구조 (11차원):
        [0] browser_height  (정규화 0~1)
        [1] browser_width   (정규화 0~1)
        [2] java_enabled    (0 또는 1)
        [3] webdriver       (0 또는 1)
        [4] automation_count (정규화 0~1)
        [5] cookie_enabled  (0 또는 1)
        [6] ua_mozilla      (0 또는 1)  ─┐
        [7] ua_python       (0 또는 1)   │ User Agent
        [8] ua_curl         (0 또는 1)   │ One-Hot
        [9] ua_bot          (0 또는 1)   │ 인코딩
       [10] ua_other        (0 또는 1)  ─┘
    """

    def __init__(self):
        # 벡터의 차원 수 (SGAN 모델의 입력 크기와 일치해야 함)
        self.feature_dim = 11

    def transform(self, raw_data: dict) -> np.ndarray:
        """
        원시 JSON 데이터를 수치 벡터로 변환합니다.

        Args:
            raw_data: 클라이언트에서 수집한 특징 딕셔너리

        Returns:
            numpy array (shape: [11,])
        """
        vector = np.zeros(self.feature_dim, dtype=np.float32)

        # ── 1. 브라우저 창 크기 (정규화) ──
        # 논문에서 가장 중요한 특징 (R²=0.542, 0.287)
        # 값을 0~1 범위로 변환합니다.
        # 예: 900px / 2160 = 0.417
        vector[0] = min(raw_data.get("browser_height", 0) / MAX_BROWSER_HEIGHT, 1.0)
        vector[1] = min(raw_data.get("browser_width", 0) / MAX_BROWSER_WIDTH, 1.0)

        # ── 2. Flagging (이진값 인코딩) ──
        # True → 1.0, False → 0.0
        vector[2] = 1.0 if raw_data.get("java_enabled", False) else 0.0
        vector[3] = 1.0 if raw_data.get("webdriver", False) else 0.0
        vector[5] = 1.0 if raw_data.get("cookie_enabled", True) else 0.0

        # ── 3. 자동화 플래그 수 (정규화) ──
        # 탐지된 자동화 도구 수를 정규화 (최대 7개 기준)
        flags = raw_data.get("automation_flags", [])
        vector[4] = min(len(flags) / 7.0, 1.0)

        # ── 4. User Agent One-Hot 인코딩 ──
        # 문자열을 숫자로 변환하는 방법 중 하나입니다.
        # 해당하는 카테고리만 1, 나머지는 0으로 표시합니다.
        #
        # 예: "Mozilla/5.0 (Windows...)" → [1, 0, 0, 0, 0]
        #     "python-requests/2.28"     → [0, 1, 0, 0, 0]
        #     "알 수 없는 UA"             → [0, 0, 0, 0, 1]
        ua = raw_data.get("user_agent", "").lower()
        ua_index = self._classify_user_agent(ua)
        vector[6 + ua_index] = 1.0

        return vector

    def _classify_user_agent(self, ua: str) -> int:
        """
        User Agent 문자열을 카테고리 인덱스로 분류합니다.

        Returns:
            0: Mozilla/5.0 (일반 브라우저)
            1: python (자동화 라이브러리)
            2: curl (커맨드라인)
            3: bot (크롤러)
            4: other (기타)
        """
        if "mozilla/5.0" in ua:
            return 0
        elif "python" in ua:
            return 1
        elif "curl" in ua:
            return 2
        elif "bot" in ua:
            return 3
        else:
            return 4

    def transform_batch(self, data_list: list) -> np.ndarray:
        """
        여러 데이터를 한 번에 변환합니다.

        Args:
            data_list: 특징 딕셔너리의 리스트

        Returns:
            numpy array (shape: [len(data_list), 11])
        """
        return np.array([self.transform(d) for d in data_list])
