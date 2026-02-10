"""
BOTracle 2단계 - 기술적 특징 분석 서버

클라이언트(브라우저)에서 수집한 기술적 특징을 받아
SGAN 모델로 봇/사람을 판별하는 REST API 서버입니다.

실행 방법:
    cd 02-TechnicalFeatureAnalysis
    pip install -r requirements.txt
    python server/app.py

API 엔드포인트:
    POST /api/analyze  - 특징 데이터를 받아 봇/사람 판별
    GET  /api/health   - 서버 상태 확인
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from sgan.preprocessor import FeaturePreprocessor
from sgan.predictor import SGANPredictor

app = Flask(__name__)
CORS(app)  # 브라우저의 CORS 제한 해제 (개발용)

# ──────────────────────────────────────────────
# 전처리기와 예측기 초기화
# ──────────────────────────────────────────────
preprocessor = FeaturePreprocessor()
predictor = SGANPredictor()

# SGAN 판정 신뢰도 임계값 (논문의 λ)
# 이 값 이상이면 최종 판정, 미만이면 3단계(DGCNN)로 전달
CONFIDENCE_THRESHOLD = 0.7


@app.route("/api/health", methods=["GET"])
def health():
    """서버 상태 확인"""
    return jsonify({
        "status": "ok",
        "model_loaded": predictor.is_loaded(),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    클라이언트에서 수집한 기술적 특징을 분석합니다.

    요청 Body (JSON):
        {
            "browser_height": 900,
            "browser_width": 1440,
            "user_agent": "Mozilla/5.0 ...",
            "java_enabled": false,
            "webdriver": false,
            "automation_flags": [],
            "platform": "MacIntel",
            "language": "ko-KR",
            "cookie_enabled": true,
            "timezone_offset": -540,
            "timestamp": 1707580800000
        }

    응답 (JSON):
        {
            "prediction": "human" | "bot",
            "confidence": 0.95,
            "above_threshold": true,
            "details": { ... }
        }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "요청 데이터가 없습니다"}), 400

    # 1. 전처리: 원시 데이터 → 수치 벡터
    feature_vector = preprocessor.transform(data)

    # 2. SGAN 예측
    prediction, confidence = predictor.predict(feature_vector)

    # 3. 신뢰도 임계값(λ) 비교
    above_threshold = confidence >= CONFIDENCE_THRESHOLD

    return jsonify({
        "prediction": prediction,          # "bot" 또는 "human"
        "confidence": round(confidence, 4),
        "above_threshold": above_threshold,  # True면 최종 판정
        "details": {
            "feature_vector": feature_vector.tolist(),
            "threshold": CONFIDENCE_THRESHOLD,
            "note": "최종 판정" if above_threshold else "신뢰도 부족 → 3단계(DGCNN) 분석 필요",
        },
    })


if __name__ == "__main__":
    print("=" * 50)
    print("BOTracle 2단계 - 기술적 특징 분석 서버")
    print(f"신뢰도 임계값 λ = {CONFIDENCE_THRESHOLD}")
    print(f"SGAN 모델 로드: {'완료' if predictor.is_loaded() else '샘플 모드 (학습된 모델 없음)'}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
