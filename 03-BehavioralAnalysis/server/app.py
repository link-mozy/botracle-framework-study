"""
BOTracle 3단계 - 행위 기반 분석 서버

클라이언트(브라우저)에서 수집한 페이지 네비게이션 데이터를 받아
세션별 WT Graph를 구성하고, DGCNN 모델로 봇/사람을 판별하는 REST API 서버입니다.

실행 방법:
    cd 03-BehavioralAnalysis
    pip install -r requirements.txt
    python server/app.py

API 엔드포인트:
    POST /api/hit          - 페이지 방문(hit) 데이터 수신
    POST /api/analyze      - 세션의 WT Graph를 DGCNN으로 분석
    GET  /api/session/<id> - 세션의 WT Graph 상태 조회
    GET  /api/health       - 서버 상태 확인
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from dgcnn.wt_graph import WTGraph
from dgcnn.predictor import DGCNNPredictor

app = Flask(__name__)
CORS(app)  # 브라우저의 CORS 제한 해제 (개발용)

# ──────────────────────────────────────────────
# 세션 저장소 + 예측기 초기화
# ──────────────────────────────────────────────

# 세션별 WT Graph를 메모리에 저장
# key: session_id (문자열), value: WTGraph 객체
sessions: dict[str, WTGraph] = {}

# DGCNN 예측기 초기화
predictor = DGCNNPredictor()

# DGCNN 판정 신뢰도 임계값 (논문의 λ)
# 이 값 이상이면 최종 판정, 미만이면 추가 히트 필요
CONFIDENCE_THRESHOLD = 0.7


@app.route("/api/health", methods=["GET"])
def health():
    """서버 상태 확인"""
    return jsonify({
        "status": "ok",
        "model_loaded": predictor.is_loaded(),
        "active_sessions": len(sessions),
    })


@app.route("/api/hit", methods=["POST"])
def receive_hit():
    """
    클라이언트에서 페이지 방문(hit) 데이터를 수신합니다.
    해당 세션의 WT Graph에 hit을 추가합니다.

    요청 Body (JSON):
        {
            "sessionId": "uuid-string",
            "detailedPagename": "category/electronics",
            "previousPagename": "home",
            "timestamp": 1707580800000,
            "pageType": "category",
            "pageTitle": "Electronics & Gadgets"
        }

    응답 (JSON):
        {
            "status": "ok",
            "session_id": "uuid-string",
            "graph_summary": {
                "node_count": 3,
                "edge_count": 2,
                "total_hits": 4,
                "first_page": "home"
            }
        }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "요청 데이터가 없습니다"}), 400

    session_id = data.get("sessionId")
    if not session_id:
        return jsonify({"error": "sessionId가 필요합니다"}), 400

    # 세션의 WT Graph 가져오기 (없으면 새로 생성)
    if session_id not in sessions:
        sessions[session_id] = WTGraph()

    graph = sessions[session_id]

    # hit 추가 (WT Graph가 동적으로 확장됨)
    graph.add_hit(data)

    return jsonify({
        "status": "ok",
        "session_id": session_id,
        "graph_summary": graph.get_summary(),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    세션의 WT Graph를 DGCNN으로 분석합니다.

    요청 Body (JSON):
        {
            "sessionId": "uuid-string"
        }

    응답 (JSON):
        {
            "prediction": "human" | "bot",
            "confidence": 0.95,
            "above_threshold": true,
            "details": {
                "graph_summary": { ... },
                "metrics": { ... },
                "threshold": 0.7,
                "note": "최종 판정"
            }
        }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "요청 데이터가 없습니다"}), 400

    session_id = data.get("sessionId")
    if not session_id:
        return jsonify({"error": "sessionId가 필요합니다"}), 400

    if session_id not in sessions:
        return jsonify({"error": "해당 세션을 찾을 수 없습니다"}), 404

    graph = sessions[session_id]

    # DGCNN 예측 (또는 규칙 기반 폴백)
    prediction, confidence = predictor.predict(graph)

    # 신뢰도 임계값(λ) 비교
    above_threshold = confidence >= CONFIDENCE_THRESHOLD

    # 그래프 메트릭 추출 (응답에 포함)
    metrics = graph.extract_metrics()

    # 직렬화 가능하도록 변환
    serialized_metrics = {
        "node_count": metrics["node_count"],
        "edge_count": metrics["edge_count"],
        "total_hits": metrics["total_hits"],
        "page_type_distribution": metrics["page_type_distribution"],
        "session_topics": metrics["session_topics"],
    }

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "above_threshold": above_threshold,
        "details": {
            "graph_summary": graph.get_summary(),
            "metrics": serialized_metrics,
            "threshold": CONFIDENCE_THRESHOLD,
            "note": "최종 판정" if above_threshold else "신뢰도 부족 → 추가 히트 필요",
        },
    })


@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id):
    """
    세션의 WT Graph 상태를 조회합니다.
    클라이언트에서 실시간 그래프 시각화에 사용합니다.

    응답 (JSON):
        {
            "session_id": "uuid-string",
            "graph": {
                "nodes": [...],
                "edges": [...],
                "summary": { ... }
            }
        }
    """
    if session_id not in sessions:
        return jsonify({"error": "해당 세션을 찾을 수 없습니다"}), 404

    graph = sessions[session_id]

    return jsonify({
        "session_id": session_id,
        "graph": graph.to_dict(),
    })


if __name__ == "__main__":
    print("=" * 50)
    print("BOTracle 3단계 - 행위 기반 분석 서버")
    print(f"신뢰도 임계값 λ = {CONFIDENCE_THRESHOLD}")
    print(f"DGCNN 모델 로드: {'완료' if predictor.is_loaded() else '샘플 모드 (학습된 모델 없음)'}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=True)
