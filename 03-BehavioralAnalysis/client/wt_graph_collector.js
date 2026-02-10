/**
 * BOTracle 3단계 - WT Graph 수집기 (클라이언트)
 *
 * 브라우저에서 실행되어 사용자의 페이지 탐색(네비게이션) 데이터를 수집하고
 * 서버로 전송합니다. 서버에서 이 데이터를 모아 WT Graph를 구성합니다.
 *
 * 2단계의 feature_extractor.js가 "기술적 특징"을 한 번 수집했다면,
 * 3단계의 wt_graph_collector.js는 "탐색 행동"을 지속적으로 수집합니다.
 *
 * ═══════════════════════════════════════════════════════════
 *   WT Graph 데이터 수집 흐름
 * ═══════════════════════════════════════════════════════════
 *
 *   사용자가 페이지를 이동할 때마다:
 *     1. 현재 페이지 이름(URL 경로) 추출
 *     2. 이전 페이지 이름 기록
 *     3. 페이지 유형 추론 (home, product, category 등)
 *     4. 타임스탬프 기록
 *     5. 서버의 /api/hit 으로 전송
 *
 *   서버에서:
 *     hit 데이터를 받아 → 세션별 WT Graph에 노드/엣지 추가
 *
 * 사용법:
 *   <script src="wt_graph_collector.js"></script>
 *   <script>
 *     const collector = new WTGraphCollector('http://localhost:5001/api/hit');
 *     collector.recordHit('home', 'home', 'Welcome Page');
 *   </script>
 */

class WTGraphCollector {
  /**
   * @param {string} serverUrl - hit 데이터를 전송할 서버 URL
   * @param {string} sessionId - 세션 ID (미지정 시 자동 생성)
   */
  constructor(serverUrl, sessionId) {
    // ── 서버 URL ──
    this.serverUrl = serverUrl || "http://localhost:5001/api/hit";

    // ── 세션 ID ──
    // 같은 세션의 hit들을 서버에서 하나의 WT Graph로 묶기 위한 식별자
    // crypto.randomUUID()가 지원되지 않는 환경을 위한 폴백 포함
    this.sessionId = sessionId || this._generateSessionId();

    // ── 탐색 기록 ──
    // 이전 페이지 이름 (첫 방문 시 null)
    this.previousPagename = null;

    // 세션의 첫 번째 페이지 이름
    this.firstHitPagename = null;

    // 총 히트 수
    this.hitCount = 0;

    // 히트 기록 (로컬 로그용)
    this.hitLog = [];
  }

  /**
   * 페이지 방문을 기록하고 서버로 전송합니다.
   *
   * 사용자가 새 페이지로 이동할 때마다 이 메서드를 호출하세요.
   *
   * @param {string} pageName  - 현재 페이지 이름 (예: "category/electronics")
   * @param {string} pageType  - 페이지 유형 (예: "category", "product")
   * @param {string} pageTitle - 페이지 제목 (RAKE 키워드 추출용)
   * @returns {Promise<object|null>} 서버 응답 또는 null (실패 시)
   */
  async recordHit(pageName, pageType, pageTitle) {
    if (!pageName) return null;

    // 첫 번째 히트 기록
    if (this.firstHitPagename === null) {
      this.firstHitPagename = pageName;
    }

    this.hitCount++;

    // ── hit 데이터 구성 (논문 Table 1) ──
    const hitData = {
      sessionId: this.sessionId,
      detailedPagename: pageName, // 현재 페이지
      previousPagename: this.previousPagename, // 이전 페이지 (null 가능)
      firstHitPagename: this.firstHitPagename, // 세션 첫 페이지
      timestamp: Date.now(), // 방문 시각 (밀리초)
      pageType: pageType || this._inferPageType(pageName), // 페이지 유형
      pageTitle: pageTitle || document.title || pageName, // 페이지 제목
    };

    // 로컬 로그에 기록
    this.hitLog.push(hitData);

    // 이전 페이지 업데이트
    this.previousPagename = pageName;

    // 서버로 전송
    return await this._sendHit(hitData);
  }

  /**
   * 현재 세션의 분석을 요청합니다.
   *
   * 서버의 /api/analyze 엔드포인트로 세션 ID를 전송하여
   * DGCNN 분석 결과를 받아옵니다.
   *
   * @returns {Promise<object|null>} 분석 결과 또는 null
   */
  async requestAnalysis() {
    const analyzeUrl = this.serverUrl.replace("/api/hit", "/api/analyze");

    try {
      const response = await fetch(analyzeUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: this.sessionId }),
      });

      return await response.json();
    } catch (error) {
      console.error("분석 요청 실패:", error);
      return null;
    }
  }

  /**
   * 현재 세션의 WT Graph 상태를 조회합니다.
   *
   * @returns {Promise<object|null>} 그래프 상태 또는 null
   */
  async getSessionState() {
    const sessionUrl = this.serverUrl.replace(
      "/api/hit",
      `/api/session/${this.sessionId}`
    );

    try {
      const response = await fetch(sessionUrl);
      return await response.json();
    } catch (error) {
      console.error("세션 조회 실패:", error);
      return null;
    }
  }

  /**
   * 세션 ID를 반환합니다.
   * @returns {string}
   */
  getSessionId() {
    return this.sessionId;
  }

  /**
   * 로컬 히트 기록을 반환합니다.
   * @returns {Array}
   */
  getHitLog() {
    return this.hitLog;
  }

  // ══════════════════════════════════════════════
  // 내부 메서드
  // ══════════════════════════════════════════════

  /**
   * hit 데이터를 서버로 전송합니다.
   * @private
   */
  async _sendHit(hitData) {
    try {
      const response = await fetch(this.serverUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(hitData),
      });

      return await response.json();
    } catch (error) {
      console.error("히트 전송 실패:", error);
      return null;
    }
  }

  /**
   * 페이지 이름으로부터 페이지 유형을 추론합니다.
   *
   * URL 경로의 패턴을 분석하여 유형을 결정합니다.
   *
   * @param {string} pageName - 페이지 이름 (URL 경로)
   * @returns {string} 페이지 유형
   * @private
   */
  _inferPageType(pageName) {
    const name = pageName.toLowerCase();

    if (name === "home" || name === "" || name === "/") {
      return "home";
    } else if (name.startsWith("product/") || name.includes("product")) {
      return "product";
    } else if (name.startsWith("category/") || name.includes("category")) {
      return "category";
    } else if (name.includes("search")) {
      return "search";
    } else if (name.includes("cart")) {
      return "cart";
    } else if (name.includes("checkout")) {
      return "checkout";
    } else if (name.includes("account") || name.includes("mypage")) {
      return "account";
    } else {
      return "other";
    }
  }

  /**
   * 세션 ID를 생성합니다.
   *
   * crypto.randomUUID()를 우선 사용하고,
   * 지원되지 않는 환경에서는 간단한 랜덤 문자열을 생성합니다.
   *
   * @returns {string} UUID 형식의 세션 ID
   * @private
   */
  _generateSessionId() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    // 폴백: 간단한 랜덤 ID 생성
    return (
      "session-" +
      Math.random().toString(36).substring(2, 10) +
      "-" +
      Date.now().toString(36)
    );
  }
}

// ══════════════════════════════════════════════
// 사용 예시
// ══════════════════════════════════════════════
// const collector = new WTGraphCollector('http://localhost:5001/api/hit');
//
// // 페이지 이동 시마다 기록
// collector.recordHit('home', 'home', 'Welcome Page');
// collector.recordHit('category/electronics', 'category', 'Electronics');
// collector.recordHit('product/laptop-pro', 'product', 'Laptop Pro');
//
// // 세션 분석 요청
// collector.requestAnalysis().then(result => {
//   console.log('분석 결과:', result);
//   // result = { prediction: 'human', confidence: 0.85, ... }
// });
