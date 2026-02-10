/**
 * BOTracle 2단계 - 기술적 특징 추출기 (클라이언트)
 *
 * 브라우저에서 실행되어 기술적 특징(technical features)을 수집하고
 * 서버로 전송합니다.
 *
 * 논문에서 가장 중요한 특징 (Feature Importance 순위):
 *   1위: 브라우저 창 높이 (R²=0.542)
 *   2위: 브라우저 창 너비 (R²=0.287)
 *   3위: Java 지원 여부  (R²=0.082)
 *   5위: User Agent      (R²=0.024)
 *
 * 사용법:
 *   <script src="feature_extractor.js"></script>
 *   또는 HTML 파일에 직접 포함
 */

class TechnicalFeatureExtractor {
  constructor(serverUrl) {
    // 특징 데이터를 전송할 서버 주소
    this.serverUrl = serverUrl || "http://localhost:5000/api/analyze";
  }

  /**
   * 모든 기술적 특징을 수집합니다.
   * 논문 Section 3.2에서 SGAN이 사용하는 특징들입니다.
   */
  extractFeatures() {
    return {
      // ──────────────────────────────────────────────
      // 브라우저 창 크기 (논문에서 가장 중요한 특징)
      // 봇은 렌더링이 불필요하므로 최소 크기(예: 1x1)를 사용하는 경향
      // ──────────────────────────────────────────────
      browser_height: window.innerHeight,
      browser_width: window.innerWidth,

      // ──────────────────────────────────────────────
      // User Agent (브라우저 식별 문자열)
      // 봇은 'python-request' 같은 자동화 도구 UA를 사용하거나
      // 실제 브라우저와 기능이 불일치하는 위조된 UA를 사용
      // ──────────────────────────────────────────────
      user_agent: navigator.userAgent,

      // ──────────────────────────────────────────────
      // Java 지원 여부
      // 현대 브라우저는 Java를 지원하지 않음
      // Java가 지원된다고 보고하면 → 오래된 또는 위조된 UA 의심
      // ──────────────────────────────────────────────
      java_enabled: navigator.javaEnabled ? navigator.javaEnabled() : false,

      // ──────────────────────────────────────────────
      // 자동화 도구 탐지 플래그
      // Selenium, Puppeteer 등 자동화 도구의 흔적을 탐지
      // ──────────────────────────────────────────────
      webdriver: navigator.webdriver || false,
      automation_flags: this.detectAutomationFlags(),

      // ──────────────────────────────────────────────
      // 추가 브라우저 메타데이터
      // ──────────────────────────────────────────────
      platform: navigator.platform,
      language: navigator.language,
      cookie_enabled: navigator.cookieEnabled,
      timezone_offset: new Date().getTimezoneOffset(),

      // 수집 시각 (서버에서 요청 간격 분석에 사용)
      timestamp: Date.now(),
    };
  }

  /**
   * 브라우저 자동화 도구의 흔적을 탐지합니다.
   *
   * 각 자동화 도구는 브라우저에 고유한 전역 변수나 속성을 남깁니다.
   * 이를 확인하여 자동화 여부를 판단합니다.
   */
  detectAutomationFlags() {
    const flags = [];

    // Selenium/WebDriver: navigator.webdriver 속성이 true
    if (navigator.webdriver) {
      flags.push("webdriver");
    }

    // PhantomJS: 구형 헤드리스 브라우저, 고유 전역 변수 존재
    if (window.callPhantom || window._phantom) {
      flags.push("phantom");
    }

    // Nightmare.js: Electron 기반 자동화 도구
    if (window.__nightmare) {
      flags.push("nightmare");
    }

    // Headless Chrome: User Agent에 'HeadlessChrome' 포함
    if (navigator.userAgent.includes("HeadlessChrome")) {
      flags.push("headless_chrome");
    }

    // Playwright: 마이크로소프트의 브라우저 자동화 도구
    if (navigator.userAgent.includes("Playwright")) {
      flags.push("playwright");
    }

    // HTML 요소에 webdriver 속성이 설정된 경우
    if (document.documentElement.getAttribute("webdriver")) {
      flags.push("webdriver_attr");
    }

    // Chrome DevTools Protocol (CDP) 흔적
    // Puppeteer 등이 남기는 전역 변수
    if (window.cdc_adoQpoasnfa76pfcZLmcfl_Array) {
      flags.push("cdp");
    }

    return flags;
  }

  /**
   * 수집한 특징을 서버로 전송합니다.
   * 서버의 /api/analyze 엔드포인트로 POST 요청을 보냅니다.
   */
  async sendToServer() {
    const features = this.extractFeatures();

    try {
      const response = await fetch(this.serverUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });

      const result = await response.json();
      return result;
    } catch (error) {
      console.error("특징 전송 실패:", error);
      return null;
    }
  }
}

// ──────────────────────────────────────────────
// 사용 예시
// ──────────────────────────────────────────────
// const extractor = new TechnicalFeatureExtractor('http://localhost:5000/api/analyze');
// extractor.sendToServer().then(result => {
//   console.log('분석 결과:', result);
//   // result = { prediction: 'human', confidence: 0.95, ... }
// });
