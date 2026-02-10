# Technical Feature Analysis (기술적 특징 분석)

## 프로젝트 개요

BOTracle 프레임워크의 2단계인 **기술적 특징 기반 봇 탐지** (SGAN) 프로젝트입니다.
BOTracle 파이프라인에서 휴리스틱으로 분류되지 않은 요청(hit)을 SGAN 모델로 분석하여 봇/사람을 판별합니다.

### 주요 기능

- 웹 클라이언트에서 기술적 특징(technical features) 추출
- 특징 데이터 전처리 및 수치 벡터 변환
- SGAN을 통한 봇/사람 분류

### 기술 스택

- 언어: JavaScript (클라이언트 특징 수집), Python (서버, SGAN 학습)
- API 방식: RESTful API
- ML 프레임워크: TensorFlow / Keras

## BOTracle 파이프라인에서의 위치

```plaintext
  요청(Hit) 수신
       │
       ▼
┌──────────────┐    분류됨
│  휴리스틱     │──────────────▶ 봇/사람 판정
│  (1단계)     │
└──────┬───────┘
       │ 미분류
       ▼
┌──────────────┐    신뢰도 ≥ λ
│  SGAN        │──────────────▶ 봇/사람 판정   ◀── 이 프로젝트
│  (2단계)     │
└──────┬───────┘
       │ 신뢰도 < λ
       ▼
┌──────────────┐    신뢰도 ≥ λ
│  DGCNN       │──────────────▶ 봇/사람 판정
│  (3단계)     │
└──────────────┘
```

SGAN은 **개별 요청(hit) 단위**로 기술적 특징을 분석합니다. 신뢰도 임계값 λ ∈ [0, 1]을 초과하면 최종 판정을 내리고, 미달하면 세션 단위의 DGCNN(3단계)으로 넘깁니다.

## 봇 특징 분류 체계 (Bot Feature Analysis)

논문에서는 웹 클라이언트의 속성을 다음과 같이 분류합니다:

### 비행동적 속성 (Non-behavioral Attributes)

| 분류 | 설명 | 예시 |
|------|------|------|
| **Identity Attributes** (신원 속성) | 사용자 식별 정보 | IP 주소, 위치, 사용자 이름 |
| **Technical Attributes** (기술 속성) | 기술적 측면 정보 | User Agent, 화면 크기, Java 지원 여부 |

### 행동적 속성 (Behavioral Attributes)

| 분류 | 설명 | 예시 |
|------|------|------|
| **Traversal Attributes** (탐색 속성) | 웹사이트 내비게이션 패턴 | 방문 페이지 경로, BFS 패턴 |
| **Interaction Attributes** (상호작용 속성) | 사이트 요소와의 상호작용 | 프로모션 코드 사용, 구매 행위 |
| **Visit Attributes** (방문 속성) | 방문 빈도와 패턴 | 방문 횟수, 방문 주기 |

> SGAN(2단계)은 **기술적 속성(Technical Attributes)**을 주로 활용합니다.
> 행동적 속성은 3단계(DGCNN)에서 WT 그래프로 분석합니다.

## SGAN 개념

**SGAN (Semi-Supervised Generative Adversarial Network)**은 레이블이 있는 데이터와 없는 데이터를 모두 활용하여 학습합니다. 웹 로그 데이터셋은 대부분 레이블이 없기 때문에, SGAN이 적합합니다.

### SGAN 구조

```plaintext
                                    ┌─────────────────────────┐
                                    │      Discriminator       │
                                    │   (공유 신경망 + 2개 헤드)  │
┌─────────────┐                     │                         │
│  Generator  │──가짜 데이터──▶      │  ┌───────────────────┐  │
│  (생성자)    │                     │  │ 공유 Hidden Layers │  │
└─────────────┘                     │  └────────┬──────────┘  │
      ▲                             │           │             │
      │                             │     ┌─────┴─────┐       │
  100차원                           │     ▼           ▼       │
  노이즈 입력                       │ ┌────────┐ ┌────────┐   │
                                    │ │판별 헤드│ │분류 헤드│   │
  실제 데이터 ──────────────▶       │ │(D head)│ │(C head)│   │
  (레이블 有/無)                    │ └────┬───┘ └────┬───┘   │
                                    │      │          │       │
                                    └──────┼──────────┼───────┘
                                           ▼          ▼
                                     p_real ∈ [0,1]  [p1,...,pn]
                                     (진짜/가짜)     (봇/사람 확률)
```

### 핵심 구성 요소

**판별자 (Discriminator)** - 두 개의 헤드가 공유 신경망을 사용:
- **판별 헤드 (Discriminator Head)**: 입력 데이터가 실제인지 생성자가 만든 가짜인지 판별 → p_real ∈ [0, 1]
- **분류 헤드 (Classifier Head)**: 데이터 포인트의 클래스(봇/사람) 예측 → Softmax 활성화

**생성자 (Generator)**:
- 100차원 잠재 벡터(latent vector)로부터 실제처럼 보이는 가짜 특징 벡터를 생성
- 생성된 가짜 데이터가 판별자를 속이지 못하면, 분류기의 레이블 데이터를 보강하는 효과

### SGAN의 장점 (봇 탐지에서)

1. **레이블 부족 문제 해결**: 웹 로그 데이터의 대부분은 레이블이 없음 → 비지도 + 지도 학습 결합
2. **생성자의 보조 역할**: 생성자가 만든 현실적인 가짜 데이터가 분류기 학습을 보강
3. **공유 가중치의 효과**: 판별 헤드와 분류 헤드가 공유 레이어를 사용하여, 비지도 학습의 이점이 분류 성능 향상에 기여

## 전처리 과정

SGAN에 입력하기 전, 원시 데이터를 1차원 수치 벡터로 변환합니다:

```plaintext
원시 요청 데이터
     │
     ▼
불필요한 특징 제거
     │
     ▼
인코딩 변환:
  - Flagging (이진값)
  - Integer Encoding (정수)
  - One-Hot Encoding (범주형)
     │
     ▼
희소 범주값 → 특수 카테고리로 통합
     │
     ▼
1차원 수치 벡터 (SGAN 입력)
```

## 논문 기반 SGAN 아키텍처 (Section 4.1)

### Discriminator (판별자)

| 레이어 | 유형 | 설정 |
|--------|------|------|
| 1 | Dense | 100 units, Sigmoid |
| 2 | LeakyReLU | α = 0.2 |
| 3 | Dense | 100 units, Sigmoid |
| 4 | LeakyReLU | α = 0.2 |
| 5 | Dense | 100 units, Sigmoid |
| 6 | LeakyReLU | α = 0.2 |
| 7 | Dropout | p = 0.4 |
| **출력** | **판별 헤드** | ExpSum 활성화 → p_real |
| **출력** | **분류 헤드** | Softmax 활성화 → [p_bot, p_human] |

- 손실 함수: Binary Cross-Entropy (판별), Sparse Categorical Cross-Entropy (분류)
- 최적화: Adam (α = 0.0002, β₁ = 0.5)

### Generator (생성자)

| 레이어 | 유형 | 설정 |
|--------|------|------|
| 입력 | Dense | 100차원 latent vector → 200 units, Sigmoid |
| 출력 | Dense | 특징 수 만큼의 units, ReLU |

- 손실 함수: Binary Cross-Entropy
- 최적화: Adam (α = 0.0002, β₁ = 0.5)

### 수학적 기반

**Cross Entropy Loss (분류)**:

```
L(Yₖ, pₖ) = (-1) · Σₖ Yₖ · log(pₖ)
```
- Yₖ: 실제 클래스 값 (해당 클래스이면 1, 아니면 0)
- pₖ: Softmax로 계산된 클래스 멤버십 확률

**Softmax 함수**:

```
S(Z, zᵢ) = e^zᵢ / Σ e^zₖ
```

**ExpSum 활성화 함수** (판별 헤드):

```
E(Z) = F(Z) / (F(Z) + 1),  여기서 F(Z) = Σ e^zₖ
```

## 학습 과정

```plaintext
┌─────────────────────────────────────────────────────┐
│                    학습 루프                          │
│                                                     │
│  1. 분류기 학습 (지도 학습)                            │
│     - 레이블 있는 데이터로 봇/사람 분류                  │
│     - Cross Entropy Loss (LC) 계산                   │
│                                                     │
│  2. 판별자 학습                                       │
│     - 실제 데이터 vs 생성된 가짜 데이터 구별             │
│     - Binary Cross Entropy Loss (LD) 계산            │
│     - LD를 생성자에게 역전파 → 생성자 성능 개선          │
│                                                     │
│  3. 생성자 학습                                       │
│     - 판별자를 속이는 현실적 데이터 생성 시도             │
│     - 성공적으로 생성된 데이터 → 분류기 학습 데이터 보강  │
│                                                     │
│  ※ 분류기와 판별자는 공유 가중치 → 비지도 학습이          │
│    분류 성능을 간접적으로 향상                           │
└─────────────────────────────────────────────────────┘
```

## 특징 중요도 분석 (Feature Importance)

논문에서 Permutation Importance Algorithm (K=50)을 적용한 결과, SGAN 분류기에서 가장 영향력 있는 특징들:

| 순위 | 특징 | R²-Score (µ ± σ) | 설명 |
|------|------|-------------------|------|
| 1 | **post_browser_height** | 0.542 ± 0.008 | 브라우저 창 높이 |
| 2 | **post_browser_width** | 0.287 ± 0.010 | 브라우저 창 너비 |
| 3 | post_java_enabled (N) | 0.082 ± 0.003 | Java 미지원 여부 |
| 4 | post_java_enabled (Y) | 0.061 ± 0.002 | Java 지원 여부 |
| 5 | user_agent (Other) | 0.024 ± 0.002 | 비표준 User Agent |
| 6 | visit_page_num | 0.022 ± 0.003 | 방문 페이지 수 |
| 7 | visit_num | 0.012 ± 0.004 | 방문 횟수 |
| 8 | hourly_visitor | 0.010 ± 0.001 | 시간별 방문 여부 |
| 9 | page_type (product) | 0.005 ± 0.001 | 상품 페이지 여부 |
| 10 | last_purchase_num | 0.004 ± 0.001 | 최근 구매 횟수 |
| 11 | user_agent (Mozilla/5.0) | 0.003 ± 0.001 | 표준 User Agent |

### 주요 인사이트

- **브라우저 창 크기**(높이, 너비)가 가장 중요한 특징 → 봇은 최소한의 창 크기를 사용하는 경향
- 그러나 이 특징들은 봇이 **쉽게 위조 가능** → 행동 기반 탐지(3단계 DGCNN)의 필요성 부각
- Java 지원 여부는 오래된/위조된 User Agent를 식별하는 데 유용

## 평가 결과 (논문 Section 5.2)

| 모델 | Accuracy | Recall | Precision | F1-Score | AUROC |
|------|----------|--------|-----------|----------|-------|
| **SGAN** | 0.9895 | 0.9875 | 0.9189 | 0.9519 | 0.9886 |
| DGCNN | 0.9845 | 0.9833 | 0.9791 | 0.9812 | 0.9892 |
| Botcha-MAM | 0.9364 | 0.8383 | 1.0 | 0.9120 | 0.9437 |
| Botcha-RAM | 0.9952 | 0.9663 | 0.9807 | 0.9735 | 0.9996 |

- SGAN은 **개별 요청 단위**에서 높은 정확도(98.95%)와 재현율(98.75%) 달성
- Precision이 상대적으로 낮은 이유: 기술적 특징만으로는 정교한 봇을 완벽히 구분하기 어려움
- DGCNN(행동 기반)은 F1-Score에서 더 우수 → 기술 + 행동 특징의 조합이 핵심

## 휴리스틱 탐지 (1단계와의 연계)

SGAN 학습 전, 휴리스틱으로 명확한 봇을 먼저 필터링합니다. 이 레이블은 SGAN 학습 데이터에도 활용됩니다:

| 휴리스틱 | 설명 | 예시 |
|----------|------|------|
| **위조된 User Agent** | 자동화 라이브러리 UA 또는 브라우저 기능과 불일치 | `python-request`, 기능 불일치 UA |
| **요청 간격 유사성** | 일정한 시간 간격의 반복 요청 | 정확히 1시간마다 방문 |
| **비현실적 창 크기** | 한 축이 50px 미만인 브라우저 창 | 1x1, 50x30 등 |

### 데이터 레이블링 (Ground Truth)

| 클래스 | 가정(Assumption) 기준 Hits | 휴리스틱 보강 후 Hits |
|--------|---------------------------|---------------------|
| Bot | 51,462 | 65,018 |
| Human | 7,630 | 7,630 |
| Unknown | 723,579 | 710,023 |

- **Human 가정**: 웹사이트 운영 조직의 직원 계정에서 발생한 트래픽
- **Bot 가정**: 클라우드 제공자의 데이터센터 IP에서 발생한 트래픽

## 특징 추출 구현 예시

```javascript
class TechnicalFeatureExtractor {
  // 논문의 SGAN에서 사용하는 기술적 특징 수집
  extractFeatures() {
    return {
      // 브라우저 창 크기 (가장 중요한 특징, R²=0.542, 0.287)
      browserHeight: window.innerHeight,
      browserWidth: window.innerWidth,

      // User Agent (R²=0.024)
      userAgent: navigator.userAgent,

      // Java 지원 여부 (R²=0.082)
      // 현대 브라우저에서는 대부분 미지원 → 오래된/위조된 UA 식별에 유용
      javaEnabled: navigator.javaEnabled ? navigator.javaEnabled() : false,

      // 방문 관련 (R²=0.022, 0.012, 0.010)
      // 서버 측에서 세션/쿠키 기반으로 수집
      // visitPageNum, visitNum, hourlyVisitor 등

      // 자동화 탐지 플래그
      webdriver: navigator.webdriver,
      automationFlags: this.detectAutomationFlags(),
    };
  }

  detectAutomationFlags() {
    const flags = [];

    // Selenium/WebDriver 탐지
    if (navigator.webdriver) flags.push('webdriver');

    // Phantom.js 탐지
    if (window.callPhantom || window._phantom) flags.push('phantom');

    // Nightmare.js 탐지
    if (window.__nightmare) flags.push('nightmare');

    // Headless Chrome 탐지
    if (navigator.userAgent.includes('HeadlessChrome')) flags.push('headless');

    // Playwright 탐지
    if (navigator.userAgent.includes('Playwright')) flags.push('playwright');

    // WebDriver 속성 탐지
    if (document.documentElement.getAttribute('webdriver')) flags.push('webdriver_attr');

    // Chrome DevTools Protocol 탐지
    if (window.cdc_adoQpoasnfa76pfcZLmcfl_Array) flags.push('cdp');

    return flags;
  }
}
```

## SGAN 모델 구현 예시

```python
import tensorflow as tf
from tensorflow import keras

class SGANDiscriminator(keras.Model):
    """
    BOTracle SGAN 판별자 (논문 Section 4.1 기반)
    - 7개의 공유 히든 레이어
    - 2개의 출력 헤드: 판별(Discriminator) + 분류(Classifier)
    """
    def __init__(self, num_classes=2):  # 0: 사람, 1: 봇
        super().__init__()

        # 공유 히든 레이어 (논문: Dense(100,Sigmoid) → LeakyReLU(0.2) × 3 + Dropout)
        self.shared_layers = keras.Sequential([
            keras.layers.Dense(100, activation='sigmoid'),   # Layer 1
            keras.layers.LeakyReLU(alpha=0.2),               # Layer 2
            keras.layers.Dense(100, activation='sigmoid'),   # Layer 3
            keras.layers.LeakyReLU(alpha=0.2),               # Layer 4
            keras.layers.Dense(100, activation='sigmoid'),   # Layer 5
            keras.layers.LeakyReLU(alpha=0.2),               # Layer 6
            keras.layers.Dropout(0.4),                       # Layer 7
        ])

        # 판별 헤드: ExpSum 활성화 → p_real ∈ [0, 1]
        self.disc_head = keras.layers.Dense(num_classes)  # ExpSum 적용 전 원시 출력

        # 분류 헤드: Softmax → 클래스 확률
        self.class_head = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        h = self.shared_layers(x)

        # 판별 헤드 - ExpSum 활성화: E(Z) = F(Z) / (F(Z) + 1)
        disc_logits = self.disc_head(h)
        f_z = tf.reduce_sum(tf.exp(disc_logits), axis=-1, keepdims=True)
        validity = f_z / (f_z + 1)

        # 분류 헤드 - Softmax
        class_pred = self.class_head(h)

        return validity, class_pred


class SGANGenerator(keras.Model):
    """
    BOTracle SGAN 생성자 (논문 Section 4.1 기반)
    - 100차원 latent vector → 200 units → feature_dim units
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.dense1 = keras.layers.Dense(200, activation='sigmoid')
        self.output_layer = keras.layers.Dense(feature_dim, activation='relu')

    def call(self, z):
        h = self.dense1(z)
        return self.output_layer(h)


# 학습 설정
LATENT_DIM = 100
LEARNING_RATE = 0.0002
BETA_1 = 0.5

gen_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
disc_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

bce_loss = keras.losses.BinaryCrossentropy()
scce_loss = keras.losses.SparseCategoricalCrossentropy()


@tf.function
def train_step(generator, discriminator, real_labeled, labels, real_unlabeled, batch_size):
    """
    SGAN 학습 스텝 (논문 개념 기반)
    - real_labeled: 레이블이 있는 실제 데이터
    - labels: 해당 레이블 (0=사람, 1=봇)
    - real_unlabeled: 레이블이 없는 실제 데이터
    """
    noise = tf.random.normal([batch_size, LATENT_DIM])

    # 1. 판별자 학습
    with tf.GradientTape() as disc_tape:
        fake_data = generator(noise)

        # 실제 데이터 → 진짜로 판별되어야 함
        real_validity, real_class = discriminator(real_labeled)
        # 가짜 데이터 → 가짜로 판별되어야 함
        fake_validity, _ = discriminator(fake_data)

        # 판별 손실 (LD)
        d_loss_real = bce_loss(tf.ones_like(real_validity), real_validity)
        d_loss_fake = bce_loss(tf.zeros_like(fake_validity), fake_validity)
        d_loss = d_loss_real + d_loss_fake

        # 분류 손실 (LC) - 레이블 있는 데이터만
        c_loss = scce_loss(labels, real_class)

        total_disc_loss = d_loss + c_loss

    # 2. 생성자 학습
    with tf.GradientTape() as gen_tape:
        fake_data = generator(noise)
        fake_validity, _ = discriminator(fake_data)
        # 생성자는 판별자를 속이려 함 → 가짜를 진짜로 분류하게
        g_loss = bce_loss(tf.ones_like(fake_validity), fake_validity)

    return total_disc_loss, g_loss
```

## SGAN의 한계와 보완

논문에서 지적하는 기술적 특징 기반 SGAN의 핵심 한계:

1. **위조 취약성**: 가장 중요한 특징인 브라우저 창 크기(R²=0.542)를 봇이 일반적인 값으로 조작하면 탐지 회피 가능
2. **정적 특징 의존**: 기술적 특징은 자동화 기술에 독립적이지 않음 → 실제 브라우저를 사용하는 봇에 취약
3. **Precision 한계**: SGAN의 Precision(0.9189)이 DGCNN(0.9791)보다 낮음

이를 보완하기 위해 3단계에서 **행동 기반 분석(DGCNN + WT 그래프)**을 수행합니다:
- 행동적 특징은 자동화 기술과 **독립적**
- 봇이 사람의 행동을 모방하면 봇의 고유한 장점(속도, 지속성)을 상실
- 각 웹사이트의 고유한 사용자 행동 패턴을 봇이 사전에 파악하기 어려움

## 참고 문헌

- Kadel, J., See, A., Sinha, R., & Fischer, M. (2024). *BOTracle: A framework for Discriminating Bots and Humans*. arXiv:2412.02266v1
- Salimans, T. et al. (2016). *Improved techniques for training GANs*. NeurIPS.
- Altmann, A. et al. (2010). *Permutation importance: a corrected feature importance measure*. Bioinformatics.
