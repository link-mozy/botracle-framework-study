"""
BOTracle 2단계 - SGAN 모델 (Semi-Supervised Generative Adversarial Network)

이 파일은 BOTracle 논문 Section 4.1의 SGAN 아키텍처를 구현합니다.
머신러닝 초급자를 위해 모든 개념에 상세한 주석을 달았습니다.

═══════════════════════════════════════════════════════════════
  SGAN이 뭔가요? (비유로 설명)
═══════════════════════════════════════════════════════════════

  위조지폐를 예로 들어보겠습니다:

  - 생성자(Generator) = 위조범
    → 진짜처럼 보이는 가짜 지폐를 만듭니다.

  - 판별자(Discriminator) = 은행 감별사
    → 지폐가 진짜인지 가짜인지 구별합니다.
    → 추가로, 진짜 지폐가 어느 나라 것인지도 분류합니다.

  이 둘이 계속 경쟁하면서 서로의 능력이 향상됩니다:
    - 위조범은 더 정교한 가짜를 만들게 되고
    - 감별사는 더 미세한 차이도 구별하게 됩니다

  BOTracle에서는:
    - 생성자: 봇처럼 보이는 가짜 특징 데이터를 생성
    - 판별자: 데이터가 진짜인지 가짜인지 판별 + 봇인지 사람인지 분류


═══════════════════════════════════════════════════════════════
  왜 "Semi-Supervised" (반지도 학습)인가요?
═══════════════════════════════════════════════════════════════

  웹 트래픽 데이터의 대부분은 "이게 봇인지 사람인지" 레이블이 없습니다.
  전체 78만 건 중 레이블이 있는 건 약 7만 건뿐입니다.

  - 지도 학습 (Supervised):    레이블 있는 데이터만 사용 → 데이터 부족
  - 비지도 학습 (Unsupervised): 레이블 없는 데이터만 사용 → 정확도 낮음
  - 반지도 학습 (Semi-Supervised): 둘 다 사용 → SGAN의 방식!

  SGAN은 레이블이 없는 데이터에서도 "진짜 vs 가짜" 패턴을 학습하여
  분류 성능을 간접적으로 향상시킵니다.


═══════════════════════════════════════════════════════════════
  판별자 구조 (논문 Section 4.1)
═══════════════════════════════════════════════════════════════

    입력 (특징 벡터, 11차원)
         │
         ▼
    ┌─ 공유 히든 레이어 ──────────────────────┐
    │  Dense(100) + Sigmoid               │
    │  LeakyReLU(α=0.2)                   │
    │  Dense(100) + Sigmoid               │
    │  LeakyReLU(α=0.2)                   │
    │  Dense(100) + Sigmoid               │
    │  LeakyReLU(α=0.2)                   │
    │  Dropout(p=0.4)                     │
    └──────────┬─────────────┬────────────┘
               │             │
               ▼             ▼
         ┌──────────┐  ┌──────────┐
         │ 판별 헤드  │  │ 분류 헤드  │
         │ (D head) │  │ (C head) │
         │ ExpSum   │  │ Softmax  │
         └────┬─────┘  └────┬─────┘
              │             │
              ▼             ▼
         p_real ∈ [0,1]  [p_human, p_bot]
         (진짜 확률)     (클래스 확률)
"""

import tensorflow as tf
from tensorflow import keras


class SGANDiscriminator(keras.Model):
    """
    SGAN 판별자 (Discriminator)

    하나의 신경망이 두 가지 역할을 동시에 수행합니다:
      1. 판별 (Discrimination): 입력 데이터가 "진짜"인지 "가짜(생성자가 만든)"인지
      2. 분류 (Classification):  입력 데이터가 "봇"인지 "사람"인지

    두 역할이 같은 히든 레이어를 공유하기 때문에,
    판별 학습에서 얻은 지식이 분류 성능 향상에도 도움됩니다.
    이것이 Semi-Supervised 학습의 핵심입니다.

    논문 아키텍처 (Section 4.1):
      - 7개의 공유 히든 레이어: Dense(100,Sigmoid) → LeakyReLU(0.2) ×3 + Dropout(0.4)
      - 판별 헤드: ExpSum 활성화 함수
      - 분류 헤드: Softmax 활성화 함수
      - 최적화: Adam (learning_rate=0.0002, beta_1=0.5)
    """

    def __init__(self, num_classes=2):
        """
        Args:
            num_classes: 분류할 클래스 수 (기본값 2: 사람=0, 봇=1)
        """
        super().__init__()

        # ══════════════════════════════════════════════
        # 공유 히든 레이어 (Shared Hidden Layers)
        # ══════════════════════════════════════════════
        # 판별과 분류가 이 레이어들을 "공유"합니다.
        # → 판별 학습의 결과가 분류에도 반영되는 구조
        #
        # 레이어 구성 설명:
        #
        # Dense(100, sigmoid):
        #   - "Dense"는 완전연결층(Fully Connected Layer)
        #   - 100개의 뉴런(노드)으로 구성
        #   - Sigmoid 활성화: 출력값을 0~1 사이로 변환
        #     → sigmoid(x) = 1 / (1 + e^(-x))
        #
        # LeakyReLU(α=0.2):
        #   - 활성화 함수의 일종
        #   - 양수: 그대로 통과 (x → x)
        #   - 음수: 조금만 통과 (x → 0.2 * x)
        #   - 일반 ReLU는 음수를 완전히 차단하지만,
        #     LeakyReLU는 음수에도 작은 기울기를 허용하여
        #     "죽은 뉴런" 문제를 방지합니다.
        #
        # Dropout(0.4):
        #   - 학습 중 무작위로 40%의 뉴런을 꺼버립니다
        #   - 과적합(overfitting) 방지: 특정 뉴런에 과도하게 의존하지 않게 함
        #   - 예측 시(inference)에는 모든 뉴런이 활성화됩니다

        self.shared_layers = keras.Sequential([
            # 레이어 1: 입력 → 100차원 (Sigmoid 활성화)
            keras.layers.Dense(100, activation="sigmoid",
                               name="shared_dense_1"),
            # 레이어 2: LeakyReLU
            keras.layers.LeakyReLU(negative_slope=0.2,
                                   name="shared_leaky_relu_1"),

            # 레이어 3: 100차원 → 100차원 (Sigmoid 활성화)
            keras.layers.Dense(100, activation="sigmoid",
                               name="shared_dense_2"),
            # 레이어 4: LeakyReLU
            keras.layers.LeakyReLU(negative_slope=0.2,
                                   name="shared_leaky_relu_2"),

            # 레이어 5: 100차원 → 100차원 (Sigmoid 활성화)
            keras.layers.Dense(100, activation="sigmoid",
                               name="shared_dense_3"),
            # 레이어 6: LeakyReLU
            keras.layers.LeakyReLU(negative_slope=0.2,
                                   name="shared_leaky_relu_3"),

            # 레이어 7: 드롭아웃 (학습 시 40% 뉴런 비활성화)
            keras.layers.Dropout(0.4,
                                 name="shared_dropout"),
        ], name="shared_hidden_layers")

        # ══════════════════════════════════════════════
        # 판별 헤드 (Discriminator Head)
        # ══════════════════════════════════════════════
        # "이 데이터가 진짜(실제 웹 트래픽)인가, 가짜(생성자가 만든)인가?"
        #
        # 출력: p_real ∈ [0, 1]
        #   - 1에 가까우면: "진짜 데이터일 가능성이 높다"
        #   - 0에 가까우면: "가짜 데이터일 가능성이 높다"
        #
        # ExpSum 활성화 함수를 사용합니다 (Salimans et al., 2016):
        #   E(Z) = F(Z) / (F(Z) + 1),  여기서 F(Z) = Σ e^(zk)
        #
        # 이 함수는 일반적인 sigmoid와 비슷하지만,
        # 여러 출력 뉴런의 값을 종합적으로 고려합니다.
        self.disc_head = keras.layers.Dense(
            num_classes,  # 클래스 수만큼 뉴런 (ExpSum에 필요)
            name="disc_head"
        )

        # ══════════════════════════════════════════════
        # 분류 헤드 (Classifier Head)
        # ══════════════════════════════════════════════
        # "이 데이터가 봇인가, 사람인가?"
        #
        # Softmax 활성화:
        #   - 출력값들의 합이 1이 되도록 변환 (확률 분포)
        #   - 예: [2.0, 1.0] → [0.73, 0.27]
        #         → "73% 확률로 클래스 0(사람), 27% 확률로 클래스 1(봇)"
        self.class_head = keras.layers.Dense(
            num_classes,
            activation="softmax",
            name="class_head"
        )

    def call(self, x, training=False):
        """
        순전파 (Forward Pass): 입력 데이터를 넣으면 결과가 나옵니다.

        Args:
            x: 입력 특징 벡터 (shape: [batch_size, feature_dim])
               batch_size = 한 번에 처리하는 데이터 수
               feature_dim = 특징 차원 수 (11)
            training: 학습 중이면 True (Dropout이 활성화됨)

        Returns:
            validity: 진짜/가짜 확률 (shape: [batch_size, 1])
            class_pred: 클래스별 확률 (shape: [batch_size, num_classes])
        """
        # 공유 히든 레이어를 통과
        h = self.shared_layers(x, training=training)

        # ── 판별 헤드: ExpSum 활성화 ──
        # 논문 수식: E(Z) = F(Z) / (F(Z) + 1),  F(Z) = Σ e^(zk)
        #
        # 단계별:
        #   1. disc_head에서 원시 출력값(logits) 계산
        #   2. 각 값에 e^x 적용 (tf.exp)
        #   3. 모두 합산 → F(Z)
        #   4. F(Z) / (F(Z) + 1) → 0~1 사이 값
        disc_logits = self.disc_head(h)
        f_z = tf.reduce_sum(tf.exp(disc_logits), axis=-1, keepdims=True)
        validity = f_z / (f_z + 1.0)

        # ── 분류 헤드: Softmax ──
        class_pred = self.class_head(h)

        return validity, class_pred


class SGANGenerator(keras.Model):
    """
    SGAN 생성자 (Generator)

    "위조범" 역할: 무작위 노이즈로부터 실제 데이터처럼 보이는
    가짜 특징 벡터를 생성합니다.

    왜 필요한가?
      생성자가 만든 가짜 데이터로 판별자를 학습시키면,
      판별자는 "진짜 데이터의 특성"을 더 잘 이해하게 됩니다.
      이 이해가 공유 레이어를 통해 분류 성능도 높여줍니다.

    논문 아키텍처 (Section 4.1):
      - 입력: 100차원 잠재 벡터 (random noise)
      - 히든: Dense(200, Sigmoid)
      - 출력: Dense(feature_dim, ReLU)
    """

    def __init__(self, feature_dim=11):
        """
        Args:
            feature_dim: 생성할 특징 벡터의 차원 수
                         전처리기의 출력 차원과 동일해야 합니다 (기본: 11)
        """
        super().__init__()

        # ══════════════════════════════════════════════
        # 생성자 레이어
        # ══════════════════════════════════════════════
        # 100차원 노이즈 → 200차원 (Sigmoid) → 11차원 (ReLU)
        #
        # 비유: 100개의 주사위를 던져서 (노이즈)
        #       200개의 중간 특성을 계산하고
        #       최종적으로 11개의 "가짜 브라우저 특징"을 만들어냅니다.

        # 히든 레이어: 200개 뉴런, Sigmoid 활성화
        self.hidden = keras.layers.Dense(
            200,
            activation="sigmoid",
            name="gen_hidden"
        )

        # 출력 레이어: feature_dim개 뉴런, ReLU 활성화
        # ReLU: 음수 → 0, 양수 → 그대로
        # (특징 벡터의 값은 0 이상이므로 ReLU가 적합)
        self.output_layer = keras.layers.Dense(
            feature_dim,
            activation="relu",
            name="gen_output"
        )

    def call(self, z, training=False):
        """
        순전파: 노이즈를 입력받아 가짜 특징 벡터를 생성합니다.

        Args:
            z: 무작위 노이즈 벡터 (shape: [batch_size, latent_dim])
               latent_dim = 잠재 공간 차원 (100)

        Returns:
            가짜 특징 벡터 (shape: [batch_size, feature_dim])
        """
        h = self.hidden(z)
        return self.output_layer(h)


# ══════════════════════════════════════════════════════════════
# SGAN 전체를 하나로 묶는 클래스
# ══════════════════════════════════════════════════════════════

class SGAN:
    """
    SGAN 모델 전체를 관리하는 래퍼 클래스.

    생성자, 판별자, 옵티마이저, 손실 함수를 모두 관리합니다.

    사용법:
        sgan = SGAN(feature_dim=11)
        sgan.train(labeled_data, labels, unlabeled_data, epochs=100)
        prediction, confidence = sgan.predict(feature_vector)
    """

    # 논문에서 사용한 하이퍼파라미터 (Section 4.1)
    LATENT_DIM = 100          # 생성자 입력 노이즈 차원
    LEARNING_RATE = 0.0002    # 학습률 (Adam 옵티마이저)
    BETA_1 = 0.5              # Adam의 β₁ 파라미터

    def __init__(self, feature_dim=11, num_classes=2):
        """
        Args:
            feature_dim: 특징 벡터 차원 수 (전처리기 출력과 동일)
            num_classes: 분류 클래스 수 (사람=0, 봇=1)
        """
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # ── 모델 생성 ──
        self.generator = SGANGenerator(feature_dim=feature_dim)
        self.discriminator = SGANDiscriminator(num_classes=num_classes)

        # ══════════════════════════════════════════════
        # 옵티마이저 (Optimizer)
        # ══════════════════════════════════════════════
        # "어떻게 가중치를 업데이트할 것인가?"를 결정하는 알고리즘
        #
        # Adam: 현재 가장 널리 쓰이는 옵티마이저
        #   - learning_rate: 한 번에 가중치를 얼마나 조정할지 (작을수록 신중)
        #   - beta_1: 과거 기울기를 얼마나 기억할지 (모멘텀)
        #
        # 생성자와 판별자는 서로 경쟁하므로 별도의 옵티마이저를 사용합니다.
        self.gen_optimizer = keras.optimizers.Adam(
            learning_rate=self.LEARNING_RATE, beta_1=self.BETA_1
        )
        self.disc_optimizer = keras.optimizers.Adam(
            learning_rate=self.LEARNING_RATE, beta_1=self.BETA_1
        )

        # ══════════════════════════════════════════════
        # 손실 함수 (Loss Function)
        # ══════════════════════════════════════════════
        # "예측이 정답에서 얼마나 벗어났는가?"를 수치로 측정합니다.
        # 이 값이 작아지도록 가중치를 조정하는 것이 "학습"입니다.
        #
        # Binary Cross-Entropy (이진 교차 엔트로피):
        #   진짜/가짜 판별에 사용 (두 가지 경우 중 하나)
        #   - 정답이 1(진짜)인데 0.9로 예측 → 손실 작음 (좋음)
        #   - 정답이 1(진짜)인데 0.1로 예측 → 손실 큼 (나쁨)
        #
        # Sparse Categorical Cross-Entropy (희소 범주형 교차 엔트로피):
        #   봇/사람 분류에 사용 (여러 클래스 중 하나)
        #   - "Sparse"는 레이블이 정수 형태라는 뜻 (0=사람, 1=봇)
        self.bce_loss = keras.losses.BinaryCrossentropy()
        self.scce_loss = keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def _train_step(self, real_labeled, labels, real_unlabeled, batch_size):
        """
        한 번의 학습 스텝을 수행합니다.

        ═══════════════════════════════════════════════
        학습 과정 (매 스텝마다 반복):
        ═══════════════════════════════════════════════

        [1단계: 판별자 학습]
          목표: "진짜 데이터와 가짜 데이터를 잘 구별하게 만들기"
                + "봇과 사람을 잘 분류하게 만들기"

          입력:
            - 실제 데이터 (레이블 있음) → "진짜"라고 학습
            - 생성자가 만든 가짜 데이터   → "가짜"라고 학습

          손실 = 판별 손실(LD) + 분류 손실(LC)

        [2단계: 생성자 학습]
          목표: "판별자를 속이는 가짜 데이터 만들기"

          방법: 가짜 데이터를 만들어 판별자에 넣고,
                판별자가 "진짜"라고 착각하게 만드는 방향으로 학습

        Args:
            real_labeled:   레이블이 있는 실제 데이터 (shape: [batch, 11])
            labels:         해당 레이블 (shape: [batch], 값: 0 또는 1)
            real_unlabeled: 레이블이 없는 실제 데이터 (shape: [batch, 11])
            batch_size:     배치 크기

        Returns:
            disc_loss: 판별자 총 손실값
            gen_loss:  생성자 손실값
            class_loss: 분류 손실값
        """
        # 무작위 노이즈 생성 (생성자의 입력)
        # 표준 정규분포에서 무작위 값을 뽑습니다
        noise = tf.random.normal([batch_size, self.LATENT_DIM])

        # ────────────────────────────────────
        # 1단계: 판별자 학습
        # ────────────────────────────────────
        # tf.GradientTape()는 "이 블록 안에서 일어나는 연산을 기록해라"는 뜻
        # 기록된 연산을 바탕으로 기울기(gradient)를 계산하여 가중치를 업데이트합니다
        with tf.GradientTape() as disc_tape:
            # 생성자가 가짜 데이터 생성
            fake_data = self.generator(noise, training=True)

            # 판별자에 실제 데이터 입력 → 진짜로 판단되어야 함
            real_validity, real_class = self.discriminator(
                real_labeled, training=True
            )
            # 판별자에 가짜 데이터 입력 → 가짜로 판단되어야 함
            fake_validity, _ = self.discriminator(fake_data, training=True)

            # 비레이블 데이터도 판별 학습에 사용 (진짜로 판단되어야 함)
            unlabeled_validity, _ = self.discriminator(
                real_unlabeled, training=True
            )

            # ── 판별 손실 (LD) 계산 ──
            # "진짜 데이터는 1에 가깝게, 가짜 데이터는 0에 가깝게"
            #
            # tf.ones_like(x)  → x와 같은 모양의 1로 채워진 텐서 (정답=진짜)
            # tf.zeros_like(x) → x와 같은 모양의 0으로 채워진 텐서 (정답=가짜)
            d_loss_real = self.bce_loss(
                tf.ones_like(real_validity), real_validity
            )
            d_loss_unlabeled = self.bce_loss(
                tf.ones_like(unlabeled_validity), unlabeled_validity
            )
            d_loss_fake = self.bce_loss(
                tf.zeros_like(fake_validity), fake_validity
            )
            d_loss = d_loss_real + d_loss_unlabeled + d_loss_fake

            # ── 분류 손실 (LC) 계산 ──
            # 레이블이 있는 데이터로만 "봇/사람" 분류 학습
            c_loss = self.scce_loss(labels, real_class)

            # ── 총 판별자 손실 ──
            total_disc_loss = d_loss + c_loss

        # 기울기 계산 후 판별자 가중치 업데이트
        # (기울기 = "손실을 줄이려면 가중치를 어느 방향으로 조정해야 하는가?")
        disc_gradients = disc_tape.gradient(
            total_disc_loss, self.discriminator.trainable_variables
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        # ────────────────────────────────────
        # 2단계: 생성자 학습
        # ────────────────────────────────────
        # 목표: 판별자를 속이기 (가짜 데이터를 "진짜"로 인식하게)
        noise = tf.random.normal([batch_size, self.LATENT_DIM])

        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(noise, training=True)
            fake_validity, _ = self.discriminator(fake_data, training=True)

            # 생성자의 손실: 가짜 데이터가 "진짜(=1)"로 판별되길 원함
            # → 판별자가 1에 가까운 값을 출력할수록 생성자의 손실이 줄어듦
            gen_loss = self.bce_loss(
                tf.ones_like(fake_validity), fake_validity
            )

        # 기울기 계산 후 생성자 가중치 업데이트
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return total_disc_loss, gen_loss, c_loss

    def train(self, labeled_features, labels, unlabeled_features,
              epochs=100, batch_size=32):
        """
        SGAN 모델을 학습시킵니다.

        Args:
            labeled_features:   레이블이 있는 데이터 (numpy array, shape: [N, 11])
            labels:             해당 레이블 (numpy array, shape: [N], 값: 0 또는 1)
            unlabeled_features: 레이블이 없는 데이터 (numpy array, shape: [M, 11])
            epochs:             전체 데이터를 몇 번 반복 학습할지
            batch_size:         한 번에 처리할 데이터 수

        Returns:
            학습 이력 (dict): epoch별 손실값 기록
        """
        import numpy as np

        history = {"disc_loss": [], "gen_loss": [], "class_loss": []}
        n_labeled = len(labeled_features)
        n_unlabeled = len(unlabeled_features)

        for epoch in range(epochs):
            # 매 에폭마다 데이터를 셔플 (무작위 순서로 섞기)
            labeled_idx = np.random.permutation(n_labeled)
            unlabeled_idx = np.random.permutation(n_unlabeled)

            epoch_disc_loss = 0.0
            epoch_gen_loss = 0.0
            epoch_class_loss = 0.0
            n_batches = max(n_labeled, n_unlabeled) // batch_size

            for i in range(n_batches):
                # 배치 추출 (데이터가 부족하면 처음부터 다시)
                l_start = (i * batch_size) % n_labeled
                l_idx = labeled_idx[l_start:l_start + batch_size]
                if len(l_idx) < batch_size:
                    l_idx = np.concatenate([
                        l_idx,
                        labeled_idx[:batch_size - len(l_idx)]
                    ])

                u_start = (i * batch_size) % n_unlabeled
                u_idx = unlabeled_idx[u_start:u_start + batch_size]
                if len(u_idx) < batch_size:
                    u_idx = np.concatenate([
                        u_idx,
                        unlabeled_idx[:batch_size - len(u_idx)]
                    ])

                batch_labeled = tf.constant(
                    labeled_features[l_idx], dtype=tf.float32
                )
                batch_labels = tf.constant(labels[l_idx], dtype=tf.int32)
                batch_unlabeled = tf.constant(
                    unlabeled_features[u_idx], dtype=tf.float32
                )

                d_loss, g_loss, c_loss = self._train_step(
                    batch_labeled, batch_labels, batch_unlabeled, batch_size
                )

                epoch_disc_loss += d_loss.numpy()
                epoch_gen_loss += g_loss.numpy()
                epoch_class_loss += c_loss.numpy()

            # 에폭 평균 손실 기록
            if n_batches > 0:
                history["disc_loss"].append(epoch_disc_loss / n_batches)
                history["gen_loss"].append(epoch_gen_loss / n_batches)
                history["class_loss"].append(epoch_class_loss / n_batches)

            # 10 에폭마다 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"D loss: {history['disc_loss'][-1]:.4f} | "
                    f"G loss: {history['gen_loss'][-1]:.4f} | "
                    f"C loss: {history['class_loss'][-1]:.4f}"
                )

        return history

    def predict(self, feature_vector):
        """
        학습된 모델로 봇/사람을 예측합니다.

        Args:
            feature_vector: 전처리된 특징 벡터 (numpy array, shape: [11] 또는 [N, 11])

        Returns:
            prediction: "bot" 또는 "human"
            confidence: 예측 신뢰도 (0.0 ~ 1.0)
        """
        # 1차원이면 2차원으로 변환 (모델은 배치 입력을 기대)
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector[tf.newaxis, :]

        x = tf.constant(feature_vector, dtype=tf.float32)

        # 판별자의 분류 헤드 결과 사용
        _, class_pred = self.discriminator(x, training=False)

        # class_pred = [p_human, p_bot]
        # argmax → 확률이 더 높은 클래스 선택
        predicted_class = tf.argmax(class_pred, axis=-1).numpy()[0]
        confidence = float(tf.reduce_max(class_pred).numpy())

        prediction = "bot" if predicted_class == 1 else "human"
        return prediction, confidence

    def save_weights(self, path):
        """모델 가중치를 파일로 저장합니다."""
        self.discriminator.save_weights(f"{path}/discriminator.weights.h5")
        self.generator.save_weights(f"{path}/generator.weights.h5")
        print(f"모델 가중치 저장 완료: {path}")

    def load_weights(self, path):
        """저장된 가중치를 불러옵니다."""
        import numpy as np

        # 더미 데이터로 모델 빌드 (가중치 로드 전 필요)
        dummy = tf.zeros([1, self.feature_dim])
        self.discriminator(dummy)
        self.generator(tf.zeros([1, self.LATENT_DIM]))

        self.discriminator.load_weights(f"{path}/discriminator.weights.h5")
        self.generator.load_weights(f"{path}/generator.weights.h5")
        print(f"모델 가중치 로드 완료: {path}")
