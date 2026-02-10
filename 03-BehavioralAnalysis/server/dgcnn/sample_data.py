"""
BOTracle 3단계 - 샘플 데이터 생성기

실제 웹 트래픽 데이터가 없으므로, 학습용 샘플 WT Graph 데이터를 생성합니다.
논문의 행동 패턴 가설을 반영하여 사람과 봇의 탐색 차이를 시뮬레이션합니다.

═══════════════════════════════════════════════════════════════
  사람 vs 봇의 탐색 패턴 차이 (논문 Section 3.3)
═══════════════════════════════════════════════════════════════

  사람 (Human):
    - 홈에서 시작하여 관심 있는 페이지만 선택적 방문
    - 뒤로가기(back navigation) 빈번
    - 같은 상품을 여러 번 재방문 (비교 쇼핑)
    - 자연스러운 시간 간격 (2~30초)
    - 깊이 우선 탐색 경향 (카테고리 → 상품 → 상세)

  봇 Type A (단순 크롤러, 40%):
    - 모든 페이지를 체계적으로 순회 (BFS 패턴)
    - 매우 빠른 간격 (100~500ms)
    - 모든 페이지를 정확히 1회 방문
    - 페이지 유형 무관하게 균일 방문

  봇 Type B (가격 모니터링, 40%):
    - 특정 페이지(상품)만 반복 방문
    - 일정한 간격으로 같은 경로 반복
    - 높은 재방문율 (같은 페이지 5~10회)

  봇 Type C (정교한 봇, 20%):
    - 사람과 유사하지만 미세한 차이
    - 뒤로가기가 없거나 매우 적음
    - 시간 간격이 규칙적
"""

import numpy as np

from dgcnn.wt_graph import WTGraph

# 재현 가능한 결과를 위한 랜덤 시드
np.random.seed(42)


# ══════════════════════════════════════════════════════════════
# 가상 쇼핑몰 사이트 구조
# ══════════════════════════════════════════════════════════════
# 실제 쇼핑몰처럼 여러 페이지를 정의합니다.
# 각 페이지는 이름, 유형, 제목을 가집니다.

SITE_PAGES = {
    "home": {
        "type": "home",
        "title": "Welcome to TechShop - Best Electronics Store",
    },
    "category/electronics": {
        "type": "category",
        "title": "Electronics & Gadgets Collection",
    },
    "category/clothing": {
        "type": "category",
        "title": "Fashion & Clothing Store",
    },
    "category/books": {
        "type": "category",
        "title": "Books & Education Materials",
    },
    "product/laptop-pro": {
        "type": "product",
        "title": "Premium Laptop Pro 15-inch",
    },
    "product/phone-x": {
        "type": "product",
        "title": "Smartphone X Ultra Pro Max",
    },
    "product/headphones": {
        "type": "product",
        "title": "Wireless Noise Cancelling Headphones",
    },
    "product/tshirt-basic": {
        "type": "product",
        "title": "Basic Cotton T-Shirt Premium",
    },
    "product/jeans-slim": {
        "type": "product",
        "title": "Slim Fit Denim Jeans",
    },
    "product/python-book": {
        "type": "product",
        "title": "Python Programming Complete Guide",
    },
    "product/tablet-lite": {
        "type": "product",
        "title": "Tablet Lite 10-inch Display",
    },
    "product/camera-dslr": {
        "type": "product",
        "title": "Professional DSLR Camera Kit",
    },
    "search": {
        "type": "search",
        "title": "Search Results",
    },
    "cart": {
        "type": "cart",
        "title": "Shopping Cart",
    },
    "checkout": {
        "type": "checkout",
        "title": "Secure Checkout",
    },
    "account": {
        "type": "account",
        "title": "My Account Dashboard",
    },
}

ALL_PAGE_NAMES = list(SITE_PAGES.keys())

# 카테고리별 상품 매핑 (자연스러운 탐색 경로 생성용)
CATEGORY_PRODUCTS = {
    "category/electronics": [
        "product/laptop-pro",
        "product/phone-x",
        "product/headphones",
        "product/tablet-lite",
        "product/camera-dslr",
    ],
    "category/clothing": [
        "product/tshirt-basic",
        "product/jeans-slim",
    ],
    "category/books": [
        "product/python-book",
    ],
}


# ══════════════════════════════════════════════════════════════
# 사람 세션 생성
# ══════════════════════════════════════════════════════════════

def generate_human_session(n_hits_range=(3, 15)) -> WTGraph:
    """
    사람 사용자의 탐색 세션을 시뮬레이션합니다.

    사람의 탐색 특성:
      1. 홈 → 카테고리 → 상품 순서의 깊이 우선 탐색
      2. 뒤로가기로 카테고리에 돌아와서 다른 상품 탐색
      3. 관심 있는 카테고리 1~2개만 집중 방문
      4. 자연스럽게 다양한 시간 간격 (2~30초)
      5. 때때로 장바구니에 추가하고 결제 진행

    Returns:
        WTGraph: 사람 탐색 패턴의 WT Graph
    """
    graph = WTGraph()
    n_hits = np.random.randint(n_hits_range[0], n_hits_range[1] + 1)

    # 기본 시작 시간 (밀리초 단위)
    base_time = 1707580800000

    # ── 사람의 탐색 경로 생성 ──
    # 1. 항상 홈에서 시작
    current_page = "home"
    prev_page = None
    visit_history = [current_page]  # 방문 기록 (뒤로가기용)

    # 관심 카테고리 선택 (1~2개)
    categories = list(CATEGORY_PRODUCTS.keys())
    n_interests = np.random.randint(1, 3)
    interest_categories = list(
        np.random.choice(categories, n_interests, replace=False)
    )

    for hit_idx in range(n_hits):
        # 시간 간격: 2~30초 (자연스러운 읽기/탐색 시간)
        time_gap = np.random.uniform(2000, 30000)
        timestamp = base_time + int(hit_idx * time_gap)

        page_info = SITE_PAGES[current_page]

        graph.add_hit({
            "detailedPagename": current_page,
            "previousPagename": prev_page,
            "timestamp": timestamp,
            "pageType": page_info["type"],
            "pageTitle": page_info["title"],
        })

        # ── 다음 페이지 결정 (사람의 자연스러운 행동) ──
        prev_page = current_page

        if hit_idx == n_hits - 1:
            break  # 마지막 히트

        # 행동 확률 분포
        action = np.random.random()

        if current_page == "home":
            # 홈에서: 관심 카테고리로 이동 (80%) 또는 검색 (20%)
            if action < 0.8:
                current_page = np.random.choice(interest_categories)
            else:
                current_page = "search"

        elif current_page.startswith("category/"):
            # 카테고리에서: 상품 보기 (70%), 뒤로가기 (20%), 다른 카테고리 (10%)
            products = CATEGORY_PRODUCTS.get(current_page, [])
            if action < 0.7 and products:
                current_page = np.random.choice(products)
            elif action < 0.9 and len(visit_history) > 1:
                # 뒤로가기: 이전 페이지로
                current_page = visit_history[-2] if len(visit_history) >= 2 else "home"
            else:
                current_page = np.random.choice(interest_categories)

        elif current_page.startswith("product/"):
            # 상품에서: 뒤로가기 (40%), 장바구니 (15%),
            #          다른 상품 (25%), 홈 (20%)
            if action < 0.4 and len(visit_history) >= 2:
                current_page = visit_history[-2]
            elif action < 0.55:
                current_page = "cart"
            elif action < 0.8:
                # 같은 카테고리의 다른 상품
                cat = np.random.choice(interest_categories)
                products = CATEGORY_PRODUCTS.get(cat, [])
                if products:
                    current_page = np.random.choice(products)
                else:
                    current_page = "home"
            else:
                current_page = "home"

        elif current_page == "cart":
            # 장바구니에서: 결제 (40%), 쇼핑 계속 (60%)
            if action < 0.4:
                current_page = "checkout"
            else:
                current_page = np.random.choice(interest_categories)

        elif current_page == "search":
            # 검색에서: 상품 클릭 (70%), 홈 (30%)
            all_products = [p for prods in CATEGORY_PRODUCTS.values() for p in prods]
            if action < 0.7 and all_products:
                current_page = np.random.choice(all_products)
            else:
                current_page = "home"

        else:
            # 기타 페이지: 홈으로
            current_page = "home"

        visit_history.append(current_page)

    return graph


# ══════════════════════════════════════════════════════════════
# 봇 세션 생성 - Type A: 단순 크롤러
# ══════════════════════════════════════════════════════════════

def generate_bot_session_type_a(n_hits_range=(10, 30)) -> WTGraph:
    """
    Type A 봇: 단순 크롤러

    특성:
      - 모든 페이지를 체계적으로 순회 (BFS/DFS 패턴)
      - 매우 빠른 간격 (100~500ms)
      - 대부분의 페이지를 정확히 1회 방문
      - 페이지 유형 무관하게 균일 방문

    비유: 도서관의 모든 책장을 순서대로 스캔하는 로봇
    """
    graph = WTGraph()
    base_time = 1707580800000

    # 방문할 페이지를 순서대로 나열 (체계적 순회)
    pages_to_visit = list(ALL_PAGE_NAMES)
    np.random.shuffle(pages_to_visit)  # 약간의 랜덤성

    n_hits = min(
        np.random.randint(n_hits_range[0], n_hits_range[1] + 1),
        len(pages_to_visit),
    )

    prev_page = None
    for i in range(n_hits):
        current_page = pages_to_visit[i % len(pages_to_visit)]

        # 봇은 매우 빠른 간격 (100~500ms)
        time_gap = np.random.uniform(100, 500)
        timestamp = base_time + int(i * time_gap)

        page_info = SITE_PAGES[current_page]

        graph.add_hit({
            "detailedPagename": current_page,
            "previousPagename": prev_page,
            "timestamp": timestamp,
            "pageType": page_info["type"],
            "pageTitle": page_info["title"],
        })

        prev_page = current_page

    return graph


# ══════════════════════════════════════════════════════════════
# 봇 세션 생성 - Type B: 가격 모니터링 봇
# ══════════════════════════════════════════════════════════════

def generate_bot_session_type_b(n_hits_range=(8, 25)) -> WTGraph:
    """
    Type B 봇: 가격 모니터링 봇

    특성:
      - 특정 상품 페이지만 반복 방문 (가격 체크)
      - 일정한 간격으로 같은 경로 반복
      - 높은 재방문율 (같은 페이지 5~10회)
      - 좁은 페이지 유형 분포 (product가 대부분)

    비유: 같은 매장만 왔다갔다 하면서 가격표를 확인하는 사람
    """
    graph = WTGraph()
    base_time = 1707580800000

    n_hits = np.random.randint(n_hits_range[0], n_hits_range[1] + 1)

    # 모니터링 대상 상품 2~4개 선택
    all_products = [p for prods in CATEGORY_PRODUCTS.values() for p in prods]
    n_targets = np.random.randint(2, 5)
    target_products = list(
        np.random.choice(all_products, min(n_targets, len(all_products)), replace=False)
    )

    # 반복 경로: 홈 → 상품1 → 홈 → 상품2 → 홈 → ...
    prev_page = None
    for i in range(n_hits):
        # 짝수 히트: 홈, 홀수 히트: 상품
        if i % 2 == 0:
            current_page = "home"
        else:
            current_page = target_products[(i // 2) % len(target_products)]

        # 일정한 간격 (500~2000ms, 변동 적음)
        time_gap = np.random.uniform(500, 2000)
        timestamp = base_time + int(i * time_gap)

        page_info = SITE_PAGES[current_page]

        graph.add_hit({
            "detailedPagename": current_page,
            "previousPagename": prev_page,
            "timestamp": timestamp,
            "pageType": page_info["type"],
            "pageTitle": page_info["title"],
        })

        prev_page = current_page

    return graph


# ══════════════════════════════════════════════════════════════
# 봇 세션 생성 - Type C: 정교한 봇
# ══════════════════════════════════════════════════════════════

def generate_bot_session_type_c(n_hits_range=(5, 15)) -> WTGraph:
    """
    Type C 봇: 정교한 봇 (사람과 유사하게 위장)

    특성:
      - 사람과 유사한 경로이지만 미세한 차이
      - 뒤로가기가 전혀 없음 (항상 앞으로만 진행)
      - 시간 간격이 거의 일정 (사람은 들쭉날쭉)
      - 순서가 약간 체계적 (카테고리 → 상품 → 장바구니가 너무 깔끔)

    비유: 사람인 척 연기하는 로봇 (대체로 그럴듯하지만 미세한 부자연스러움)
    """
    graph = WTGraph()
    base_time = 1707580800000

    n_hits = np.random.randint(n_hits_range[0], n_hits_range[1] + 1)

    # 정교한 경로: 홈 → 카테고리 → 상품들 → 장바구니 → 결제
    # (뒤로가기 없이 일직선 진행)
    path = ["home"]

    # 카테고리 1개 선택
    cat = np.random.choice(list(CATEGORY_PRODUCTS.keys()))
    path.append(cat)

    # 해당 카테고리의 상품 2~3개 순서대로 방문
    products = CATEGORY_PRODUCTS[cat]
    n_products = min(np.random.randint(2, 4), len(products))
    for p in products[:n_products]:
        path.append(p)

    # 장바구니, 결제로 진행
    path.extend(["cart", "checkout"])

    prev_page = None
    for i in range(min(n_hits, len(path))):
        current_page = path[i]

        # 거의 일정한 간격 (3~5초, 변동 작음)
        time_gap = np.random.uniform(3000, 5000)
        timestamp = base_time + int(i * time_gap)

        page_info = SITE_PAGES[current_page]

        graph.add_hit({
            "detailedPagename": current_page,
            "previousPagename": prev_page,
            "timestamp": timestamp,
            "pageType": page_info["type"],
            "pageTitle": page_info["title"],
        })

        prev_page = current_page

    return graph


# ══════════════════════════════════════════════════════════════
# 데이터셋 생성
# ══════════════════════════════════════════════════════════════

def generate_dataset(n_human=300, n_bot=300):
    """
    학습용 전체 데이터셋을 생성합니다.

    Args:
        n_human: 생성할 사람 세션 수
        n_bot:   생성할 봇 세션 수

    Returns:
        dict:
            graphs:   WTGraph 객체 리스트
            labels:   라벨 배열 (0=사람, 1=봇)
    """
    graphs = []
    labels = []

    # ── 사람 세션 생성 ──
    print(f"  사람 세션 {n_human}개 생성 중...")
    for _ in range(n_human):
        graph = generate_human_session()
        graphs.append(graph)
        labels.append(0)  # 0 = 사람

    # ── 봇 세션 생성 ──
    n_bot_a = int(n_bot * 0.4)   # Type A: 40%
    n_bot_b = int(n_bot * 0.4)   # Type B: 40%
    n_bot_c = n_bot - n_bot_a - n_bot_b  # Type C: 20%

    print(f"  봇 세션 생성 중... (A: {n_bot_a}, B: {n_bot_b}, C: {n_bot_c})")

    for _ in range(n_bot_a):
        graphs.append(generate_bot_session_type_a())
        labels.append(1)  # 1 = 봇

    for _ in range(n_bot_b):
        graphs.append(generate_bot_session_type_b())
        labels.append(1)

    for _ in range(n_bot_c):
        graphs.append(generate_bot_session_type_c())
        labels.append(1)

    # 데이터 셔플 (순서 무작위화)
    labels = np.array(labels, dtype=np.int32)
    shuffle_idx = np.random.permutation(len(labels))

    graphs = [graphs[i] for i in shuffle_idx]
    labels = labels[shuffle_idx]

    # 통계 출력
    print(f"  데이터셋 생성 완료:")
    print(f"    전체: {len(labels)}개 (사람: {n_human}, 봇: {n_bot})")
    print(
        f"    평균 노드 수: {np.mean([len(g.nodes) for g in graphs]):.1f}"
    )
    print(
        f"    평균 엣지 수: {np.mean([len(g.edges) for g in graphs]):.1f}"
    )

    return {
        "graphs": graphs,
        "labels": labels,
    }
