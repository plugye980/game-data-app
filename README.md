# 똥겜 추천기

좋아하는 게임을 고르면, 평가가 낮은 게임(Steam 긍정 비율 70% 미만) 중 취향에 맞을 만한 게임을 추천하는 Streamlit 앱.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

성능 평가만 따로 실행: `python evaluate.py` → `results/metrics.csv`, `results/coefficients.csv`

## 데이터

`https://raw.githubusercontent.com/plugye980/game-data1/refs/heads/main/data/` 에서 불러오며,
접속이 안 되면 `data/` 폴더의 같은 파일을 사용한다.

## 모델

| 변수 | 역할 | 정의 |
|---|---|---|
| `is_recommended` | 종속변수 | 리뷰 작성자의 추천 여부 (1/0) |
| `funny_rate` | 독립변수 | 게임 리뷰 중 funny 표를 1개 이상 받은 리뷰 비율 |
| `med_hours` | 독립변수 | 게임 리뷰 작성자 플레이 시간 중앙값 (log1p 변환) |

- 로지스틱 회귀 (class_weight=balanced), 학습/평가 80:20 층화 분할
- 게임 단위 특성은 5-fold out-of-fold로 계산 (리뷰 자기 자신 정보 누수 방지)
- 추천 점수 = w × 취향 유사도(태그·설명 TF-IDF + 협업 필터링) + (1 − w) × 예측 추천 확률

## 파일

- `recommender.py` — 데이터 로드, 변수 구성, 분류 모델, 추천 로직
- `evaluate.py` — 성능 평가 지표 산출
- `app.py` — Streamlit 앱
