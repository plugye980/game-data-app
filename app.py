import pandas as pd
import streamlit as st

from recommender import (
    BAD_THRESHOLD,
    DATA_URL,
    BadGameRecommender,
    Params,
    build_dataset,
    load_data,
    train_and_evaluate,
)

st.set_page_config(page_title="똥겜 추천기", page_icon="🎮", layout="wide")


@st.cache_resource(show_spinner="데이터를 불러오고 모델을 학습하는 중...")
def get_model():
    data = load_data()
    dataset = build_dataset(data["games"], data["recs"])
    result = train_and_evaluate(dataset)
    rec = BadGameRecommender(data["games"], data["recs"], result["model"])
    return data, dataset, result, rec


data, dataset, result, rec = get_model()
games = rec.games

# ------------------------------------------------------------------ 사이드바
with st.sidebar:
    st.header("설정")
    k = st.slider("추천 개수", 5, 30, 10)
    max_pr = st.slider(
        "똥겜 기준: 긍정 비율 최대(%)", 0, BAD_THRESHOLD - 1, BAD_THRESHOLD - 1,
        help="이 값 이하인 게임만 추천 후보가 됩니다.",
    )
    min_rev = st.slider("샘플 리뷰 최소 개수", 3, 30, 3)
    w_sim = st.slider(
        "취향 유사도 비중", 0.0, 1.0, 0.5, 0.05,
        help="나머지 비중은 분류 모델의 추천 확률(funny_rate, med_hours 기반)입니다.",
    )
    alpha = st.slider(
        "유사도 내 태그·설명 비중", 0.0, 1.0, 0.6, 0.05,
        help="나머지는 같은 유저에게 함께 추천받은 정도(협업 필터링)입니다.",
    )
    st.divider()
    src = set(data["sources"].values())
    st.caption(
        "데이터: GitHub raw URL" if src == {"github"}
        else "데이터: GitHub raw URL 접속 실패 → 로컬 data/ 사용"
    )
    st.caption(f"게임 {len(games):,} · 리뷰 {len(data['recs']):,} · 유저 {len(data['users']):,}")

params = Params(w_sim=w_sim, alpha=alpha, max_positive_ratio=max_pr, min_reviews=min_rev)

st.title("🎮 똥겜 추천기")
st.caption("좋아하는 게임을 고르면, 평가는 낮지만 취향에 맞을 만한 게임을 찾아 줍니다.")

tab_rec, tab_eval, tab_data = st.tabs(["똥겜 추천", "모델 성능", "데이터·변수"])

# ------------------------------------------------------------------ 추천
with tab_rec:
    order = games.sort_values(["n_reviews", "user_reviews"], ascending=False)
    labels = {
        r.app_id: f"{r.title} ({str(r.date_release)[:4]})" for r in order.itertuples()
    }
    picked = st.multiselect(
        "좋아하는 게임을 골라 주세요 (여러 개 가능)",
        options=list(labels),
        format_func=labels.get,
        placeholder="게임 이름 검색...",
    )

    if not picked:
        st.info("게임을 하나 이상 고르면 추천이 표시됩니다.")
    else:
        out = rec.recommend(picked, params, k=k)
        if out.empty:
            st.warning("조건에 맞는 똥겜 후보가 없습니다. 사이드바 기준을 완화해 보세요.")
        for i, r in out.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                c1.image(
                    "https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/"
                    f"{r.app_id}/header.jpg",
                    width="stretch",
                )
                c2.markdown(
                    f"**{i + 1}. [{r.title}](https://store.steampowered.com/app/{r.app_id})**"
                    f" &nbsp; `{r.rating}` · 긍정 {r.positive_ratio}% · 리뷰 {r.user_reviews:,}개"
                    f" · ${r.price_final:.2f}"
                )
                c2.markdown(
                    f"추천 확률 **{r.pred_prob:.1%}** · 취향 유사도 **{r.similarity:.2f}**"
                    f" · funny_rate **{r.funny_rate:.1%}** · med_hours **{r.med_hours:.1f}h**"
                    f" · 종합 점수 **{r.score:.3f}**"
                )
                if r.shared_tags:
                    c2.caption("겹치는 태그: " + " · ".join(r.shared_tags))

# ------------------------------------------------------------------ 성능
with tab_eval:
    st.subheader("분류 모델: is_recommended ~ funny_rate + med_hours")
    a, b, c = st.columns(3)
    a.metric("학습 데이터", f"{result['n_train']:,}")
    b.metric("평가 데이터", f"{result['n_test']:,}")
    c.metric("추천(1) 비율", f"{result['pos_rate']:.1%}")

    met = result["metrics"]
    st.markdown("**평가 지표 (테스트 20%)**")
    st.dataframe(
        met[["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]].astype(float).round(4),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**혼동 행렬 — Logistic**")
        r0 = met.iloc[0]
        st.dataframe(
            pd.DataFrame(
                [[int(r0.tn), int(r0.fp)], [int(r0.fn), int(r0.tp)]],
                index=["실제 0 (비추천)", "실제 1 (추천)"],
                columns=["예측 0", "예측 1"],
            ),
            width="stretch",
        )
    with right:
        st.markdown("**회귀 계수 (표준화 기준)**")
        coef = result["coef"].copy()
        coef.loc[len(coef)] = ["intercept", result["intercept"]]
        st.dataframe(coef.round(4), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 데이터
with tab_data:
    st.subheader("변수 구성")
    st.table(
        pd.DataFrame(
            {
                "변수": ["is_recommended", "funny_rate", "med_hours"],
                "역할": ["종속변수", "독립변수", "독립변수"],
                "정의": [
                    "리뷰 작성자가 게임을 추천했는지 (1/0)",
                    "해당 게임 리뷰 중 funny 표를 1개 이상 받은 리뷰 비율",
                    "해당 게임 리뷰 작성자들의 플레이 시간 중앙값 (모델 입력 시 log1p 변환)",
                ],
            }
        )
    )
    st.markdown(
        f"- 학습 대상: 긍정 비율 {BAD_THRESHOLD}% 미만 게임(똥겜)의 리뷰\n"
        "- 리뷰 자신의 정보가 특성에 섞이지 않도록 5-fold out-of-fold 방식으로 게임 단위 특성 계산\n"
        "- 추천 점수 = 취향 유사도 × w + 추천 확률(후보 내 정규화) × (1 − w)\n"
        "- 원본 데이터에 태그가 없는 게임은 함께 추천된 게임들의 태그를 빌려 유사도 계산\n"
        f"- 데이터 출처: `{DATA_URL}`"
    )
    st.markdown("**학습 데이터 미리보기**")
    st.dataframe(dataset.head(200), width="stretch")
    st.markdown("**변수 요약 통계**")
    st.dataframe(dataset[["funny_rate", "med_hours", "is_recommended"]].describe().round(3), width="stretch")
