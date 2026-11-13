"""저퀄리티 게임 추천 모델.

1) 분류 모델 (로지스틱 회귀)
   - 종속변수: is_recommended (리뷰 단위 추천 여부)
   - 독립변수: funny_rate  = 게임 리뷰 중 funny 표를 1개 이상 받은 리뷰 비율
              med_hours   = 게임 리뷰 작성자들의 플레이 시간 중앙값
   - 리뷰 자신의 값이 특성에 섞이지 않도록 5-fold out-of-fold 방식으로 특성을 만든다.
   - 학습 대상: 저퀄리티 게임(positive_ratio < 70) 리뷰

2) 추천
   - 후보: 저퀄리티 게임 중 샘플 리뷰가 min_reviews개 이상인 게임
   - 최종 점수 = w * 취향 유사도(유저가 고른 게임과의 태그·설명·공동 추천 유사도)
              + (1 - w) * 분류 모델이 예측한 추천 확률
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, normalize

DATA_URL = "https://raw.githubusercontent.com/plugye980/game-data1/refs/heads/main/data/"
LOCAL_DIR = Path(__file__).parent / "data"
FILES = ["games.csv", "games_metadata.json", "recommendations.csv", "users.csv"]

BAD_THRESHOLD = 70          # positive_ratio < 70 → 저퀄리티 게임 (Steam 'Mixed' 이하)
FEATURES = ["funny_rate", "med_hours"]
TARGET = "is_recommended"
SEED = 42
N_FOLDS = 5


# ================================================================ 데이터 로드
def _fetch(name: str, timeout: float) -> tuple[bytes, str]:
    """GitHub raw URL에서 받고, 실패하면 로컬 data/ 폴더를 쓴다."""
    try:
        with urllib.request.urlopen(DATA_URL + name, timeout=timeout) as r:
            return r.read(), "github"
    except Exception:
        return (LOCAL_DIR / name).read_bytes(), "local"


def load_data(timeout: float = 10.0) -> dict:
    raw, sources = {}, {}
    online = True
    for name in FILES:
        if online:
            raw[name], sources[name] = _fetch(name, timeout)
            online = sources[name] == "github"  # 한 번 실패하면 나머지는 바로 로컬
        else:
            raw[name], sources[name] = (LOCAL_DIR / name).read_bytes(), "local"

    games = pd.read_csv(io.BytesIO(raw["games.csv"]))
    meta = pd.read_json(io.BytesIO(raw["games_metadata.json"]), lines=True)
    recs = pd.read_csv(io.BytesIO(raw["recommendations.csv"]))
    users = pd.read_csv(io.BytesIO(raw["users.csv"]))

    games = games.merge(meta, on="app_id", how="left")
    games["tags"] = games["tags"].apply(lambda t: t if isinstance(t, list) else [])
    games["description"] = games["description"].fillna("")
    games = games.reset_index(drop=True)
    recs["is_recommended"] = recs["is_recommended"].astype(bool)
    return {"games": games, "recs": recs, "users": users, "sources": sources}


# ================================================================ 변수 구성
def game_features(recs: pd.DataFrame) -> pd.DataFrame:
    """게임 단위 funny_rate, med_hours, 샘플 리뷰 수."""
    g = recs.assign(is_funny=recs["funny"] > 0).groupby("app_id")
    return pd.DataFrame(
        {
            "funny_rate": g["is_funny"].mean(),
            "med_hours": g["hours"].median(),
            "n_reviews": g.size(),
        }
    )


def oof_features(recs: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """각 리뷰의 특성을 '다른 fold' 리뷰로만 계산 (자기 자신 정보 누수 방지)."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(recs))
    out = pd.DataFrame(index=recs.index, columns=FEATURES, dtype=float)
    for k in range(n_folds):
        feats = game_features(recs[fold != k])
        rows = fold == k
        joined = recs.loc[rows, ["app_id"]].join(feats, on="app_id")
        out.loc[rows, FEATURES] = joined[FEATURES].to_numpy()
    return out


def build_dataset(games: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    """저퀄리티 게임 리뷰 + OOF 특성 + 종속변수."""
    bad_ids = set(games.loc[games["positive_ratio"] < BAD_THRESHOLD, "app_id"])
    df = pd.concat([recs, oof_features(recs)], axis=1)
    df = df[df["app_id"].isin(bad_ids)].dropna(subset=FEATURES)
    df[TARGET] = df[TARGET].astype(int)
    return df[["app_id", "user_id", *FEATURES, TARGET]].reset_index(drop=True)


# ================================================================ 분류 모델
def make_model() -> object:
    # med_hours는 분포가 심하게 치우쳐 있어 log1p 변환 후 표준화
    log_hours = FunctionTransformer(
        lambda x: np.column_stack([x[:, 0], np.log1p(x[:, 1])]), validate=True
    )
    return make_pipeline(
        log_hours,
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    )


def _metrics(y, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan"),
        "log_loss": log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_and_evaluate(dataset: pd.DataFrame, test_size: float = 0.2) -> dict:
    x = dataset[FEATURES].to_numpy(dtype=float)
    y = dataset[TARGET].to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=SEED
    )
    model = make_model().fit(x_tr, y_tr)

    # 비교용 기준선: 학습셋의 추천 비율을 그대로 확률로 사용
    base_prob = np.full(len(y_te), y_tr.mean())

    lr = model[-1]
    coef = pd.DataFrame(
        {"feature": ["funny_rate", "log1p(med_hours)"], "coef(표준화)": lr.coef_[0]}
    )
    metrics = pd.DataFrame(
        {
            "Logistic(funny_rate, med_hours)": _metrics(y_te, model.predict_proba(x_te)[:, 1]),
            "Baseline(상수 확률)": _metrics(y_te, base_prob),
        }
    ).T
    final = make_model().fit(x, y)  # 앱에서 쓸 최종 모델은 전체 데이터로 재학습
    return {
        "model": final,
        "metrics": metrics,
        "coef": coef,
        "intercept": float(lr.intercept_[0]),
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "pos_rate": float(y.mean()),
    }


# ================================================================ 추천
@dataclass
class Params:
    w_sim: float = 0.5        # 취향 유사도 비중 (나머지는 추천 확률)
    alpha: float = 0.6        # 취향 유사도 내부: 콘텐츠 vs 공동 추천
    max_positive_ratio: int = BAD_THRESHOLD - 1
    min_reviews: int = 3      # 샘플 리뷰 최소 개수 (특성 안정성)


class BadGameRecommender:
    def __init__(self, games: pd.DataFrame, recs: pd.DataFrame, clf) -> None:
        self.games = games.copy()
        self.idx_of = {a: i for i, a in enumerate(self.games["app_id"])}

        feats = game_features(recs)
        self.games = self.games.join(feats, on="app_id")
        self.games["n_reviews"] = self.games["n_reviews"].fillna(0).astype(int)
        has = self.games[FEATURES].notna().all(axis=1).to_numpy()
        prob = np.full(len(self.games), np.nan)
        prob[has] = clf.predict_proba(self.games.loc[has, FEATURES].to_numpy(float))[:, 1]
        self.games["pred_prob"] = prob

        self._build_content()
        self._build_cf(recs)

    def _build_content(self) -> None:
        tag_vec = TfidfVectorizer(analyzer=lambda tags: tags, min_df=2)
        tag_m = tag_vec.fit_transform(self.games["tags"])
        desc_vec = TfidfVectorizer(
            stop_words="english", max_features=20000, min_df=3, sublinear_tf=True
        )
        desc_m = desc_vec.fit_transform(self.games["description"])
        self.content = normalize(
            sparse.hstack([normalize(tag_m), 0.3 * normalize(desc_m)]).tocsr()
        )

    def _build_cf(self, recs: pd.DataFrame) -> None:
        pos = recs[recs["is_recommended"]]
        cols = pos["app_id"].map(self.idx_of)
        keep = cols.notna().to_numpy()
        cols = cols[keep].astype(int).to_numpy()
        rows, _ = pd.factorize(pos.loc[keep, "user_id"])
        x = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(rows.max() + 1, len(self.games))
        )
        x.data[:] = 1.0
        self.cf = normalize(x.tocsc(), axis=0)

    def _seed_content(self, i: int, cf_row: np.ndarray, n_neighbors: int = 20):
        """시드 게임의 콘텐츠 벡터와 태그.

        원본 데이터에 태그·설명이 비어 있는 게임(주로 인기작)은
        함께 추천된(CF) 이웃 게임들의 벡터·태그를 빌려 쓴다.
        """
        if self.content[i].nnz > 0:
            return self.content[i], set(self.games.at[i, "tags"])
        has_content = np.diff(self.content.indptr) > 0
        w = np.where(has_content, cf_row, 0.0)
        w[i] = 0.0
        nb = np.argsort(-w)[:n_neighbors]
        nb = nb[w[nb] > 0]
        if len(nb) == 0:
            return None, set()
        vec = sparse.csr_matrix(w[nb] @ self.content[nb])
        counts = pd.Series([t for j in nb for t in self.games.at[j, "tags"]]).value_counts()
        return vec, set(counts.index[:15])

    def candidate_mask(self, p: Params) -> np.ndarray:
        g = self.games
        return (
            (g["positive_ratio"] <= p.max_positive_ratio)
            & (g["n_reviews"] >= p.min_reviews)
            & g["pred_prob"].notna()
        ).to_numpy().copy()

    @staticmethod
    def _minmax(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sub = v[mask]
        if len(sub) == 0 or sub.max() - sub.min() < 1e-12:
            return np.zeros_like(v)
        return (v - sub.min()) / (sub.max() - sub.min())

    def recommend(self, seed_app_ids: list[int], p: Params, k: int = 10) -> pd.DataFrame:
        seed_idx = [self.idx_of[a] for a in seed_app_ids if a in self.idx_of]
        mask = self.candidate_mask(p)
        mask[seed_idx] = False
        if not seed_idx or not mask.any():
            return self.games.iloc[[]]

        cf_each = np.asarray((self.cf[:, seed_idx].T @ self.cf).todense())  # (시드, 게임)
        vecs, seed_tags = [], set()
        for row, i in zip(cf_each, seed_idx):
            vec, tags = self._seed_content(i, row)
            if vec is not None:
                vecs.append(vec)
            seed_tags |= tags
        if vecs:
            profile = normalize(np.asarray(sparse.vstack(vecs).mean(axis=0)))
            content = np.asarray(self.content @ profile.T).ravel()
        else:
            content = np.zeros(len(self.games))
        cf = cf_each.sum(axis=0)

        if cf[mask].max() <= 0:
            alpha = 1.0
        elif content[mask].max() <= 0:
            alpha = 0.0
        else:
            alpha = p.alpha
        sim = alpha * self._minmax(content, mask) + (1 - alpha) * self._minmax(cf, mask)
        sim = self._minmax(sim, mask)
        prob = self.games["pred_prob"].fillna(0).to_numpy()
        total = p.w_sim * sim + (1 - p.w_sim) * self._minmax(prob, mask)
        total[~mask] = -np.inf

        top = np.argsort(-total)[:k]
        top = top[np.isfinite(total[top])]
        out = self.games.iloc[top].copy()
        out["score"] = total[top]
        out["similarity"] = sim[top]
        out["shared_tags"] = [
            [t for t in self.games.at[i, "tags"] if t in seed_tags][:6] for i in top
        ]
        return out.reset_index(drop=True)
