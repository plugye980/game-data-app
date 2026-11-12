"""분류 모델 성능 평가 → results/ 에 저장."""
from pathlib import Path

import pandas as pd

from recommender import BadGameRecommender, Params, build_dataset, load_data, train_and_evaluate

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

data = load_data()
print("data source:", data["sources"])

ds = build_dataset(data["games"], data["recs"])
print(f"dataset rows: {len(ds):,}  games: {ds['app_id'].nunique():,}  positive rate: {ds['is_recommended'].mean():.3f}")
print(ds[["funny_rate", "med_hours"]].describe().round(3))

res = train_and_evaluate(ds)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(f"\ntrain {res['n_train']:,} / test {res['n_test']:,}")
print(res["metrics"].round(4))
print(res["coef"].round(4), "\nintercept", round(res["intercept"], 4))

res["metrics"].round(4).to_csv(OUT / "metrics.csv", encoding="utf-8-sig")
res["coef"].round(4).to_csv(OUT / "coefficients.csv", index=False, encoding="utf-8-sig")

# 추천 동작 확인
rec = BadGameRecommender(data["games"], data["recs"], res["model"])
print("candidates:", int(rec.candidate_mask(Params()).sum()))
g = data["games"]
seeds = g[g["title"].isin(["Terraria", "Stardew Valley", "Garry's Mod"])]["app_id"].tolist()
print("seeds:", seeds)
cols = ["title", "positive_ratio", "funny_rate", "med_hours", "pred_prob", "similarity", "score", "shared_tags"]
print(rec.recommend(seeds, Params(), k=10)[cols].round(3).to_string())
