import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.common import vehicle_table

bat = pd.read_csv("data/raw/battery_monthly.csv")
vt = vehicle_table(bat)
vt["fast_group"] = np.where(vt.fast_share >= vt.fast_share.median(), "High fast-charge share", "Low fast-charge share")
summary = {"n_vehicles": int(len(vt)), "n_events": int(vt.event.sum())}

# ---------- 1. Kaplan-Meier: probability a battery stays above 80% SoH ----------
curves = []
for col, split in [("usage", "Usage type"), ("fast_group", "Fast-charging intensity")]:
    for name, d in vt.groupby(col):
        kmf = KaplanMeierFitter().fit(d.duration, d.event, label=name)
        sf = kmf.survival_function_.reset_index(); sf.columns = ["month", "survival"]
        ci = kmf.confidence_interval_.reset_index(drop=True)
        sf["lower"], sf["upper"] = ci.iloc[:, 0].values, ci.iloc[:, 1].values
        curves.append(sf.assign(split=split, group=name))
    a, b = [g for _, g in vt.groupby(col)]
    summary[f"logrank_p_{col}"] = float(logrank_test(a.duration, b.duration, a.event, b.event).p_value)
pd.concat(curves).to_csv("outputs/km_curves.csv", index=False)

# ---------- 2. Cox proportional hazards: which factors shorten battery life? ----------
cols = ["fast_share", "avg_temp", "cycles_pm", "dod"]
cx = vt[["duration", "event", "is_fleet"] + cols].copy()
cx[cols] = (cx[cols] - cx[cols].mean()) / cx[cols].std()          # hazard ratios are per 1 SD
cph = CoxPHFitter(penalizer=0.01).fit(cx, "duration", "event")
hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index()
hr.columns = ["covariate", "hazard_ratio", "ci_low", "ci_high", "p_value"]
hr.round(4).to_csv("outputs/cox_hazard_ratios.csv", index=False)
summary["cox_concordance"] = float(cph.concordance_index_)
print(hr.round(3).to_string(index=False), "\nConcordance:", round(cph.concordance_index_, 3))

# ---------- 3. Average SoH curves ----------
sc = (bat.groupby(["usage", "month_idx"]).soh_pct
         .agg(mean="mean", p10=lambda x: x.quantile(.1), p90=lambda x: x.quantile(.9), n="size").reset_index())
sc[sc.n >= 30].round(2).to_csv("outputs/soh_curves.csv", index=False)

# ---------- 4. Predict SoH six months ahead (vehicle-grouped split, no leakage) ----------
b = bat.sort_values(["vehicle_id", "month_idx"]).copy()
gb = b.groupby("vehicle_id")
b["soh_fut6"], b["soh_lag6"] = gb.soh_pct.shift(-6), gb.soh_pct.shift(6)
for c in ["fast_share", "avg_batt_temp", "dod", "cycles_month"]:
    b[c + "_r6"] = gb[c].transform(lambda x: x.rolling(6, min_periods=3).mean())
b["slope6"] = (b.soh_pct - b.soh_lag6) / 6
b["is_fleet"] = (b.usage == "Taxi/Fleet").astype(int)
b["city_code"] = b.city.astype("category").cat.codes
FEAT = ["soh_pct", "month_idx", "cum_cycles", "fast_share_r6", "avg_batt_temp_r6", "dod_r6",
        "cycles_month_r6", "slope6", "is_fleet", "city_code"]
use = b.dropna(subset=FEAT + ["soh_fut6"]).copy()
use["delta"] = use.soh_fut6 - use.soh_pct
tri, tei = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(use, groups=use.vehicle_id))
trn, tst = use.iloc[tri], use.iloc[tei]

gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, random_state=42, verbose=-1).fit(trn[FEAT], trn.delta)
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(trn[FEAT], trn.delta)
preds = {"Persistence (no change)": np.zeros(len(tst)), "Linear extrapolation": 6 * tst.slope6.values,
         "Ridge regression": ridge.predict(tst[FEAT]), "LightGBM": gbm.predict(tst[FEAT])}
rows = []
for k, d in preds.items():
    p = tst.soh_pct.values + d
    rows.append({"Model": k, "MAE_soh_pts": mean_absolute_error(tst.soh_fut6, p),
                 "RMSE_soh_pts": np.sqrt(mean_squared_error(tst.soh_fut6, p)), "R2_on_change": r2_score(tst.delta, d)})
bm = pd.DataFrame(rows).round(4); bm.to_csv("outputs/battery_model_metrics.csv", index=False)
print(bm.to_string(index=False))
pd.DataFrame({"feature": FEAT, "importance": gbm.booster_.feature_importance("gain")}).sort_values(
    "importance", ascending=False).to_csv("outputs/battery_feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(tst.soh_fut6, tst.soh_pct + preds["LightGBM"], s=4, alpha=.3)
lim = [tst.soh_fut6.min(), tst.soh_fut6.max()]; ax.plot(lim, lim, "k--")
ax.set(xlabel="Actual SoH in 6 months (%)", ylabel="Predicted (%)", title="6-month-ahead SoH forecast (unseen vehicles)")
plt.tight_layout(); plt.savefig("outputs/battery_pred_vs_actual.png", dpi=140); plt.close()

# ---------- 5. Predictive-maintenance watchlist ----------
last = b.groupby("vehicle_id").tail(1)
last = last[last.soh_pct >= 80].dropna(subset=FEAT).copy()
last["pred_soh_6m"] = last.soh_pct + gbm.predict(last[FEAT])
last["soh_loss_per_month"] = -last.slope6
last["risk"] = np.select([last.pred_soh_6m < 80, last.pred_soh_6m < 83], ["Critical: inspect/replace", "Watch"], "Healthy")
wl = last[["vehicle_id", "city", "usage", "model", "month_idx", "soh_pct", "pred_soh_6m", "soh_loss_per_month",
           "fast_share_r6", "avg_batt_temp_r6", "risk"]].round(3).sort_values("pred_soh_6m")
wl.to_csv("outputs/battery_watchlist.csv", index=False)
print(wl.risk.value_counts())
json.dump(summary, open("outputs/battery_summary.json", "w"), indent=2)