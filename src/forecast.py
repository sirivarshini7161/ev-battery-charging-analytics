import os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ParameterSampler
from src.common import hourly_demand

os.makedirs("outputs", exist_ok=True)
ses = pd.read_csv("data/raw/sessions.csv", parse_dates=["start_time"])
wx = pd.read_csv("data/raw/weather.csv", parse_dates=["timestamp"]).rename(columns={"timestamp": "ts"})
df = (hourly_demand(ses).merge(wx, on=["city", "ts"], how="left")
        .sort_values(["city", "ts"]).reset_index(drop=True))
Y = "energy_kwh"

# ---------- features (all known at 00:00 of the forecast day) ----------
g = df.groupby("city")[Y]
for L in [24, 48, 72, 168, 336]:
    df[f"lag_{L}"] = g.shift(L)
df["roll24"] = g.transform(lambda x: x.shift(24).rolling(24).mean())
df["roll168"] = g.transform(lambda x: x.shift(24).rolling(168).mean())
df["roll24_std"] = g.transform(lambda x: x.shift(24).rolling(24).std())
df["hod_mean7"] = sum(g.shift(24 * k) for k in range(1, 8)) / 7
df["hour"], df["dow"], df["month"] = df.ts.dt.hour, df.ts.dt.dayofweek, df.ts.dt.month
df["is_weekend"] = (df.dow >= 5).astype(int)
df["heat"], df["cold"] = np.clip(df.temp_c - 32, 0, None), np.clip(18 - df.temp_c, 0, None)
df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
df["t"] = (df.ts - df.ts.min()).dt.days
df["city_c"] = df.city.astype("category")
FE = ["city_c", "hour", "dow", "month", "is_weekend", "temp_c", "rain", "festival", "heat", "cold",
      "lag_24", "lag_48", "lag_72", "lag_168", "lag_336", "roll24", "roll168", "roll24_std",
      "hod_mean7", "hour_sin", "hour_cos"]
df = df.dropna(subset=FE).reset_index(drop=True)

test_start = df.ts.max().normalize() - pd.Timedelta(days=89)       # last 90 days = hold-out
tr, te = df[df.ts < test_start], df[df.ts >= test_start]

# ---------- hyper-parameter search: rolling-origin validation (3 folds x 45 days) ----------
space = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.08], "n_estimators": [300, 500, 700],
         "min_child_samples": [20, 50, 100], "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9]}
def mk(p, **kw):
    return lgb.LGBMRegressor(**p, subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1, **kw)
folds = [(test_start - pd.Timedelta(days=d), test_start - pd.Timedelta(days=d - 45)) for d in (135, 90, 45)]
best, best_mae = None, 1e18
for p in ParameterSampler(space, n_iter=6, random_state=42):
    maes = []
    for cut, vend in folds:
        a, b = tr[tr.ts < cut], tr[(tr.ts >= cut) & (tr.ts < vend)]
        maes.append(np.abs(b[Y] - mk(p).fit(a[FE], a[Y]).predict(b[FE])).mean())
    print(p, "CV MAE:", round(np.mean(maes), 2))
    if np.mean(maes) < best_mae: best, best_mae = p, np.mean(maes)
print("Best params:", best)

gbm = mk(best).fit(tr[FE], tr[Y])
q10 = mk(best, objective="quantile", alpha=0.10).fit(tr[FE], tr[Y])
q90 = mk(best, objective="quantile", alpha=0.90).fit(tr[FE], tr[Y])

def ridge_X(d):
    X = d[["lag_24", "lag_168", "roll168", "hod_mean7", "temp_c", "rain", "festival", "t"]].copy()
    X["heat"], X["cold"] = d.heat, d.cold
    return pd.concat([X, pd.get_dummies(d.hour, prefix="h"), pd.get_dummies(d.dow, prefix="d"),
                      pd.get_dummies(d.city, prefix="c")], axis=1).astype(float)
ridge = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(ridge_X(tr), tr[Y])

# ---------- evaluation on the unseen last 90 days ----------
yt = te[Y].values
pred = np.clip(gbm.predict(te[FE]), 0, None)
lo = np.minimum(np.clip(q10.predict(te[FE]), 0, None), pred)
hi = np.maximum(q90.predict(te[FE]), pred)
peak = te.hour.between(18, 21).values
cands = {"Naive (t-24h)": te.lag_24.values, "Seasonal naive (t-168h)": te.lag_168.values,
         "Ridge regression": np.clip(ridge.predict(ridge_X(te)), 0, None), "LightGBM (tuned)": pred}

def met(y, p, pk):
    e = y - p
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "WAPE_pct": 100 * np.abs(e).sum() / y.sum(),
            "Peak_MAE": np.abs(e[pk]).mean(), "Bias": (p - y).mean()}
M = pd.DataFrame([{"Model": k, **met(yt, v, peak)} for k, v in cands.items()]).round(3)
M.to_csv("outputs/forecast_metrics.csv", index=False); print(M.to_string(index=False))

bc = []
for c in sorted(te.city.unique()):
    mk_ = (te.city == c).values
    for nm in ["Seasonal naive (t-168h)", "LightGBM (tuned)"]:
        bc.append({"city": c, "model": nm, **met(yt[mk_], cands[nm][mk_], peak[mk_])})
pd.DataFrame(bc).round(3)[["city", "model", "MAE", "WAPE_pct"]].to_csv("outputs/forecast_metrics_by_city.csv", index=False)

cover = float(((yt >= lo) & (yt <= hi)).mean())
w_sn, w_gb = M.set_index("Model").WAPE_pct["Seasonal naive (t-168h)"], M.set_index("Model").WAPE_pct["LightGBM (tuned)"]
summary = {"coverage_80": cover, "avg_interval_width": float((hi - lo).mean()),
           "improvement_vs_seasonal_naive_pct": float(100 * (w_sn - w_gb) / w_sn),
           "best_params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in best.items()},
           "test_start": str(test_start.date())}
json.dump(summary, open("outputs/forecast_summary.json", "w"), indent=2)
print(f"80% interval coverage: {cover:.1%} | WAPE improvement vs seasonal naive: {summary['improvement_vs_seasonal_naive_pct']:.1f}%")

out = te[["city", "ts"]].assign(actual=yt.round(2), forecast=pred.round(2), p10=lo.round(2),
                                p90=hi.round(2), naive_168=te.lag_168.values.round(2))
out.to_csv("outputs/forecast_test.csv.gz", index=False)
fi = pd.DataFrame({"feature": gbm.feature_name_, "importance": gbm.booster_.feature_importance("gain")})
fi["importance"] = (100 * fi.importance / fi.importance.sum()).round(2)
fi.sort_values("importance", ascending=False).to_csv("outputs/feature_importance.csv", index=False)

w = out[out.city == "Hyderabad"].tail(24 * 14)
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(w.ts, w.p10, w.p90, alpha=.25, label="80% prediction interval")
ax.plot(w.ts, w.actual, lw=1.2, label="Actual"); ax.plot(w.ts, w.forecast, lw=1.2, label="Forecast")
ax.set_title("Hyderabad: day-ahead hourly demand forecast (hold-out, last 14 days)"); ax.set_ylabel("kWh per hour")
ax.legend(); plt.tight_layout(); plt.savefig("outputs/forecast_plot.png", dpi=140); plt.close()