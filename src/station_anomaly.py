import json
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

ses = pd.read_csv("data/raw/sessions.csv", parse_dates=["start_time"])
st = pd.read_csv("data/raw/stations.csv")
fl = pd.read_csv("data/raw/fault_log.csv", parse_dates=["date"])
ses["date"] = ses.start_time.dt.normalize()
ses["intr"] = (ses.status == "Interrupted").astype(int)
ses["charge_h"] = ses.charge_minutes / 60

daily = ses.groupby(["station_id", "date"]).agg(sessions=("energy_kwh", "size"), energy=("energy_kwh", "sum"),
                                                charge_h=("charge_h", "sum"), intr=("intr", "sum"))
idx = pd.MultiIndex.from_product([st.station_id, pd.date_range(ses.date.min(), ses.date.max())], names=["station_id", "date"])
daily = daily.reindex(idx, fill_value=0).reset_index()
daily["dow"] = daily.date.dt.dayofweek

# features relative to each station's own normal behaviour
daily["base"] = daily.groupby(["station_id", "dow"]).sessions.transform(lambda x: x.shift(1).rolling(6, min_periods=3).median())
daily["ratio"] = (daily.sessions + 1) / (daily.base + 1)
daily["int_rate"] = np.where(daily.sessions > 0, daily.intr / daily.sessions.clip(lower=1), 0.0)
daily["avg_power"] = np.where(daily.charge_h > 0, daily.energy / daily.charge_h.clip(lower=1e-9), np.nan)
daily["power_ratio"] = (daily.avg_power / daily.groupby("station_id").avg_power.transform("median")).fillna(1.0)
zero = daily.sessions == 0
grp = (~zero).groupby(daily.station_id).cumsum()
daily["zero_run"] = zero.astype(int).groupby([daily.station_id, grp]).cumsum()

F = ["ratio", "int_rate", "power_ratio", "zero_run"]
sc = daily.dropna(subset=["ratio"]).copy()
iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1).fit(sc[F])
sc["anomaly_score"] = -iso.score_samples(sc[F])
sc["flag"] = (iso.predict(sc[F]) == -1).astype(int)
sc["anomaly_type"] = np.where(sc.flag == 0, "Normal", np.where(sc.sessions == 0, "Outage",
                     np.where((sc.int_rate > 0.2) | (sc.power_ratio < 0.75), "Degraded charger", "Demand anomaly")))

daily = daily.merge(sc[["station_id", "date", "anomaly_score", "flag", "anomaly_type"]],
                    on=["station_id", "date"], how="left")
daily["flag"] = daily.flag.fillna(0).astype(int)
daily["anomaly_type"] = daily.anomaly_type.fillna("Insufficient history")
daily = daily.merge(fl.rename(columns={"fault_state": "injected_fault"}), on=["station_id", "date"], how="left")
daily["injected_fault"] = daily.injected_fault.fillna(0).astype(int)

ev = daily[daily.anomaly_type != "Insufficient history"]
truth, flag = ev.injected_fault > 0, ev.flag == 1
tp = int((truth & flag).sum())
prec, rec = tp / max(flag.sum(), 1), tp / max(truth.sum(), 1)
met = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
       "flagged_days": int(flag.sum()), "true_fault_days": int(truth.sum()),
       "outage_recall": float(flag[ev.injected_fault == 2].mean()),
       "degraded_recall": float(flag[ev.injected_fault == 1].mean())}
json.dump(met, open("outputs/anomaly_metrics.json", "w"), indent=2)
print({k: round(v, 3) if isinstance(v, float) else v for k, v in met.items()})

daily[["station_id", "date", "sessions", "int_rate", "power_ratio", "anomaly_score", "flag", "anomaly_type",
       "injected_fault"]].round(4).to_csv("outputs/station_daily_health.csv.gz", index=False)