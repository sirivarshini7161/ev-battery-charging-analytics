import json
import numpy as np, pandas as pd

ses = pd.read_csv("data/raw/sessions.csv", parse_dates=["start_time"])
st = pd.read_csv("data/raw/stations.csv")
ses["month"] = ses.start_time.dt.to_period("M").astype(str)

mo = ses.groupby(["city", "month"]).agg(sessions=("energy_kwh", "size"), energy_kwh=("energy_kwh", "sum"),
                                        revenue_inr=("revenue_inr", "sum")).reset_index()
mo.round(1).to_csv("outputs/overview_monthly.csv.gz", index=False)

dates = pd.date_range(ses.start_time.min().normalize(), ses.start_time.max().normalize())
ndays = pd.Series(dates.dayofweek).value_counts()
hm = (ses.assign(dow=ses.start_time.dt.dayofweek, hour=ses.start_time.dt.hour)
         .groupby(["location_type", "dow", "hour"]).energy_kwh.sum().reset_index())
hm["avg_kwh"] = (hm.energy_kwh / hm.dow.map(ndays)).round(2)
hm.drop(columns="energy_kwh").to_csv("outputs/heatmap.csv.gz", index=False)

hrs = (ses.start_time.max() - ses.start_time.min()).total_seconds() / 3600
ss = (ses.groupby("station_id").agg(sessions=("energy_kwh", "size"), energy_kwh=("energy_kwh", "sum"),
        revenue_inr=("revenue_inr", "sum"), avg_session_kwh=("energy_kwh", "mean"),
        interruption_pct=("status", lambda x: (x == "Interrupted").mean() * 100),
        charge_h=("charge_minutes", lambda x: x.sum() / 60)).reset_index().merge(st, on="station_id"))
ss["utilization_pct"] = 100 * ss.charge_h / (ss.ports * hrs)
ss.drop(columns="charge_h").round(2).to_csv("outputs/station_summary.csv.gz", index=False)

h = ses.start_time.dt.hour
k = {"sessions": int(len(ses)), "energy_mwh": float(ses.energy_kwh.sum() / 1000),
     "revenue_cr": float(ses.revenue_inr.sum() / 1e7), "avg_session_kwh": float(ses.energy_kwh.mean()),
     "interruption_pct": float((ses.status == "Interrupted").mean() * 100),
     "dc_energy_share_pct": float(100 * ses.loc[ses.charger_type.str.startswith("DC"), "energy_kwh"].sum() / ses.energy_kwh.sum()),
     "evening_peak_share_pct": float(100 * ses.loc[(h >= 18) & (h < 22), "energy_kwh"].sum() / ses.energy_kwh.sum()),
     "stations": int(len(st)), "users": int(ses.user_id.nunique()),
     "period": f"{ses.start_time.min().date()} to {ses.start_time.max().date()}"}
json.dump(k, open("outputs/kpis.json", "w"), indent=2)
print(k)