import numpy as np, pandas as pd

def tou(h):
    """Time-of-use tariff multiplier: evening peak 18-22h costs more, night is cheaper."""
    h = np.asarray(h)
    return np.where((h >= 18) & (h < 22), 1.30, np.where((h >= 23) | (h < 6), 0.80, 1.00))

def hourly_demand(ses):
    """City x hour energy demand on a complete hourly grid (zeros filled)."""
    s = ses.assign(ts=ses.start_time.dt.floor("h"))
    g = (s.groupby(["city", "ts"])
          .agg(energy_kwh=("energy_kwh", "sum"), sessions=("energy_kwh", "size")).reset_index())
    rng = pd.date_range(s.start_time.min().floor("D"),
                        s.start_time.max().floor("D") + pd.Timedelta(hours=23), freq="h")
    idx = pd.MultiIndex.from_product([sorted(s.city.unique()), rng], names=["city", "ts"])
    return g.set_index(["city", "ts"]).reindex(idx, fill_value=0).reset_index()

def vehicle_table(bat):
    """One row per vehicle for survival analysis and regression."""
    b = bat.sort_values(["vehicle_id", "month_idx"])
    vt = (b.groupby("vehicle_id")
           .agg(city=("city", "first"), usage=("usage", "first"), model=("model", "first"),
                duration=("month_idx", "max"), last_soh=("soh_pct", "last"),
                fast_share=("fast_share", "mean"), avg_temp=("avg_batt_temp", "mean"),
                cycles_pm=("cycles_month", "mean"), dod=("dod", "mean")).reset_index())
    vt["event"] = (vt.last_soh < 80).astype(int)          # end of life = SoH below 80%
    vt["loss_pm"] = (100 - vt.last_soh) / vt.duration     # SoH points lost per month
    vt["is_fleet"] = (vt.usage == "Taxi/Fleet").astype(int)
    return vt