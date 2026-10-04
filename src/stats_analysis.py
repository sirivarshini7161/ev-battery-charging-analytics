import os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.multitest import multipletests
from src.common import hourly_demand, vehicle_table

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(0)
ses = pd.read_csv("data/raw/sessions.csv", parse_dates=["start_time"])
wx = pd.read_csv("data/raw/weather.csv", parse_dates=["timestamp"])
bat = pd.read_csv("data/raw/battery_monthly.csv")
dem = hourly_demand(ses)

# ---------- 1. Time-series diagnostics: stationarity (ADF) and STL seasonality strength ----------
def strength(part, resid):
    return max(0.0, 1 - np.var(resid) / np.var(part + resid))

rows = []
for c, g in dem.groupby("city"):
    x = g.energy_kwh.values
    daily = g.set_index("ts").energy_kwh.resample("D").sum().values
    s24, s168 = STL(x, period=24).fit(), STL(x, period=168).fit()
    rows.append({"city": c,
                 "adf_p_level": adfuller(daily, autolag="AIC")[1],
                 "adf_p_differenced": adfuller(np.diff(daily), autolag="AIC")[1],
                 "daily_cycle_strength": strength(s24.seasonal, s24.resid),
                 "weekly_cycle_strength": strength(s168.seasonal, s168.resid),
                 "trend_strength": strength(s168.trend, s168.resid)})
tsd = pd.DataFrame(rows).round(4)
tsd.to_csv("outputs/ts_diagnostics.csv", index=False)
print(tsd.to_string(index=False))

g = dem[dem.city == "Hyderabad"]; r = STL(g.energy_kwh.values, period=168).fit(); n = 24 * 28
fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for a, (nm, vv) in zip(ax, [("Observed kWh", g.energy_kwh.values), ("Trend", r.trend),
                            ("Weekly seasonal", r.seasonal), ("Residual", r.resid)]):
    a.plot(g.ts.values[-n:], vv[-n:], lw=1); a.set_ylabel(nm)
ax[0].set_title("STL decomposition: Hyderabad hourly charging demand (last 28 days)")
plt.tight_layout(); plt.savefig("outputs/stl_decomposition.png", dpi=140); plt.close()

# ---------- 2. Hypothesis tests ----------
tests = []
def add(name, stat, p, eff, finding):
    tests.append({"test": name, "statistic": float(stat), "p_value": float(p),
                  "effect_size": float(eff), "finding": finding})

ses["date"] = ses.start_time.dt.normalize()
ld = ses.groupby(["location_type", "date"]).energy_kwh.sum().reset_index()
ld["weekend"] = ld.date.dt.dayofweek >= 5
for loc in ["Office", "Mall"]:
    d = ld[ld.location_type == loc]
    a, b = d.loc[~d.weekend, "energy_kwh"], d.loc[d.weekend, "energy_kwh"]
    u, p = stats.mannwhitneyu(a, b)
    add(f"{loc}: weekday vs weekend daily demand (Mann-Whitney U)", u, p, 2 * u / (len(a) * len(b)) - 1,
        f"median weekday {a.median():,.0f} vs weekend {b.median():,.0f} kWh/day (effect = rank-biserial r)")

# weather effects on demand, after removing trend and day-of-week with a log-linear model
cd = dem.assign(date=dem.ts.dt.normalize()).groupby(["city", "date"]).energy_kwh.sum().reset_index()
cd["dow"] = cd.date.dt.dayofweek; cd["t"] = (cd.date - cd.date.min()).dt.days; cd["resid"] = 0.0
for c, g in cd.groupby("city"):
    cd.loc[g.index, "resid"] = smf.ols("np.log(energy_kwh) ~ t + C(dow)", data=g).fit().resid
wd = (wx.assign(date=wx.timestamp.dt.normalize()).groupby(["city", "date"])
        .agg(temp=("temp_c", "mean"), rain=("rain", "max")).reset_index())
dd = cd.merge(wd, on=["city", "date"])

a, b = dd.loc[dd.rain == 1, "resid"], dd.loc[dd.rain == 0, "resid"]
u, p = stats.mannwhitneyu(a, b)
add("Rainy vs dry days: demand residual (Mann-Whitney U)", u, p, 2 * u / (len(a) * len(b)) - 1,
    f"{100 * (np.exp(a.mean() - b.mean()) - 1):+.1f}% demand on rainy days after trend/weekday adjustment")

dd["extreme"] = (dd.temp > 32) | (dd.temp < 18)
a, b = dd.loc[dd.extreme, "resid"], dd.loc[~dd.extreme, "resid"]
u, p = stats.mannwhitneyu(a, b)
add("Extreme-temperature vs mild days: demand residual (Mann-Whitney U)", u, p, 2 * u / (len(a) * len(b)) - 1,
    f"{100 * (np.exp(a.mean() - b.mean()) - 1):+.1f}% demand on extreme-temperature days")

# DC vs AC energy per session (Welch t-test + bootstrap CI + Cohen's d)
dc = ses.loc[ses.charger_type.str.startswith("DC"), "energy_kwh"].values
ac = ses.loc[ses.charger_type == "AC 7kW", "energy_kwh"].values
t_, p = stats.ttest_ind(dc, ac, equal_var=False)
d_ = (dc.mean() - ac.mean()) / np.sqrt((dc.var(ddof=1) + ac.var(ddof=1)) / 2)
boot = [rng.choice(dc, len(dc)).mean() - rng.choice(ac, len(ac)).mean() for _ in range(300)]
lo, hi = np.percentile(boot, [2.5, 97.5])
add("DC vs AC energy per session (Welch t-test)", t_, p, d_,
    f"mean difference {dc.mean() - ac.mean():.2f} kWh, bootstrap 95% CI [{lo:.2f}, {hi:.2f}] (effect = Cohen's d)")

# battery tests at vehicle level
vt = vehicle_table(bat)
H, p = stats.kruskal(*[g.loss_pm.values for _, g in vt.groupby("city")])
k = vt.city.nunique()
add("SoH loss rate across cities (Kruskal-Wallis)", H, p, (H - k + 1) / (len(vt) - k),
    "medians: " + ", ".join(f"{c} {g.loss_pm.median():.3f}" for c, g in vt.groupby("city")) + " pts/month (effect = epsilon-squared)")
rho, p = stats.spearmanr(vt.fast_share, vt.loss_pm)
add("Fast-charge share vs SoH loss rate (Spearman)", rho, p, rho,
    "higher fast-charging share is associated with faster capacity fade" if rho > 0 else "no positive association")

res = pd.DataFrame(tests)
res["p_adj_BH"] = multipletests(res.p_value, method="fdr_bh")[1]      # Benjamini-Hochberg FDR correction
res["significant_5pct_FDR"] = res.p_adj_BH < 0.05
res.to_csv("outputs/stat_tests.csv", index=False)
print(res[["test", "p_value", "p_adj_BH", "effect_size", "significant_5pct_FDR"]].to_string(index=False))

# ---------- 3. Robust regression: what drives battery fade? ----------
m = smf.ols("loss_pm ~ fast_share + avg_temp + cycles_pm + dod + is_fleet", data=vt).fit(cov_type="HC3")
ci = m.conf_int()
ols = pd.DataFrame({"coef": m.params, "ci_low": ci[0], "ci_high": ci[1], "p_value": m.pvalues}).round(5)
ols.index.name = "term"; ols.reset_index().to_csv("outputs/battery_ols.csv", index=False)
json.dump({"battery_ols_r2": float(m.rsquared), "n_vehicles": int(len(vt))}, open("outputs/stats_summary.json", "w"))
print(ols, "\nR2 =", round(m.rsquared, 3))