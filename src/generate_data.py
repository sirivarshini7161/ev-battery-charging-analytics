import os
import numpy as np, pandas as pd
from src.common import tou

rng = np.random.default_rng(42)
os.makedirs("data/raw", exist_ok=True)

START, END = "2024-10-01", "2026-09-30"
days = pd.date_range(START, END, freq="D")
hours = pd.date_range(START, END + " 23:00", freq="h")
D, T = len(days), len(hours)
hod, dow = hours.hour.values, hours.dayofweek.values
wk = dow >= 5

CITY = {"Hyderabad": (27, 5, 1.0), "Bengaluru": (24, 3, 1.2), "Mumbai": (28, 3, 1.1),
        "Delhi": (25, 11, 1.3), "Chennai": (29, 4, 0.9)}      # mean temp, seasonal amp, EV scale
CN = list(CITY)
CHG = {"AC 7kW": (7, 0.22, 14.0), "DC 30kW": (30, 0.35, 18.0), "DC 60kW": (60, 0.50, 21.0)}  # kW, arrivals/h, Rs/kWh
LOCS = ["Mall", "Highway", "Office", "Residential"]

# ---------------- stations ----------------
NS = 40
stations = pd.DataFrame({
    "station_id": [f"ST{i:03d}" for i in range(1, NS + 1)],
    "city": np.resize(CN, NS),
    "location_type": rng.choice(LOCS, NS, p=[.30, .20, .30, .20]),
    "charger_type": rng.choice(list(CHG), NS, p=[.45, .35, .20]),
    "ports": rng.integers(2, 9, NS)})
stations.to_csv("data/raw/stations.csv", index=False)

# ---------------- weather & calendar ----------------
rain_p = np.where(np.isin(days.month, [6, 7, 8, 9]), 0.5, 0.08)
temp, rain = {}, {}
for c, (m, a, _) in CITY.items():
    daily = m + a * np.sin(2 * np.pi * (days.dayofyear.values - 110) / 365) + rng.normal(0, 1.5, D)
    temp[c] = np.repeat(daily, 24) + 4 * np.sin(2 * np.pi * (hod - 9) / 24)
    rain[c] = np.repeat(rng.random(D) < rain_p, 24)
FEST = pd.to_datetime(["2024-10-31", "2024-11-01", "2025-01-14", "2025-03-14", "2025-08-15",
                       "2025-10-20", "2025-10-21", "2025-12-25", "2026-01-26", "2026-03-04", "2026-08-15"])
festival = hours.normalize().isin(FEST)
wx = pd.concat([pd.DataFrame({"city": c, "timestamp": hours, "temp_c": temp[c].round(2),
                              "rain": rain[c].astype(int), "festival": festival.astype(int)}) for c in CN])
wx.to_csv("data/raw/weather.csv", index=False)

# ---------------- demand intensity ----------------
hh = np.arange(24)
g = lambda mu, sd: np.exp(-0.5 * ((hh - mu) / sd) ** 2)
PROFILE = {"Office": 0.15 + g(10, 2.0) + 0.3 * g(14, 2.5),
           "Mall": 0.15 + 0.5 * g(13, 2.5) + g(19, 2.2),
           "Residential": 0.2 + g(21, 2.5) + 0.35 * g(7, 1.5),
           "Highway": 0.3 + g(11, 3.5) + 0.8 * g(17, 3.0)}
PROFILE = {k: v / v.mean() for k, v in PROFILE.items()}
WEEKEND = {"Office": (1.25, 0.5), "Mall": (0.9, 1.35), "Residential": (0.97, 1.08), "Highway": (0.9, 1.3)}
FEST_EFF = {"Office": 0.6, "Mall": 1.25, "Highway": 1.25, "Residential": 1.0}
growth = 1 + 0.6 * np.arange(T) / T                        # EV adoption growth
st_factor = rng.lognormal(0, 0.25, NS)

# ---------------- charger faults (hidden ground truth): 1 = degraded, 2 = outage ----------------
fault = np.zeros((NS, D), dtype=int)
for i in rng.choice(NS, 10, replace=False):
    for _ in range(3):
        L = int(rng.integers(3, 9)); s0 = int(rng.integers(30, D - L))
        fault[i, s0:s0 + L] = int(rng.choice([1, 2]))
fl = [(stations.station_id[i], days[d].date(), int(fault[i, d])) for i, d in zip(*np.nonzero(fault))]
pd.DataFrame(fl, columns=["station_id", "date", "fault_state"]).to_csv("data/raw/fault_log.csv", index=False)

lam = np.zeros((NS, T))
for i, r in stations.iterrows():
    wf = np.where(wk, WEEKEND[r.location_type][1], WEEKEND[r.location_type][0])
    fest = np.where(festival, FEST_EFF[r.location_type], 1.0)
    t_ = temp[r.city]
    weather = (1 + 0.012 * np.clip(t_ - 32, 0, None) + 0.015 * np.clip(18 - t_, 0, None)) * np.where(rain[r.city], 0.88, 1.0)
    fm = np.repeat(fault[i], 24)
    fmult = np.where(fm == 2, 0.0, np.where(fm == 1, 0.7, 1.0))
    lam[i] = (CHG[r.charger_type][1] * st_factor[i] * CITY[r.city][2] * PROFILE[r.location_type][hod]
              * wf * fest * weather * growth * fmult)

cnt = rng.poisson(lam)
si, hi = np.nonzero(cnt)
rep = cnt[si, hi]
si, hi = np.repeat(si, rep), np.repeat(hi, rep)
n = len(si)

# ---------------- session attributes ----------------
VEH = pd.DataFrame({
    "model": ["Tata Nexon EV", "Tata Tiago EV", "MG Windsor EV", "MG ZS EV", "Mahindra XUV400",
              "BYD Atto 3", "Hyundai Ioniq 5", "MG Comet EV"],
    "kwh": [40.5, 24, 38, 50.3, 39.4, 60.5, 72.6, 17.3],
    "ac_kw": [7.2, 3.3, 7.4, 7.4, 7.2, 7.0, 7.2, 3.3],
    "dc_kw": [50, 25, 45, 76, 50, 80, 150, 0],
    "p": [.24, .18, .12, .10, .14, .08, .04, .10]})

st_id = stations.station_id.values[si]; st_city = stations.city.values[si]
st_loc = stations.location_type.values[si]; st_type = stations.charger_type.values[si]
kw = np.array([CHG[t][0] for t in stations.charger_type])[si]
price_base = np.array([CHG[t][2] for t in stations.charger_type])[si]
is_dc = kw > 7
cidx = np.array([CN.index(c) for c in stations.city])[si]
amb = np.vstack([temp[c] for c in CN])[cidx, hi]
fstate = fault[si, hi // 24]
start = hours.values[hi] + rng.integers(0, 60, n).astype("timedelta64[m]")

p_dc = np.where(VEH.dc_kw.values > 0, VEH.p.values, 0.0); p_dc = p_dc / p_dc.sum()
v = np.where(is_dc, rng.choice(len(VEH), n, p=p_dc), rng.choice(len(VEH), n, p=VEH.p.values))
cap = VEH.kwh.values[v]
lim = np.where(is_dc, VEH.dc_kw.values[v], VEH.ac_kw.values[v])

soc0 = np.clip(rng.beta(2, 4.5, n) * 0.75 + 0.05, 0.05, 0.8)
target = np.where(is_dc, np.clip(rng.normal(0.80, 0.07, n), 0.6, 0.95), np.clip(rng.normal(0.92, 0.10, n), 0.6, 1.0))
soc1 = np.minimum(np.maximum(soc0 + 0.05, target), 1.0)
interrupted = rng.random(n) < np.where(fstate == 1, 0.45, 0.03)
frac = np.where(interrupted, rng.uniform(0.1, 0.6, n), 1.0)
soc1 = soc0 + (soc1 - soc0) * frac
energy = (soc1 - soc0) * cap / 0.92                        # billed kWh (92% charging efficiency)

thermal = np.clip(1 - 0.006 * np.clip(np.abs(amb - 25) - 7, 0, None), 0.6, 1.0)
taper = 1 - 0.5 * np.clip(soc1 - 0.8, 0, None)
avg_p = np.minimum(kw, lim) * np.where(is_dc, 0.88, 0.95) * thermal * taper * np.where(fstate == 1, 0.55, 1.0)
charge_min = energy / avg_p * 60
idle_mean = np.where(is_dc, 8, np.select([st_loc == "Office", st_loc == "Residential", st_loc == "Mall"], [240, 300, 40], 10))
plugged = charge_min + np.minimum(rng.exponential(idle_mean), 600)
price = price_base * tou(hod[hi])

# ---------------- users (persistent behaviour archetypes) ----------------
ARCH = ["Commuter", "NightHome", "Traveller", "Fleet"]
P_ARCH = {"Office": [.8, .05, .05, .10], "Residential": [.1, .8, .0, .10],
          "Mall": [.35, .35, .2, .10], "Highway": [.05, .0, .7, .25]}
W_DC, W_AC = np.array([.3, .1, 2., 3.]), np.array([1.2, 1.2, .4, .3])
arch = np.empty(n, dtype=object)
for loc, p in P_ARCH.items():
    for dc in (True, False):
        m = (st_loc == loc) & (is_dc == dc)
        q = np.array(p) * (W_DC if dc else W_AC); q = q / q.sum()
        arch[m] = rng.choice(ARCH, m.sum(), p=q)
POOL = {"Commuter": 1500, "NightHome": 1500, "Traveller": 2500, "Fleet": 120}
user = np.empty(n, dtype=object)
for c in CN:
    for a, size in POOL.items():
        m = (st_city == c) & (arch == a)
        user[m] = [f"{c[:3].upper()}-{a[:3].upper()}-{k}" for k in rng.integers(0, size, m.sum())]

ses = pd.DataFrame({
    "station_id": st_id, "city": st_city, "location_type": st_loc, "charger_type": st_type,
    "user_id": user, "vehicle_model": VEH.model.values[v], "battery_kwh": cap, "start_time": start,
    "soc_start": soc0.round(3), "soc_end": soc1.round(3), "energy_kwh": energy.round(3),
    "charge_minutes": charge_min.round(1), "plugged_minutes": plugged.round(1),
    "avg_power_kw": avg_p.round(2), "ambient_temp_c": amb.round(1), "price_per_kwh": price.round(2),
    "revenue_inr": (energy * price).round(2), "status": np.where(interrupted, "Interrupted", "Completed")})
ses = ses.sort_values("start_time").reset_index(drop=True)
ses.insert(0, "session_id", np.arange(1, len(ses) + 1))
ses.to_csv("data/raw/sessions.csv", index=False)

# ---------------- battery telemetry panel (physics-inspired degradation) ----------------
NV = 1500
veh = pd.DataFrame({"vehicle_id": [f"EV{i:04d}" for i in range(NV)],
                    "city": rng.choice(CN, NV),
                    "model": rng.choice(VEH.model.values, NV, p=VEH.p.values),
                    "usage": rng.choice(["Private", "Taxi/Fleet"], NV, p=[.7, .3])})
rows = []
for _, r in veh.iterrows():
    fleet = r.usage == "Taxi/Fleet"
    base_cyc = max(4, rng.normal(38 if fleet else 14, 8 if fleet else 4))
    base_fast = rng.beta(5, 3) if fleet else rng.beta(2, 6)
    base_dod = np.clip(rng.normal(0.70 if fleet else 0.55, 0.10), 0.25, 0.95)
    mean_t, amp, _ = CITY[r.city]
    start_m, off, vf = rng.integers(0, 12), rng.normal(0, 1.5), rng.lognormal(0, 0.15)
    n_obs = int(rng.integers(12, 61)); t = np.arange(1, n_obs + 1)
    cal = (start_m + t) % 12
    tmp = mean_t + 3 + amp * np.sin(2 * np.pi * (cal * 30.4 - 110) / 365) + off + rng.normal(0, 1, n_obs)
    cyc = np.clip(rng.normal(base_cyc, base_cyc * 0.12, n_obs), 1, None)
    fast = np.clip(base_fast + rng.normal(0, 0.05, n_obs), 0, 1)
    dod = np.clip(base_dod + rng.normal(0, 0.04, n_obs), 0.2, 1)
    stress = (1 + 0.8 * fast + 1.2 * (dod - 0.5)) * np.exp(0.045 * (tmp - 25))
    fade_c = np.cumsum(0.006 * cyc * stress) * vf
    fade_cal = 0.6 * np.sqrt(t) * np.exp(0.045 * (np.cumsum(tmp) / t - 25))
    soh = 100 - fade_cal - fade_c + rng.normal(0, 0.35, n_obs)
    below = np.nonzero(soh < 80)[0]
    end = int(below[0]) + 1 if len(below) else n_obs        # stop at end-of-life
    rows.append(pd.DataFrame({"vehicle_id": r.vehicle_id, "month_idx": t, "usage": r.usage, "city": r.city,
        "model": r.model, "cycles_month": cyc.round(1), "cum_cycles": np.cumsum(cyc).round(0),
        "fast_share": fast.round(3), "dod": dod.round(3), "avg_batt_temp": tmp.round(1),
        "soh_pct": soh.round(2)}).iloc[:end])
bat = pd.concat(rows, ignore_index=True)
bat.to_csv("data/raw/battery_monthly.csv", index=False)

print(f"Sessions: {len(ses):,} | Stations: {NS} | Fault days: {(fault > 0).sum()} | "
      f"Battery rows: {len(bat):,} | Vehicles reaching EOL: {bat.groupby('vehicle_id').soh_pct.last().lt(80).sum()}")