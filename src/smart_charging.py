import numpy as np, pandas as pd
from src.common import tou

ses = pd.read_csv("data/raw/sessions.csv", parse_dates=["start_time"])
t0 = ses.start_time.max().normalize() - pd.Timedelta(days=89)      # simulate on the last 90 days
s = ses[ses.start_time >= t0].reset_index(drop=True)
s["h"] = ((s.start_time.dt.floor("h") - t0) // pd.Timedelta(hours=1)).astype(int)
s["base_price"] = s.price_per_kwh / tou(s.start_time.dt.hour.values)
s["slack_h"] = np.floor((s.plugged_minutes - s.charge_minutes) / 60).clip(0, 10).astype(int)
flex = ((s.charger_type == "AC 7kW") & (s.slack_h >= 2)).values     # sessions with real flexibility

cities = sorted(s.city.unique())
cid = s.city.map({c: i for i, c in enumerate(cities)}).values
h, e, sl, bp = s.h.values, s.energy_kwh.values, s.slack_h.values, s.base_price.values
order = np.argsort(s.start_time.values)
H = int(h.max()) + 12
hod = np.arange(H) % 24
print(f"Sessions: {len(s):,} | flexible: {flex.sum():,} ({100 * flex.mean():.1f}%)")

def simulate(share, seed=7):
    part = flex & (np.random.default_rng(seed).random(len(s)) < share)
    load = np.zeros((len(cities), H))
    np.add.at(load, (cid[~part], h[~part]), e[~part])               # inflexible load stays put
    new_h = h.copy()
    for i in order[part[order]]:                                     # greedy valley-filling, chronological
        window = load[cid[i], h[i]: h[i] + sl[i] + 1]
        new_h[i] = h[i] + int(np.argmin(window))
        load[cid[i], new_h[i]] += e[i]
    return load, new_h, int(part.sum())

base_cost = (e * bp * tou(h % 24)).sum()
rows, prof = [], {}
base_peak = None
for share in [0, .25, .5, .75, 1.0]:
    load, new_h, nshift = simulate(share)
    net = load.sum(0)
    base_peak = net.max() if share == 0 else base_peak
    cost = (e * bp * tou(new_h % 24)).sum()
    rows.append({"adoption_pct": int(share * 100), "network_peak_kw": round(net.max(), 1),
                 "peak_reduction_pct": round(100 * (base_peak - net.max()) / base_peak, 2),
                 "load_factor_pct": round(100 * net.mean() / net.max(), 2),
                 "evening_peak_share_pct": round(100 * net[(hod >= 18) & (hod < 22)].sum() / net.sum(), 2),
                 "customer_cost_savings_pct": round(100 * (base_cost - cost) / base_cost, 2),
                 "sessions_shifted": nshift})
    prof["baseline" if share == 0 else f"adoption_{int(share * 100)}"] = (
        pd.Series(net).groupby(hod).mean().round(2).values)
sc = pd.DataFrame(rows)
sc.to_csv("outputs/smart_charging_scenarios.csv", index=False)
pd.DataFrame({"hour": range(24), **prof}).to_csv("outputs/smart_charging_profile.csv", index=False)
print(sc.to_string(index=False))
