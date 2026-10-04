import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ses = pd.read_csv("data/raw/sessions.csv", parse_dates=["start_time"],
                  usecols=["user_id", "station_id", "charger_type", "start_time", "energy_kwh", "plugged_minutes", "soc_start"])
h = ses.start_time.dt.hour
ses["is_dc"] = ses.charger_type.str.startswith("DC").astype(int)
ses["night"] = ((h >= 21) | (h < 6)).astype(int)
ses["peak"] = ((h >= 18) & (h < 22)).astype(int)
ses["weekend"] = (ses.start_time.dt.dayofweek >= 5).astype(int)
months = (ses.start_time.max() - ses.start_time.min()).days / 30.4

u = ses.groupby("user_id").agg(sessions=("energy_kwh", "size"), total_kwh=("energy_kwh", "sum"),
        avg_kwh=("energy_kwh", "mean"), dc_share=("is_dc", "mean"), night_share=("night", "mean"),
        peak_share=("peak", "mean"), weekend_share=("weekend", "mean"), avg_plugged=("plugged_minutes", "mean"),
        n_stations=("station_id", "nunique")).reset_index()
u = u[u.sessions >= 3].copy()
u["sessions_pm"] = u.sessions / months

F = ["sessions_pm", "avg_kwh", "dc_share", "night_share", "peak_share", "weekend_share", "avg_plugged", "n_stations"]
X = u[F].copy()
for c in ["sessions_pm", "avg_plugged", "n_stations"]:
    X[c] = np.log1p(X[c])
Xs = StandardScaler().fit_transform(X)

samp = np.random.default_rng(0).choice(len(Xs), min(5000, len(Xs)), replace=False)
sil = [(k, silhouette_score(Xs[samp], KMeans(k, n_init=10, random_state=42).fit_predict(Xs[samp]))) for k in range(3, 8)]
pd.DataFrame(sil, columns=["k", "silhouette"]).round(4).to_csv("outputs/silhouette.csv", index=False)
K = max(sil, key=lambda t: t[1])[0]
print("Silhouette:", [(k, round(s, 3)) for k, s in sil], "-> chosen k =", K)

u["cluster"] = KMeans(K, n_init=10, random_state=42).fit_predict(Xs)
prof = u.groupby("cluster")[F].mean()
prof["users"] = u.groupby("cluster").size()
prof["energy_share_pct"] = 100 * u.groupby("cluster").total_kwh.sum() / u.total_kwh.sum()
med = u.sessions_pm.median()
def label(r, c):
    t = []
    if r.sessions_pm >= 1.8 * med: t.append("Heavy-use")
    t.append("DC fast" if r.dc_share >= 0.5 else "AC")
    t.append("night" if r.night_share >= 0.4 else "daytime")
    return f"S{c} · " + " · ".join(t)
names = {c: label(r, c) for c, r in prof.iterrows()}
u["segment"] = u.cluster.map(names)
prof.insert(0, "segment", prof.index.map(names))
prof.round(3).to_csv("outputs/user_segment_profile.csv", index=False)

pc = PCA(2, random_state=0).fit(Xs)
idx = np.random.default_rng(1).choice(len(u), min(3000, len(u)), replace=False)
sc = pc.transform(Xs[idx])
pd.DataFrame({"pc1": sc[:, 0].round(3), "pc2": sc[:, 1].round(3), "segment": u.segment.values[idx]}
             ).to_csv("outputs/user_segment_scatter.csv.gz", index=False)
print(prof.round(2).to_string(index=False))