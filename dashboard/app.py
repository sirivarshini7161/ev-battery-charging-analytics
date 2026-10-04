import os, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EV Charging & Battery Analytics", page_icon="⚡", layout="wide")
O = "outputs"

@st.cache_data
def rd(name, **kw):
    return pd.read_csv(f"{O}/{name}", **kw)

@st.cache_data
def js(name):
    with open(f"{O}/{name}") as f:
        return json.load(f)

st.title("⚡ EV Charging Demand & Battery Health Analytics")
st.caption("Demand forecasting · smart-charging simulation · battery survival analysis · customer segmentation · charger anomaly detection")

tabs = st.tabs(["📊 Overview", "🔮 Demand Forecast", "🔌 Smart Charging", "🔋 Battery Health",
                "👥 Customer Segments", "🛠️ Station Health", "📐 Statistical Evidence", "📘 Methodology"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    k = js("kpis.json")
    c = st.columns(6)
    c[0].metric("Sessions", f"{k['sessions']:,}")
    c[1].metric("Energy delivered", f"{k['energy_mwh']:,.0f} MWh")
    c[2].metric("Revenue", f"₹{k['revenue_cr']:,.2f} Cr")
    c[3].metric("Avg session", f"{k['avg_session_kwh']:.1f} kWh")
    c[4].metric("Interrupted sessions", f"{k['interruption_pct']:.1f}%")
    c[5].metric("Evening-peak share", f"{k['evening_peak_share_pct']:.1f}%")
    st.caption(f"{k['stations']} stations · {k['users']:,} users · {k['period']} · synthetic data")
    mo = rd("overview_monthly.csv.gz")
    st.plotly_chart(px.line(mo, x="month", y="energy_kwh", color="city", title="Monthly energy delivered (kWh)"))
    l, r = st.columns(2)
    hm = rd("heatmap.csv.gz")
    loc = l.selectbox("Location type", sorted(hm.location_type.unique()))
    p = hm[hm.location_type == loc].pivot(index="dow", columns="hour", values="avg_kwh")
    p.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    l.plotly_chart(px.imshow(p, aspect="auto", color_continuous_scale="Viridis",
                             title=f"{loc}: average kWh by weekday and hour"))
    ss = rd("station_summary.csv.gz")
    u = ss.groupby("charger_type").utilization_pct.mean().reset_index()
    r.plotly_chart(px.bar(u, x="charger_type", y="utilization_pct", title="Average charger utilization (%)"))
    st.subheader("Station league table")
    st.dataframe(ss.sort_values("energy_kwh", ascending=False))

# ------------------------------------------------------------------ Forecast
with tabs[1]:
    f = rd("forecast_test.csv.gz", parse_dates=["ts"])
    fs = js("forecast_summary.json")
    m = rd("forecast_metrics.csv")
    c = st.columns(4)
    c[0].metric("Hold-out period starts", fs["test_start"])
    c[1].metric("WAPE gain vs seasonal naive", f"{fs['improvement_vs_seasonal_naive_pct']:.1f}%")
    c[2].metric("80% interval coverage", f"{100 * fs['coverage_80']:.1f}%")
    c[3].metric("Avg interval width", f"{fs['avg_interval_width']:.1f} kWh")
    city = st.selectbox("City", sorted(f.city.unique()))
    days = st.select_slider("Window (days)", [3, 7, 14, 30, 60, 90], value=14)
    w = f[f.city == city].tail(24 * days)
    cap = st.slider("Site / feeder capacity (kWh per hour)", 0, int(w.p90.max() * 1.3) + 1,
                    int(w.p90.quantile(0.9)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.ts, y=w.p90, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=w.ts, y=w.p10, fill="tonexty", line=dict(width=0), name="80% interval",
                             fillcolor="rgba(99,110,250,0.25)"))
    fig.add_trace(go.Scatter(x=w.ts, y=w.actual, name="Actual", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=w.ts, y=w.forecast, name="Forecast", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=w.ts, y=w.naive_168, name="Seasonal naive", line=dict(dash="dot", width=1)))
    fig.add_hline(y=cap, line_dash="dash", line_color="red", annotation_text="capacity")
    fig.update_layout(title=f"{city}: day-ahead hourly demand forecast (unseen hold-out)", yaxis_title="kWh per hour")
    st.plotly_chart(fig)
    risk = w[w.p90 > cap]
    if len(risk):
        st.warning(f"⚠️ {len(risk)} of {len(w)} hours have a 90th-percentile forecast above the capacity line.")
    else:
        st.success("No hours exceed the capacity line at the 90th percentile.")
    l, r = st.columns(2)
    l.subheader("Model comparison (hold-out)")
    l.dataframe(m)
    l.caption("WAPE = sum of absolute errors / sum of actuals. Peak_MAE covers 18:00–21:00 only.")
    fi = rd("feature_importance.csv").head(12).sort_values("importance")
    r.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="LightGBM feature importance (gain %)"))
    st.subheader("Accuracy by city")
    st.dataframe(rd("forecast_metrics_by_city.csv"))

# ------------------------------------------------------------------ Smart charging
with tabs[2]:
    sc = rd("smart_charging_scenarios.csv")
    st.markdown("Flexible AC sessions (plugged in ≥2 h longer than needed) are shifted to the lowest-load hour inside "
                "their own plug-in window. Simulated on the last 90 days.")
    l, r = st.columns(2)
    l.plotly_chart(px.bar(sc, x="adoption_pct", y="peak_reduction_pct", text="peak_reduction_pct",
                          title="Network peak reduction (%) by adoption rate"))
    r.plotly_chart(px.bar(sc, x="adoption_pct", y="customer_cost_savings_pct", text="customer_cost_savings_pct",
                          title="Customer tariff savings (%) by adoption rate"))
    pr = rd("smart_charging_profile.csv").melt("hour", var_name="scenario", value_name="load_kwh")
    st.plotly_chart(px.line(pr, x="hour", y="load_kwh", color="scenario", title="Average network load by hour of day"))
    st.dataframe(sc)

# ------------------------------------------------------------------ Battery
with tabs[3]:
    bs = js("battery_summary.json")
    c = st.columns(4)
    c[0].metric("Vehicles monitored", f"{bs['n_vehicles']:,}")
    c[1].metric("Reached end-of-life (<80% SoH)", f"{bs['n_events']:,}")
    c[2].metric("Cox concordance", f"{bs['cox_concordance']:.3f}")
    c[3].metric("Log-rank p (usage)", f"{bs['logrank_p_usage']:.2g}")
    soh = rd("soh_curves.csv")
    fig = go.Figure()
    for i, (u, d) in enumerate(soh.groupby("usage")):
        col = ["#636EFA", "#EF553B"][i % 2]
        fig.add_trace(go.Scatter(x=d.month_idx, y=d.p90, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=d.month_idx, y=d.p10, fill="tonexty", line=dict(width=0), showlegend=False,
                                 fillcolor="rgba(120,120,120,0.15)"))
        fig.add_trace(go.Scatter(x=d.month_idx, y=d["mean"], name=u, line=dict(color=col)))
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="end-of-life (80%)")
    fig.update_layout(title="State of health by months in service (mean, 10–90% band)", yaxis_title="SoH %")
    st.plotly_chart(fig)
    l, r = st.columns(2)
    km = rd("km_curves.csv")
    sp = l.selectbox("Compare by", sorted(km.split.unique()))
    l.plotly_chart(px.line(km[km.split == sp], x="month", y="survival", color="group",
                           title="Kaplan–Meier: probability battery stays above 80% SoH"))
    hr = rd("cox_hazard_ratios.csv")
    fig = go.Figure(go.Scatter(x=hr.hazard_ratio, y=hr.covariate, mode="markers",
                               error_x=dict(type="data", symmetric=False, array=hr.ci_high - hr.hazard_ratio,
                                            arrayminus=hr.hazard_ratio - hr.ci_low)))
    fig.add_vline(x=1, line_dash="dash")
    fig.update_layout(title="Cox hazard ratios (per 1 SD, 95% CI)", xaxis_type="log")
    r.plotly_chart(fig)
    st.subheader("Predictive-maintenance watchlist (predicted SoH in 6 months)")
    wl = rd("battery_watchlist.csv")
    risk = st.multiselect("Risk level", sorted(wl.risk.unique()), [x for x in wl.risk.unique() if x != "Healthy"])
    st.dataframe(wl[wl.risk.isin(risk)])
    st.subheader("6-month-ahead SoH model (unseen vehicles)")
    st.dataframe(rd("battery_model_metrics.csv"))
    if os.path.exists(f"{O}/battery_pred_vs_actual.png"):
        st.image(f"{O}/battery_pred_vs_actual.png", width=420)

# ------------------------------------------------------------------ Segments
with tabs[4]:
    prof, scat, sil = rd("user_segment_profile.csv"), rd("user_segment_scatter.csv.gz"), rd("silhouette.csv")
    l, r = st.columns([2, 1])
    l.plotly_chart(px.scatter(scat, x="pc1", y="pc2", color="segment", opacity=0.6,
                              title="User segments (PCA projection of behaviour features)"))
    r.plotly_chart(px.line(sil, x="k", y="silhouette", markers=True, title="Silhouette score by k"))
    st.subheader("Segment profiles")
    st.dataframe(prof)
    st.plotly_chart(px.bar(prof, x="segment", y="energy_share_pct", title="Share of total energy by segment (%)"))

# ------------------------------------------------------------------ Station health
with tabs[5]:
    am = js("anomaly_metrics.json")
    c = st.columns(4)
    c[0].metric("Precision", f"{am['precision']:.2f}")
    c[1].metric("Recall", f"{am['recall']:.2f}")
    c[2].metric("Outage recall", f"{am['outage_recall']:.2f}")
    c[3].metric("Degraded-charger recall", f"{am['degraded_recall']:.2f}")
    st.caption("Isolation Forest is unsupervised. It is scored here against a hidden fault log from the simulator.")
    hh = rd("station_daily_health.csv.gz", parse_dates=["date"])
    sid = st.selectbox("Station", sorted(hh.station_id.unique()))
    d = hh[hh.station_id == sid]
    fig = px.line(d, x="date", y="sessions", title=f"{sid}: daily sessions with flagged days")
    fl = d[d.flag == 1]
    fig.add_trace(go.Scatter(x=fl.date, y=fl.sessions, mode="markers", name="Flagged",
                             marker=dict(color="red", size=8), text=fl.anomaly_type))
    st.plotly_chart(fig)
    st.subheader("All flagged station-days")
    st.dataframe(hh[hh.flag == 1].sort_values("anomaly_score", ascending=False))

# ------------------------------------------------------------------ Stats
with tabs[6]:
    st.subheader("Hypothesis tests (Benjamini–Hochberg FDR-adjusted)")
    st.dataframe(rd("stat_tests.csv"))
    st.subheader("Time-series diagnostics")
    st.caption("ADF p-value above 0.05 on levels means non-stationary (growth trend). Differencing removes it.")
    st.dataframe(rd("ts_diagnostics.csv"))
    if os.path.exists(f"{O}/stl_decomposition.png"):
        st.image(f"{O}/stl_decomposition.png")
    st.subheader("What drives battery fade? (OLS, robust HC3 errors)")
    ss_ = js("stats_summary.json")
    st.caption(f"R² = {ss_['battery_ols_r2']:.3f} · n = {ss_['n_vehicles']:,} vehicles")
    st.dataframe(rd("battery_ols.csv"))

# ------------------------------------------------------------------ Methodology
with tabs[7]:
    st.markdown("""
**Data.** Synthetic, simulated with realistic structure: daily and weekly cycles by location type, weather, festivals,
EV adoption growth, charger faults, and a physics-inspired battery degradation model (calendar plus cycle fade,
with temperature, fast-charging and depth-of-discharge stress). Replace with real data
(e.g. Caltech ACN-Data, NASA battery aging data) by keeping the same column names.

**Forecasting.** Direct day-ahead hourly forecast per city. Every feature (lags ≥ 24 h, rolling means, weather,
calendar) is available at 00:00 of the forecast day. Hyper-parameters are tuned with rolling-origin validation.
The last 90 days are held out. Benchmarks: naive, seasonal naive, ridge. Intervals come from quantile LightGBM.

**Smart charging.** Greedy valley-filling of flexible AC sessions inside each session's own plug-in window,
simulated at 0–100% adoption. Savings use a time-of-use tariff.

**Battery.** Kaplan–Meier and Cox PH handle censored lifetimes. A vehicle-grouped split gives the 6-month SoH
forecast, so no vehicle appears in both train and test.

**Anomalies.** Isolation Forest on station-relative features, evaluated against injected faults.

**Limitations.** Synthetic data. Demand is attributed to the session start hour. Cost and tariff values are assumptions.
""")