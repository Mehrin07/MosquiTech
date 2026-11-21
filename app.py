"""MosquiTech - AI-Powered Dengue Early Warning & Urban Risk Intelligence.   Run:  streamlit run app.py

All heavy work is cached: models are loaded once (st.cache_resource), features/predictions are computed
once (st.cache_data). Interacting with widgets only filters pre-computed tables. No training happens here
unless the models folder is empty on the very first launch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.data_loader import DataError
from src.prediction import (ModelsMissingError, driver_percentiles, horizon_table, load_models,
                            models_exist, predict_all, prepare_data, summarize_risk)
from src.explain import describe, explain_instance, global_group_importance, make_background
from src.intelligence import SUGGESTED_QUESTIONS, CallBudget, ask, build_context, evidence_brief
from src.llm_client import ENV_KEY, GeminiClient, LLMError, LLMUnavailable
from src.recommendations import DISCLAIMER, recommend
from src.validation import calibration_table, confusion, evidence_status, monthly_tier_table, seasonal_history, tier_summary

st.set_page_config(page_title="MosquiTech", page_icon="🦟", layout="wide", initial_sidebar_state="expanded")

def _streamlit_wide_kwargs() -> dict:
    """Newer Streamlit (>=1.50) uses width='stretch'; older versions use use_container_width=True."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


WIDE = _streamlit_wide_kwargs()
TIER_COLORS = C.RISK_COLORS
HEAT_SCALE = [[0.0, "#1f6f4a"], [0.25, "#2ecc71"], [0.5, "#f4d03f"], [0.75, "#e67e22"], [1.0, "#e74c3c"]]

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; max-width: 1400px;}
      .mt-hero {background: linear-gradient(120deg,#0f2a4a 0%,#0b3d3a 100%); border:1px solid #1e3a5f;
                border-radius:16px; padding:18px 26px; margin-bottom:14px;}
      .mt-hero h1 {margin:0; font-size:2.0rem; letter-spacing:.5px; color:#f1f5f9;}
      .mt-hero p {margin:2px 0 0 0; color:#7dd3c8; font-size:1.0rem; font-style:italic;}
      .mt-card {background:#111c33; border:1px solid #22345a; border-radius:14px; padding:14px 16px; height:100%;}
      .mt-card .lbl {color:#94a3b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.8px;}
      .mt-card .val {color:#f8fafc; font-size:1.9rem; font-weight:700; line-height:1.15;}
      .mt-card .sub {color:#94a3b8; font-size:.8rem;}
      .badge {display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; color:#0b1426;}
      .mt-note {background:#10223f; border-left:4px solid #2dd4bf; padding:10px 14px; border-radius:6px;
                color:#cbd5e1; font-size:.9rem;}
      .mt-rec {background:#111c33; border:1px solid #22345a; border-left:5px solid #2dd4bf; border-radius:10px;
               padding:10px 14px; margin-bottom:8px;}
      .mt-rec small {color:#94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------------------
# Cached loaders  (no work is repeated when the user changes a widget)
# --------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def get_models() -> dict:
    if models_exist():
        try:
            return load_models()
        except Exception:
            pass  # e.g. models saved with a different scikit-learn version -> retrain below
    from src import train_model
    with st.spinner("Trained models missing or incompatible - training once (about 3-4 minutes)..."):
        train_model.main(quiet=True)
    return load_models()


@st.cache_data(show_spinner="Building ward-level rainfall-lag features...")
def get_features() -> pd.DataFrame:
    return prepare_data()


@st.cache_data(show_spinner="Running model inference for all wards...")
def get_predictions() -> pd.DataFrame:
    return predict_all(get_features(), get_models())


@st.cache_data
def get_reports() -> dict:
    out = {}
    for name in ["model_comparison", "test_metrics", "feature_importance", "validation_predictions", "test_predictions"]:
        path = C.REPORTS_DIR / f"{name}.csv"
        out[name] = pd.read_csv(path) if path.exists() else None
    for name in ["target_date"]:
        for k in ("validation_predictions", "test_predictions"):
            if out[k] is not None:
                out[k][name] = pd.to_datetime(out[k][name])
    return out


@st.cache_data
def get_lag_correlation() -> pd.DataFrame:
    f = get_features().sort_values(["ward_id", "date"])
    g = f.groupby("ward_id")["rainfall_mm"]
    rows = [{"lag_weeks": k, "correlation": float(g.shift(k).corr(f["dengue_cases"]))} for k in range(0, 9)]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_background() -> pd.DataFrame:
    return make_background(get_features())


@st.cache_data(show_spinner="Computing Shapley explanation...")
def get_explanation(ward: str, asof_str: str, horizon: int) -> dict:
    """Prediction-level explanation for one ward-week (cached per selection)."""
    f = get_features()
    row = f[(f["ward_id"] == ward) & (f["date"] == pd.Timestamp(asof_str))]
    if row.empty:
        raise ValueError("No feature data for this ward and week.")
    return explain_instance(get_models()["bundles"][horizon]["model"], row.iloc[0], get_background())


@st.cache_data(show_spinner="Computing global driver importance...")
def get_global_groups(horizon: int) -> pd.DataFrame:
    f = get_features()
    test_rows = f[f["date"] > pd.Timestamp(C.VAL_END)].dropna(subset=C.FEATURE_COLS)
    sample = test_rows.sample(n=min(60, len(test_rows)), random_state=C.RANDOM_STATE)
    return global_group_importance(get_models()["bundles"][horizon]["model"], sample, get_background())


@st.cache_data(show_spinner=False)
def get_context(ward: str, asof_str: str, horizon: int) -> dict:
    """Grounded context for MosquiTech Intelligence (real model outputs + dataset values only)."""
    return build_context(feats=get_features(), preds=get_predictions(), models=get_models(), reports=get_reports(),
                         background=get_background(), asof=pd.Timestamp(asof_str), ward=ward, horizon=horizon,
                         explanation=get_explanation(ward, asof_str, horizon))


@st.cache_data(show_spinner=False)
def get_peak_week() -> pd.Timestamp:
    """Held-out-period forecast week with the most HIGH/CRITICAL wards (3-week horizon)."""
    p = get_predictions()
    p = p[p["date"] > pd.Timestamp(C.VAL_END)]
    counts = p["risk_h3"].isin(["HIGH", "CRITICAL"]).groupby(p["date"]).sum()
    return pd.Timestamp(counts.idxmax())


def _set_asof(date) -> None:
    st.session_state["asof"] = pd.Timestamp(date)


# --------------------------------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------------------------------
def card(label: str, value, sub: str = "", color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return f'<div class="mt-card"><div class="lbl">{label}</div><div class="val"{style}>{value}</div><div class="sub">{sub}</div></div>'


def badge(tier: str) -> str:
    return f'<span class="badge" style="background:{TIER_COLORS[tier]}">{tier}</span>'


def style_fig(fig, height: int = 380):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(17,28,51,0.6)", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def show(fig, height: int = 380):
    st.plotly_chart(style_fig(fig, height), **WIDE)


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d %b %Y")


def tier_of(value: float, thresholds) -> str:
    from src.risk_classification import classify
    return classify(value, thresholds)


def ward_selector(wards: pd.DataFrame, key: str, default_city: str | None = None):
    cities = sorted(wards["city"].unique())
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", cities, index=cities.index(default_city) if default_city in cities else 0, key=f"{key}_city")
    options = sorted(wards.loc[wards["city"] == city, "ward_id"])
    if not options:
        st.warning("No wards available for this city.")
        return city, None
    ward = c2.selectbox("Ward", options, key=f"{key}_ward")
    return city, ward


def get_pred_row(preds: pd.DataFrame, ward: str, asof) -> pd.Series | None:
    r = preds[(preds["ward_id"] == ward) & (preds["date"] == asof)]
    return None if r.empty else r.iloc[0]


# --------------------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------------------
def season_panel(feats, asof, hz, thresholds, elevated):
    """Evidence for whether the current tier mix is plausible for the time of year (dataset facts only)."""
    target = asof + pd.Timedelta(weeks=hz)
    hist = seasonal_history(feats, target, thresholds)
    st.markdown("#### Is this forecast plausible for the time of year?")
    a, b = st.columns([2, 3])
    with a:
        if hist:
            st.markdown(
                f"**Observed history:** in the same calendar weeks ({hist['calendar_window']}), {hist['share_actual_LOW_pct']:.0f}% of "
                f"ward-weeks were LOW and {hist['share_actual_HIGH_or_CRITICAL_pct']:.0f}% were HIGH/CRITICAL "
                f"({hist['ward_weeks']:,} ward-weeks over {hist['years']} years; mean {hist['mean_cases_per_ward_week']:.1f}, max {hist['max_cases_in_a_ward_week']} cases/week).")
            if elevated == 0 and hist["share_actual_LOW_pct"] >= 90:
                st.success("An all-LOW outlook is consistent with history for this time of year; it is not a malfunction. "
                           "See Model Performance → risk-tier validation for out-of-sample accuracy. "
                           "Use the sidebar button to jump to the 2025 peak week.")
            elif elevated == 0:
                st.warning("The model forecasts no HIGH/CRITICAL wards although this time of year was often active historically. "
                           "Treat with caution and check recent weather and case inputs.")
            else:
                st.info(f"{elevated} ward(s) are HIGH/CRITICAL, which fits a season in which {hist['share_actual_HIGH_or_CRITICAL_pct']:.0f}% of ward-weeks were HIGH/CRITICAL historically.")
        st.caption(f"Evidence status of this forecast: {evidence_status(target, feats['date'].max())}.")
    with b:
        m = feats.assign(month=feats["date"].dt.month).groupby("month")["dengue_cases"].mean().reindex(range(1, 13)).fillna(0)
        fig = go.Figure(go.Bar(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], y=m.values,
                               marker_color=["#2dd4bf" if i + 1 == target.month else "#475569" for i in range(12)]))
        fig.update_layout(title="Observed mean weekly cases per ward by month, 2016-2025 (target month highlighted)", yaxis_title="cases / ward / week")
        show(fig, 260)


def section_shap(feats, ward, asof, hz):
    """Shapley prediction-level explanation + global driver importance."""
    st.markdown("#### Prediction-level explanation (Shapley values)")
    try:
        ex = get_explanation(ward, str(pd.Timestamp(asof).date()), hz)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.markdown('<div class="mt-note">' + " ".join(describe(ex, hz)) + "</div>", unsafe_allow_html=True)
    g = pd.Series(ex["groups"]).sort_values()
    col = ["#e67e22" if v > 0 else "#38bdf8" for v in g.values]
    a, b = st.columns(2)
    fg = go.Figure(go.Bar(y=g.index, x=g.values, orientation="h", marker_color=col, text=[f"{v:+.2f}" for v in g.values], textposition="outside"))
    fg.update_layout(title="Contribution by driver group (cases/week vs typical ward-week)", xaxis_title="cases / week")
    with a:
        show(fg, 360)
    top = ex["features"].head(10).iloc[::-1]
    ff = go.Figure(go.Bar(y=[f"{r.label} = {r.value:,.1f}" for r in top.itertuples()], x=top["contribution"], orientation="h",
                          marker_color=["#e67e22" if v > 0 else "#38bdf8" for v in top["contribution"]]))
    ff.update_layout(title="Top 10 individual features for this prediction", xaxis_title="cases / week")
    with b:
        show(ff, 360)
    order = list(ex["groups"])
    wf = go.Figure(go.Waterfall(x=["Typical ward-week"] + order + ["This prediction"], measure=["absolute"] + ["relative"] * len(order) + ["total"],
                                y=[ex["baseline"]] + [ex["groups"][k] for k in order] + [0],
                                increasing=dict(marker=dict(color="#e67e22")), decreasing=dict(marker=dict(color="#38bdf8")),
                                totals=dict(marker=dict(color="#2dd4bf"))))
    wf.update_layout(title="From a typical ward-week to this ward's prediction", yaxis_title="predicted cases / week")
    show(wf, 360)
    gl = get_global_groups(hz)
    fgl = go.Figure(go.Bar(y=gl["group"], x=gl["mean_abs_contribution"], orientation="h", marker_color="#2dd4bf",
                           text=[f"{v:.0f}%" for v in gl["share_pct"]], textposition="outside"))
    fgl.update_layout(title="Across 60 sampled 2025 ward-weeks: average size of each driver group's contribution", xaxis_title="mean |contribution| (cases/week)",
                      yaxis=dict(autorange="reversed"))
    show(fgl, 320)
    st.caption("Shapley values split the model's prediction among its inputs (baseline + contributions = prediction; checked in tests). "
               "They show how the MODEL uses inputs, not what causes dengue. Inputs that move together (e.g. rainfall lags) share credit, "
               "and the method evaluates mixed combinations of conditions that may not occur in reality.")


def risk_validation_section(reports, thresholds):
    vp, tp = reports["validation_predictions"], reports["test_predictions"]
    if vp is None or tp is None:
        return
    st.markdown("#### Risk-tier validation (out-of-sample)")
    st.caption(f"Tiers use training-period cut-points ({thresholds[0]:.0f} / {thresholds[1]:.0f} / {thresholds[2]:.0f} cases/week). "
               "They were not tuned on these data. Below: do the forecast tiers match what actually happened?")
    rows = []
    for name, d in (("Validation 2023-24", vp), ("Test 2025", tp)):
        s = tier_summary(d)
        rows.append({"Period": name, "Ward-weeks": f"{s['ward_weeks']:,}", "Exact tier": f"{s['exact_tier_accuracy']*100:.1f}%",
                     "Within one tier": f"{s['within_one_tier']*100:.1f}%", "HIGH/CRIT recall": f"{s['high_or_critical_recall']*100:.1f}%",
                     "HIGH/CRIT precision": f"{s['high_or_critical_precision']*100:.1f}%",
                     "Missed alarms": f"{s['missed_alarms_pred_low_actual_high_or_critical']} of {s['actual_high_or_critical']}",
                     "False alarms": s["false_alarms_pred_high_or_critical_actual_low"]})
    st.dataframe(pd.DataFrame(rows), **WIDE, hide_index=True)
    fig = go.Figure()
    for name, d, colr in (("Validation 2023-24", vp, "#7dd3fc"), ("Test 2025", tp, "#f4d03f")):
        c = calibration_table(d)
        fig.add_trace(go.Bar(x=c["tier"], y=c["mean_actual"], name=f"Actual mean cases - {name}", marker_color=colr))
    fig.update_layout(barmode="group", title="Calibration: mean ACTUAL cases for wards forecast in each tier", yaxis_title="actual cases / ward / week")
    a, b = st.columns(2)
    with a:
        show(fig, 340)
    mt = monthly_tier_table(pd.concat([vp, tp])[lambda d: d["horizon"] == 3])
    fm = go.Figure()
    fm.add_trace(go.Bar(x=mt["month"], y=mt["actual_LOW_pct"], name="Actual LOW %", marker_color="#475569"))
    fm.add_trace(go.Bar(x=mt["month"], y=mt["predicted_LOW_pct"], name="Forecast LOW %", marker_color="#2dd4bf"))
    fm.update_layout(barmode="group", title="Share of ward-weeks that are LOW, by month (3-week forecasts)", xaxis=dict(title="Month", dtick=1), yaxis_title="%")
    with b:
        show(fm, 340)
    st.caption("Dry-season months are almost entirely LOW in the actual data too, so all-LOW forecasts in those months are expected. "
               "On the 2025 test year the CRITICAL tier over-forecast (forecast mean above actual mean): more false alarms than misses.")


def tab_overview(feats, preds, asof, thresholds, latest_obs):
    st.subheader("Executive Overview")
    hz = st.radio("Forecast horizon", list(C.HORIZONS), index=1, horizontal=True, key="ov_h",
                  format_func=lambda x: f"{x} weeks ahead")
    now = preds[preds["date"] == asof]
    counts = summarize_risk(now, hz)
    n_cities, n_wards = feats["city"].nunique(), feats["ward_id"].nunique()

    cols = st.columns(6)
    cols[0].markdown(card("Monitored cities", n_cities), unsafe_allow_html=True)
    cols[1].markdown(card("Monitored wards", n_wards), unsafe_allow_html=True)
    cols[2].markdown(card("High-risk wards", counts["HIGH"], f"{hz}-week outlook", TIER_COLORS["HIGH"]), unsafe_allow_html=True)
    cols[3].markdown(card("Critical-risk wards", counts["CRITICAL"], f"{hz}-week outlook", TIER_COLORS["CRITICAL"]), unsafe_allow_html=True)
    cols[4].markdown(card("Latest observation", fmt_date(latest_obs), "last week in dataset"), unsafe_allow_html=True)
    cols[5].markdown(card("Forecast issued", fmt_date(asof), f"for week of {fmt_date(asof + pd.Timedelta(weeks=hz))}"), unsafe_allow_html=True)

    elevated = counts["HIGH"] + counts["CRITICAL"]
    total_pred = now[f"pred_h{hz}"].sum()
    st.markdown(
        f'<div class="mt-note"><b>Overall risk summary:</b> For the week of {fmt_date(asof + pd.Timedelta(weeks=hz))}, the model expects about '
        f'<b>{total_pred:.0f}</b> dengue cases across all {n_wards} wards. <b>{elevated}</b> ward(s) are HIGH or CRITICAL, '
        f'{counts["MEDIUM"]} MEDIUM and {counts["LOW"]} LOW.</div>',
        unsafe_allow_html=True,
    )
    season_panel(feats, asof, hz, thresholds, elevated)

    a, b = st.columns(2)
    allh = pd.DataFrame([{"Horizon": f"{h} wk", "Tier": t, "Wards": summarize_risk(now, h)[t]} for h in C.HORIZONS for t in C.RISK_LABELS])
    fig = go.Figure()
    for t in C.RISK_LABELS:
        d = allh[allh["Tier"] == t]
        fig.add_trace(go.Bar(x=d["Horizon"], y=d["Wards"], name=t, marker_color=TIER_COLORS[t]))
    fig.update_layout(barmode="stack", title="Wards by risk tier, all horizons", yaxis_title="Wards")
    with a:
        show(fig, 340)

    by_city = now.groupby(["city", f"risk_h{hz}"]).size().unstack(fill_value=0).reindex(columns=list(C.RISK_LABELS), fill_value=0)
    fig2 = go.Figure()
    for t in C.RISK_LABELS:
        fig2.add_trace(go.Bar(y=by_city.index, x=by_city[t], name=t, orientation="h", marker_color=TIER_COLORS[t]))
    fig2.update_layout(barmode="stack", title=f"Risk tiers by city ({hz}-week outlook)", xaxis_title="Wards")
    with b:
        show(fig2, 340)

    st.markdown("**Top 10 wards needing preventive attention**")
    top = now.sort_values(f"pred_h{hz}", ascending=False).head(10)
    st.dataframe(pd.DataFrame({
        "Ward": top["ward_id"], "City": top["city"], "Predicted cases/wk": top[f"pred_h{hz}"].round(1),
        "80% range": [f"{lo:.1f} - {hi:.1f}" for lo, hi in zip(top[f"lo_h{hz}"], top[f"hi_h{hz}"])],
        "Risk": top[f"risk_h{hz}"]}), **WIDE, hide_index=True)


def tab_prediction(feats, preds, wards, asof, thresholds):
    st.subheader("Dengue Prediction")
    city, ward = ward_selector(wards, "pred")
    hz = st.radio("Prediction horizon", list(C.HORIZONS), index=1, horizontal=True, key="pred_h",
                  format_func=lambda x: f"{x} weeks ahead")
    if ward is None:
        return
    row = get_pred_row(preds, ward, asof)
    if row is None:
        st.warning("This ward has no prediction for the selected forecast week (not enough history yet).")
        return
    ht = horizon_table(row)
    sel = ht[ht["horizon_weeks"] == hz].iloc[0]
    recent = feats[(feats["ward_id"] == ward) & (feats["date"] == asof)].iloc[0]

    c = st.columns(4)
    c[0].markdown(card("Predicted cases / week", f"{sel['predicted_cases']:.1f}", f"80% range {sel['lower']:.1f} - {sel['upper']:.1f}"), unsafe_allow_html=True)
    c[1].markdown(card("Risk category", badge(sel["risk"]), f"{hz}-week horizon"), unsafe_allow_html=True)
    c[2].markdown(card("Target week", fmt_date(sel["target_date"]), f"issued {fmt_date(asof)}"), unsafe_allow_html=True)
    c[3].markdown(card("Recent 4-wk average", f"{recent['cases_roll4']:.1f}", "observed cases / week"), unsafe_allow_html=True)

    hist = feats[(feats["ward_id"] == ward) & (feats["date"] <= asof)].tail(104)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["dengue_cases"], mode="lines", name="Observed cases", line=dict(color="#7dd3fc", width=2)))
    err_up = (ht["upper"] - ht["predicted_cases"]).clip(lower=0)
    err_dn = (ht["predicted_cases"] - ht["lower"]).clip(lower=0)
    fig.add_trace(go.Scatter(
        x=ht["target_date"], y=ht["predicted_cases"], mode="lines+markers", name="Forecast (80% interval)",
        line=dict(color="#94a3b8", dash="dot"),
        marker=dict(size=12, color=[TIER_COLORS[r] for r in ht["risk"]], line=dict(width=1, color="white")),
        error_y=dict(type="data", symmetric=False, array=err_up, arrayminus=err_dn, color="#94a3b8")))
    later = feats[(feats["ward_id"] == ward) & (feats["date"] > asof) & (feats["date"] <= asof + pd.Timedelta(weeks=max(C.HORIZONS)))]
    if not later.empty:
        fig.add_trace(go.Scatter(x=later["date"], y=later["dengue_cases"], mode="markers", name="Actual (observed later)",
                                 marker=dict(symbol="x", size=9, color="#f8fafc")))
    for name, t in zip(C.RISK_LABELS[1:], thresholds):
        fig.add_hline(y=t, line_dash="dash", line_color=TIER_COLORS[name], opacity=0.5,
                      annotation_text=f"{name} >= {t:.0f}", annotation_position="top left")
    fig.update_layout(title=f"{ward} - weekly dengue cases and forecast", yaxis_title="Cases per week")
    show(fig, 420)

    a, b = st.columns(2)
    fig2 = go.Figure(go.Bar(x=[f"{h} wk" for h in ht["horizon_weeks"]], y=ht["predicted_cases"],
                            marker_color=[TIER_COLORS[r] for r in ht["risk"]],
                            error_y=dict(type="data", symmetric=False, array=err_up, arrayminus=err_dn),
                            text=[f"{p:.1f}" for p in ht["predicted_cases"]], textposition="outside"))
    fig2.update_layout(title="Predictions across horizons", yaxis_title="Predicted cases / week")
    with a:
        show(fig2, 320)
    with b:
        st.markdown("**Horizon comparison**")
        st.dataframe(pd.DataFrame({
            "Horizon": [f"{h} weeks" for h in ht["horizon_weeks"]], "Target week": [fmt_date(d) for d in ht["target_date"]],
            "Predicted": ht["predicted_cases"].round(1), "Low": ht["lower"].round(1), "High": ht["upper"].round(1),
            "Risk": ht["risk"]}), **WIDE, hide_index=True)
        st.caption("Intervals are empirical 80% ranges built from validation-set errors; they are not guarantees.")


def tab_ward_risk(feats, preds, wards, asof, thresholds):
    st.subheader("Ward Risk Intelligence")
    c1, c2 = st.columns([1, 2])
    hz = c1.selectbox("Horizon", list(C.HORIZONS), index=1, key="wr_h", format_func=lambda x: f"{x} weeks ahead")
    cities = sorted(wards["city"].unique())
    chosen = c2.multiselect("Cities", cities, default=cities, key="wr_cities")
    now = preds[(preds["date"] == asof) & (preds["city"].isin(chosen))].sort_values(f"pred_h{hz}", ascending=False)
    if now.empty:
        st.warning("Select at least one city.")
        return

    n_top = st.slider("Wards to rank", 5, max(5, min(40, len(now))), min(15, len(now)), key="wr_n") if len(now) > 5 else len(now)
    top = now.head(n_top).iloc[::-1]
    fig = go.Figure(go.Bar(y=top["ward_id"], x=top[f"pred_h{hz}"], orientation="h",
                           marker_color=[TIER_COLORS[r] for r in top[f"risk_h{hz}"]],
                           error_x=dict(type="data", symmetric=False, array=(top[f"hi_h{hz}"] - top[f"pred_h{hz}"]).clip(lower=0),
                                        arrayminus=(top[f"pred_h{hz}"] - top[f"lo_h{hz}"]).clip(lower=0), color="#94a3b8"),
                           text=top[f"risk_h{hz}"], textposition="inside"))
    fig.update_layout(title=f"Wards ranked by predicted cases ({hz} weeks ahead)", xaxis_title="Predicted cases / week")
    show(fig, max(360, 24 * n_top + 100))

    st.markdown("#### Ward risk heatmap")
    st.caption("This dataset has no geographic coordinates or ward boundaries, so MosquiTech shows a ward-by-time heatmap "
               "instead of a map. No locations are invented.")
    order = list(now["ward_id"])
    a, b = st.columns([1, 2])
    mat = now.set_index("ward_id").loc[order, [f"pred_h{h}" for h in C.HORIZONS]]
    figh = go.Figure(go.Heatmap(z=mat.values, x=[f"{h} wk" for h in C.HORIZONS], y=order, colorscale=HEAT_SCALE,
                                zmin=0, zmax=max(thresholds[2] * 1.2, 1), colorbar=dict(title="cases/wk"),
                                hovertemplate="Ward %{y}<br>Horizon %{x}<br>%{z:.1f} cases/wk<extra></extra>"))
    figh.update_layout(title="Forecast by horizon", yaxis=dict(autorange="reversed"))
    with a:
        show(figh, max(420, 16 * len(order) + 100))

    start = asof - pd.Timedelta(weeks=15)
    hist = preds[(preds["date"] >= start) & (preds["date"] <= asof) & (preds["city"].isin(chosen))]
    pivot = hist.pivot(index="ward_id", columns="date", values=f"pred_h{hz}").reindex(order)
    figt = go.Figure(go.Heatmap(z=pivot.values, x=[fmt_date(d) for d in pivot.columns], y=list(pivot.index), colorscale=HEAT_SCALE,
                                zmin=0, zmax=max(thresholds[2] * 1.2, 1), colorbar=dict(title="cases/wk"),
                                hovertemplate="Ward %{y}<br>Issued %{x}<br>%{z:.1f} cases/wk<extra></extra>"))
    figt.update_layout(title=f"{hz}-week forecast issued each week (last 16 weeks)", yaxis=dict(autorange="reversed"))
    with b:
        show(figt, max(420, 16 * len(order) + 100))

    st.markdown("#### Compare wards")
    picks = st.multiselect("Choose 2-5 wards", sorted(now["ward_id"]), default=list(now["ward_id"].head(3)), max_selections=5, key="wr_cmp")
    if len(picks) >= 2:
        x, y = st.columns(2)
        fig_c = go.Figure()
        for w in picks:
            r = now[now["ward_id"] == w].iloc[0]
            fig_c.add_trace(go.Bar(x=[f"{h} wk" for h in C.HORIZONS], y=[r[f"pred_h{h}"] for h in C.HORIZONS], name=w))
        fig_c.update_layout(barmode="group", title="Predicted cases by horizon", yaxis_title="cases / week")
        with x:
            show(fig_c, 340)
        fig_l = go.Figure()
        for w in picks:
            hh = feats[(feats["ward_id"] == w) & (feats["date"] <= asof)].tail(52)
            fig_l.add_trace(go.Scatter(x=hh["date"], y=hh["dengue_cases"], mode="lines", name=w))
        fig_l.update_layout(title="Observed cases, last 52 weeks", yaxis_title="cases / week")
        with y:
            show(fig_l, 340)
    else:
        st.info("Pick at least two wards to compare.")


def tab_weather(feats, wards, asof):
    st.subheader("Weather Analytics")
    city, ward = ward_selector(wards, "wx")
    if ward is None:
        return
    w = feats[(feats["ward_id"] == ward) & (feats["date"] <= asof)]
    years = sorted(w["date"].dt.year.unique())
    lo, hi = st.select_slider("Years shown", options=years, value=(years[-2] if len(years) > 1 else years[0], years[-1]), key="wx_years")
    d = w[(w["date"].dt.year >= lo) & (w["date"].dt.year <= hi)]
    if d.empty:
        st.warning("No weather data in the selected range.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["date"], y=d["rainfall_mm"], name="Weekly rainfall (mm)", marker_color="#38bdf8"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["dengue_cases"], name="Dengue cases (right axis)", yaxis="y2", line=dict(color="#f87171")))
    fig.update_layout(title=f"{ward}: rainfall history vs dengue cases", yaxis_title="Rainfall (mm)",
                      yaxis2=dict(title="Cases", overlaying="y", side="right", showgrid=False))
    show(fig, 360)

    a, b = st.columns(2)
    ft = go.Figure()
    ft.add_trace(go.Scatter(x=d["date"], y=d["max_temp_c"], name="Max", line=dict(color="#fb923c", width=1)))
    ft.add_trace(go.Scatter(x=d["date"], y=d["avg_temp_c"], name="Average", line=dict(color="#fde047", width=2)))
    ft.add_trace(go.Scatter(x=d["date"], y=d["min_temp_c"], name="Min", line=dict(color="#7dd3fc", width=1)))
    ft.update_layout(title="Weekly temperature (deg C)")
    with a:
        show(ft, 320)
    fh = go.Figure()
    fh.add_trace(go.Scatter(x=d["date"], y=d["relative_humidity_pct"], name="Weekly", line=dict(color="#a78bfa", width=1)))
    fh.add_trace(go.Scatter(x=d["date"], y=d["humid_roll4"], name="4-week mean", line=dict(color="#f0abfc", width=2)))
    fh.update_layout(title="Relative humidity (%)")
    with b:
        show(fh, 320)

    fc = go.Figure()
    fc.add_trace(go.Scatter(x=d["date"], y=d["rain_cum2"], name="2-week cumulative", line=dict(color="#38bdf8")))
    fc.add_trace(go.Scatter(x=d["date"], y=d["rain_cum4"], name="4-week cumulative", line=dict(color="#2dd4bf", width=2)))
    fc.update_layout(title="Cumulative rainfall (mm)")
    st.plotly_chart(style_fig(fc, 320), **WIDE)

    st.markdown("#### Historical weather patterns")
    cw = feats[feats["city"] == city].copy()
    cw["month"] = cw["date"].dt.month
    m = cw.groupby("month").agg(rain=("rainfall_mm", "mean"), temp=("avg_temp_c", "mean"), cases=("dengue_cases", "mean")).reset_index()
    x, y = st.columns(2)
    fm = go.Figure()
    fm.add_trace(go.Bar(x=m["month"], y=m["rain"], name="Mean weekly rainfall (mm)", marker_color="#38bdf8"))
    fm.add_trace(go.Scatter(x=m["month"], y=m["cases"], name="Mean weekly cases per ward (right)", yaxis="y2", line=dict(color="#f87171", width=3)))
    fm.update_layout(title=f"{city}: seasonal pattern (2016-2025)", xaxis=dict(title="Month", dtick=1),
                     yaxis_title="Rainfall (mm)", yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Cases"))
    with x:
        show(fm, 340)
    lag = get_lag_correlation()
    fl = go.Figure(go.Bar(x=lag["lag_weeks"], y=lag["correlation"], marker_color=["#2dd4bf" if v == lag["correlation"].max() else "#475569" for v in lag["correlation"]]))
    fl.update_layout(title="Correlation: rainfall k weeks ago vs dengue cases (all wards)", xaxis=dict(title="Lag (weeks)", dtick=1), yaxis_title="Pearson r")
    with y:
        show(fl, 340)
    best = lag.loc[lag["correlation"].idxmax()]
    st.caption(f"Exploratory view across all wards: correlation peaks at a lag of {int(best['lag_weeks'])} weeks (r = {best['correlation']:.2f}), "
               "which is consistent with the 2-4 week breeding-and-incubation delay. Correlation does not prove causation.")


def tab_explain(feats, preds, wards, asof, models):
    st.subheader("Risk Explanation")
    hz = st.radio("Horizon", list(C.HORIZONS), index=1, horizontal=True, key="ex_h", format_func=lambda x: f"{x} weeks ahead")
    bundle = models["bundles"][hz]
    imp = pd.DataFrame(bundle["importance"]).sort_values("share_pct", ascending=True).tail(12)
    fig = go.Figure(go.Bar(y=imp["label"], x=imp["share_pct"], orientation="h", marker_color="#2dd4bf"))
    fig.update_layout(title=f"What the {hz}-week model relies on most (permutation importance, validation data)", xaxis_title="Share of total importance (%)")
    show(fig, 440)
    st.markdown(f'<div class="mt-note">Model: <b>{bundle["model_name"]}</b>. Importance = how much prediction error rises when a feature is '
                'randomly shuffled on held-out validation data. It describes what the model uses, <b>not what causes dengue</b>. '
                'Correlated features (e.g. rainfall lags and cumulative rainfall) share credit, so individual bars should be read as approximate.</div>',
                unsafe_allow_html=True)

    st.markdown("#### Drivers for a specific ward")
    city, ward = ward_selector(wards, "ex")
    if ward is None:
        return
    row = feats[(feats["ward_id"] == ward) & (feats["date"] == asof)]
    if row.empty:
        st.warning("No feature data for this ward and week.")
        return
    row = row.iloc[0]
    ref = feats[feats["date"] <= pd.Timestamp(C.TRAIN_END)]
    pct = driver_percentiles(row, ref)
    labels = {"rain_cum4": "4-wk cumulative rainfall", "rain_cum2": "2-wk cumulative rainfall", "rain_lag3": "Rainfall 3 wks ago",
              "temp_roll4": "4-wk mean temperature", "humid_roll4": "4-wk mean humidity", "cases_roll4": "Recent dengue cases (4-wk mean)",
              "population_density_sqkm": "Population density", "water_logging_risk_index": "Waterlogging risk", "sanitation_score": "Sanitation score"}
    vals = {"rain_cum4": f"{row['rain_cum4']:.0f} mm", "rain_cum2": f"{row['rain_cum2']:.0f} mm", "rain_lag3": f"{row['rain_lag3']:.0f} mm",
            "temp_roll4": f"{row['temp_roll4']:.1f} C", "humid_roll4": f"{row['humid_roll4']:.0f} %", "cases_roll4": f"{row['cases_roll4']:.1f}",
            "population_density_sqkm": f"{row['population_density_sqkm']:,.0f}/sq km", "water_logging_risk_index": f"{row['water_logging_risk_index']:.2f}",
            "sanitation_score": f"{row['sanitation_score']:.2f}"}
    keys = list(labels)
    fig2 = go.Figure(go.Bar(y=[f"{labels[k]} ({vals[k]})" for k in keys], x=[pct[k] for k in keys], orientation="h",
                            marker_color=["#e67e22" if pct[k] >= 75 else "#475569" for k in keys]))
    fig2.add_vline(x=75, line_dash="dash", line_color="#94a3b8")
    fig2.update_layout(title=f"{ward}: where current conditions sit versus historical (training-period) values",
                       xaxis=dict(title="Percentile (100 = highest ever seen in training data)", range=[0, 100]), yaxis=dict(autorange="reversed"))
    show(fig2, 420)
    st.caption("Orange bars are in the top quartile of historical values. This is a descriptive comparison to history, not proof of cause. "
               "For sanitation score a LOW percentile is the concerning direction.")
    section_shap(feats, ward, asof, hz)


def tab_performance(models, reports, thresholds):
    st.subheader("Model Performance")
    tm, mc = reports["test_metrics"], reports["model_comparison"]
    if tm is None or mc is None:
        st.info("Evaluation reports not found. Run `python -m src.train_model` to generate them.")
        return
    split = models["metadata"]["split"]
    st.markdown(f'<div class="mt-note"><b>Chronological split</b> (by the week being predicted): train up to {split["train_end"]}, '
                f'validation up to {split["val_end"]}, test = the year after. Models were chosen on validation only; the test year was scored once.</div>',
                unsafe_allow_html=True)

    st.markdown("#### Test-year accuracy (untouched data)")
    st.dataframe(pd.DataFrame({
        "Horizon": [f"{h} weeks" for h in tm["horizon"]], "Selected model": tm["model"], "MAE": tm["MAE"].round(3),
        "RMSE": tm["RMSE"].round(3), "R2": tm["R2"].round(3), "Baseline MAE": tm["baseline_MAE"].round(3),
        "Baseline RMSE": tm["baseline_RMSE"].round(3), "Baseline R2": tm["baseline_R2"].round(3),
        "80% interval coverage": (tm["interval_coverage"] * 100).round(1).astype(str) + "%",
        "Risk-tier accuracy": (tm["tier_accuracy"] * 100).round(1).astype(str) + "%",
        "Within one tier": (tm["within_one_tier"] * 100).round(1).astype(str) + "%"}), **WIDE, hide_index=True)
    st.caption("Baseline = 'the next weeks will look like the last 4 weeks'. A useful model must beat it.")

    st.markdown("#### Model comparison (validation set)")
    val = mc[mc["split"] == "validation"]
    a, b = st.columns(2)
    for col, metric in ((a, "MAE"), (b, "RMSE")):
        fig = go.Figure()
        for h in C.HORIZONS:
            d = val[val["horizon"] == h]
            fig.add_trace(go.Bar(x=d["model"], y=d[metric], name=f"{h} wk"))
        fig.update_layout(barmode="group", title=f"Validation {metric} by model (lower is better)", xaxis=dict(tickangle=-30))
        with col:
            show(fig, 400)
    st.dataframe(val.assign(selected=np.where(val["selected"], "yes", "")).round(3)[["horizon", "model", "MAE", "RMSE", "R2", "selected"]],
                 **WIDE, hide_index=True)

    st.markdown("#### Chronological validation error dashboard")
    vp, tp = reports["validation_predictions"], reports["test_predictions"]
    if vp is not None and tp is not None:
        e = pd.concat([vp, tp])
        e["err"] = e["predicted"] - e["actual"]
        e["month"] = e["target_date"].dt.to_period("M").dt.to_timestamp()
        g = e.groupby(["horizon", "month"]).agg(MAE=("err", lambda s: s.abs().mean()), RMSE=("err", lambda s: float(np.sqrt((s ** 2).mean())))).reset_index()
        metric = st.radio("Error metric", ["MAE", "RMSE"], horizontal=True, key="pf_metric")
        fig = go.Figure()
        for h in C.HORIZONS:
            d = g[g["horizon"] == h]
            fig.add_trace(go.Scatter(x=d["month"], y=d[metric], mode="lines+markers", name=f"{h} wk"))
        fig.add_shape(type="line", x0=str(pd.Timestamp(split["val_end"]) + pd.Timedelta(days=1))[:10], x1=str(pd.Timestamp(split["val_end"]) + pd.Timedelta(days=1))[:10],
                      y0=0, y1=1, yref="paper", line=dict(color="#94a3b8", dash="dash"))
        fig.update_layout(title=f"Monthly {metric} by target month (left of dashed line: validation, right: test)", yaxis_title=f"{metric} (cases/wk per ward)")
        show(fig, 380)

        hz = st.radio("Horizon for detail", list(C.HORIZONS), index=1, horizontal=True, key="pf_h", format_func=lambda x: f"{x} weeks")
        t = tp[tp["horizon"] == hz]
        agg = t.groupby("target_date").agg(actual=("actual", "sum"), predicted=("predicted", "sum"), lower=("lower", "sum"), upper=("upper", "sum")).reset_index()
        x, y = st.columns([3, 2])
        fa = go.Figure()
        fa.add_trace(go.Scatter(x=agg["target_date"], y=agg["actual"], name="Actual", line=dict(color="#f8fafc")))
        fa.add_trace(go.Scatter(x=agg["target_date"], y=agg["predicted"], name="Predicted", line=dict(color="#2dd4bf")))
        fa.update_layout(title=f"Test year, all wards combined ({hz}-week forecasts)", yaxis_title="Cases per week (all wards)")
        with x:
            show(fa, 360)
        order = list(C.RISK_LABELS)
        cm = pd.crosstab(t["actual_tier"], t["predicted_tier"]).reindex(index=order, columns=order, fill_value=0)
        fcm = go.Figure(go.Heatmap(z=cm.values, x=order, y=order, colorscale="Teal", text=cm.values, texttemplate="%{text}", showscale=False))
        fcm.update_layout(title="Risk tiers: actual (rows) vs predicted (cols)", xaxis_title="Predicted tier", yaxis_title="Actual tier", yaxis=dict(autorange="reversed"))
        with y:
            show(fcm, 360)

    risk_validation_section(reports, thresholds)
    val_r2 = val[val["selected"]]["R2"].mean()
    test_r2 = tm["R2"].mean()
    st.markdown("#### Limitations")
    st.markdown(
        f"- **Honest accuracy:** mean R2 of selected models is {val_r2:.2f} on validation and {test_r2:.2f} on the test year; expect real-world performance to look more like the test year.\n"
        "- **Accuracy falls with horizon** because more can change between the forecast date and the target week.\n"
        "- **Expected values, not worst cases:** the models predict average counts, so they tend to under-predict sharp peaks, and CRITICAL is rarely triggered by point forecasts.\n"
        f"- **Risk tiers are relative:** cut-points ({thresholds[0]:.0f}, {thresholds[1]:.0f}, {thresholds[2]:.0f} cases/week) come from the training-period 70th/85th/95th percentiles; they are not clinical thresholds.\n"
        "- **Intervals are empirical** (from validation errors) with ~80% nominal coverage; they are not guarantees.\n"
        "- **Importance is not causation:** feature importance shows what the model uses, not what causes outbreaks.\n"
        "- **Scope:** one dataset (60 wards, 5 cities, 2016-2025); no mobility, serotype, intervention or geographic data; no map because no coordinates were supplied.")


def tab_recommendations(feats, preds, wards, asof, thresholds):
    st.subheader("Public Health Recommendations")
    st.warning(DISCLAIMER)
    hz = st.radio("Horizon", list(C.HORIZONS), index=1, horizontal=True, key="rc_h", format_func=lambda x: f"{x} weeks ahead")
    now = preds[preds["date"] == asof].sort_values(f"pred_h{hz}", ascending=False)
    ref = feats[feats["date"] <= pd.Timestamp(C.TRAIN_END)]

    city, ward = ward_selector(wards, "rc")
    if ward is None:
        return
    row = get_pred_row(preds, ward, asof)
    frow = feats[(feats["ward_id"] == ward) & (feats["date"] == asof)]
    if row is None or frow.empty:
        st.warning("No prediction available for this ward and week.")
        return
    frow = frow.iloc[0]
    tier, pred = row[f"risk_h{hz}"], row[f"pred_h{hz}"]
    st.markdown(f"**{ward}** - predicted **{pred:.1f}** cases/week in {hz} weeks &nbsp; {badge(tier)}", unsafe_allow_html=True)
    recs = recommend(tier, pred, {"water_logging_risk_index": frow["water_logging_risk_index"], "sanitation_score": frow["sanitation_score"]},
                     driver_percentiles(frow, ref), float(frow["cases_roll4"]))
    colors = {"High": "#e74c3c", "Medium": "#f4d03f", "Low": "#2ecc71"}
    for r in recs:
        st.markdown(f'<div class="mt-rec" style="border-left-color:{colors[r["priority"]]}"><b>[{r["priority"]}]</b> {r["action"]}<br><small>Why: {r["reason"]}</small></div>',
                    unsafe_allow_html=True)

    st.markdown("#### Where to focus first")
    top = now.head(5)
    st.dataframe(pd.DataFrame({"Ward": top["ward_id"], "City": top["city"], "Predicted cases/wk": top[f"pred_h{hz}"].round(1), "Risk": top[f"risk_h{hz}"]}),
                 **WIDE, hide_index=True)


# --------------------------------------------------------------------------------------------------
def tab_intelligence(feats, preds, wards, asof, models):
    st.subheader("🤖 MosquiTech Intelligence")
    st.caption("Ask questions about a ward's forecast. The assistant only explains numbers produced by the ML model and the dataset; "
               "it cannot change predictions or risk categories.")
    client = GeminiClient.from_env()
    if client.available:
        st.success(f"AI provider: Google Gemini · model `{client.model}`. Your question and the ward context shown below are sent to Google's API. "
                   "The context contains aggregated ward-level data only (no personal data).")
    else:
        st.warning(f"AI assistant is not configured: set `{ENV_KEY}` (copy `.env.example` to `.env`) and restart the app. "
                   "All other dashboard features work without it, and the rule-based evidence summary below is still available.")

    city, ward = ward_selector(wards, "mi")
    hz = st.radio("Prediction horizon", list(C.HORIZONS), index=1, horizontal=True, key="mi_h", format_func=lambda x: f"{x} weeks ahead")
    if ward is None:
        return
    asof_str = str(pd.Timestamp(asof).date())
    try:
        ctx = get_context(ward, asof_str, hz)
    except ValueError as exc:
        st.warning(str(exc))
        return

    ml = ctx["ML_PREDICTION_do_not_override"]["selected_horizon"]
    c = st.columns(3)
    c[0].markdown(card("Predicted cases / week", f"{ml['predicted_cases_per_week']:.1f}", f"80% range {ml['interval_80pct'][0]:.1f} - {ml['interval_80pct'][1]:.1f}"), unsafe_allow_html=True)
    c[1].markdown(card("ML risk category", badge(ml["ml_risk_category"]), "set by the model, not the AI"), unsafe_allow_html=True)
    c[2].markdown(card("Evidence status", "", ctx["evidence_status"]), unsafe_allow_html=True)

    with st.expander("Evidence summary (rule-based from the same data, not written by an LLM)", expanded=not client.available):
        for line in evidence_brief(ctx):
            st.markdown(f"- {line}")
        st.caption(DISCLAIMER)

    # ---- chat state --------------------------------------------------------------------------------
    ss = st.session_state
    if "mi_history" not in ss:
        ss["mi_history"] = []
    if "mi_budget" not in ss:
        ss["mi_budget"] = CallBudget(max_calls=25)
    if "mi_cache" not in ss:
        ss["mi_cache"] = {}
    key_ctx = f"{ward}|{asof_str}|{hz}"

    st.markdown("**Suggested questions**")
    pending = None
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 3].button(q, key=f"mi_sq{i}", **WIDE):
            pending = q
    typed = st.text_input("Or ask your own question", key="mi_q", max_chars=500, placeholder="e.g. Why is the 4-week forecast lower than the 2-week one?")
    asked = st.button("Ask MosquiTech Intelligence", key="mi_ask", type="primary")
    question = pending or (typed.strip() if asked else "")

    if question:
        cache_key = f"{key_ctx}|{question.lower()}"
        if cache_key in ss["mi_cache"]:
            ss["mi_history"].append({**ss["mi_cache"][cache_key], "q": question, "key": key_ctx, "cached": True})
        elif not client.available:
            st.warning(f"The AI assistant needs `{ENV_KEY}`. No answer was generated. The evidence summary above shows the underlying facts.")
        elif not ss["mi_budget"].consume():
            st.warning("This session reached its AI question limit (cost control). Reload the page to start a new session.")
        else:
            turns = [h for h in ss["mi_history"] if h["key"] == key_ctx]
            try:
                with st.spinner("MosquiTech Intelligence is reading the model outputs..."):
                    resp = ask(client, ctx, question, [{"q": h["q"], "a": h["a"]} for h in turns])
                entry = {"a": resp.text, "model": resp.model, "truncated": resp.truncated,
                         "tokens": (resp.prompt_tokens or 0) + (resp.output_tokens or 0)}
                ss["mi_cache"][cache_key] = entry
                ss["mi_history"].append({**entry, "q": question, "key": key_ctx, "cached": False})
            except ValueError as exc:
                ss["mi_budget"].used -= 1
                st.warning(str(exc))
            except LLMUnavailable as exc:
                ss["mi_budget"].used -= 1
                st.warning(str(exc))
            except LLMError as exc:
                ss["mi_budget"].used -= 1
                st.error(f"{exc} No answer was generated; the ML predictions above are unaffected.")

    for h in ss["mi_history"]:
        with st.chat_message("user"):
            st.markdown(h["q"])
        with st.chat_message("assistant"):
            st.markdown(h["a"])
            note = f"AI-generated by {h['model']} from the context below"
            if h.get("cached"):
                note += " (repeat question: reused earlier answer, no new API call)"
            if h.get("truncated"):
                note += " · answer was cut off at the length limit"
            st.caption(note + ". Verify against the dashboard values. Decision support, not medical advice.")
    if ss["mi_history"]:
        if st.button("Clear conversation", key="mi_clear"):
            ss["mi_history"] = []
    st.caption(f"AI questions left this session: {ss['mi_budget'].remaining}")

    with st.expander("Exact data sent to the AI for this ward (transparency)"):
        st.json(ctx)


def guarded(fn, *args):
    try:
        fn(*args)
    except Exception as exc:  # keep the rest of the dashboard alive if one module fails
        st.error(f"This section could not be displayed: {exc}")


def main():
    st.markdown('<div class="mt-hero"><h1>🦟 MosquiTech</h1><p>AI-Powered Dengue Early Warning &amp; Urban Risk Intelligence</p></div>', unsafe_allow_html=True)
    try:
        models = get_models()
        feats = get_features()
        preds = get_predictions()
    except (DataError, ModelsMissingError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Startup failed: {exc}")
        st.stop()
    if preds.empty:
        st.error("No predictions could be produced from the dataset.")
        st.stop()

    thresholds = models["thresholds"]
    latest_obs = feats["date"].max()
    dates = sorted(pd.to_datetime(preds["date"].unique()), reverse=True)
    wards = feats.drop_duplicates("ward_id")[["ward_id", "city"]]

    with st.sidebar:
        st.markdown("### 🦟 MosquiTech")
        st.caption("Ward-level dengue forecasts, 2-4 weeks ahead")
        asof = pd.Timestamp(st.selectbox("Forecast issued on (week)", dates, index=0, format_func=fmt_date, key="asof"))
        st.caption(f"Latest observation in dataset: **{fmt_date(latest_obs)}**. Choose an earlier week to review past forecasts "
                   "against what actually happened.")
        st.caption(f"Evidence status (3-week target): {evidence_status(asof + pd.Timedelta(weeks=3), latest_obs)}.")
        st.button("⏩ Jump to 2025 peak outbreak week", on_click=_set_asof, args=(get_peak_week(),), key="jump_peak", **WIDE)
        st.button("⏮ Jump to latest week", on_click=_set_asof, args=(dates[0],), key="jump_latest", **WIDE)
        st.markdown("**Risk tiers (cases / ward / week)**")
        st.markdown(f"{badge('LOW')} &lt; {thresholds[0]:.0f}<br>{badge('MEDIUM')} {thresholds[0]:.0f} - {thresholds[1]:.0f}<br>"
                    f"{badge('HIGH')} {thresholds[1]:.0f} - {thresholds[2]:.0f}<br>{badge('CRITICAL')} &ge; {thresholds[2]:.0f}", unsafe_allow_html=True)
        st.caption("Thresholds come from the training-period 70th / 85th / 95th percentiles of weekly ward cases.")
        st.caption("Decision-support tool - not medical advice.")

    tabs = st.tabs(["📊 Overview", "🔮 Prediction", "🗺️ Ward Risk", "🌧️ Weather", "🧠 Explanation", "🤖 Intelligence", "📈 Model Performance", "🩺 Recommendations"])
    reports = get_reports()
    with tabs[0]:
        guarded(tab_overview, feats, preds, asof, thresholds, latest_obs)
    with tabs[1]:
        guarded(tab_prediction, feats, preds, wards, asof, thresholds)
    with tabs[2]:
        guarded(tab_ward_risk, feats, preds, wards, asof, thresholds)
    with tabs[3]:
        guarded(tab_weather, feats, wards, asof)
    with tabs[4]:
        guarded(tab_explain, feats, preds, wards, asof, models)
    with tabs[5]:
        guarded(tab_intelligence, feats, preds, wards, asof, models)
    with tabs[6]:
        guarded(tab_performance, models, reports, thresholds)
    with tabs[7]:
        guarded(tab_recommendations, feats, preds, wards, asof, thresholds)


main()
