import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from database_and_engine import (
    generate_synthetic_telemetry,
    run_deltapulse_velocity_engine,
    evaluate_bedside_context,
    build_shift_event_strip,
    PATIENT_PROFILES,
    ARTIFACT_TAGS,
    DB_NAME,
    SHIFT_MINUTES,
    CRASH_START_MIN,
    CHECKPOINT_LATE_MIN
)
from ml_benchmark import run_multi_model_benchmark, render_ml_benchmark_tab, TOUCH_CHART_CONFIG

st.set_page_config(
    page_title="DeltaPulse | 12-Bed ICU Central Station",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_data
def load_and_score_cohort():
    if not os.path.exists(DB_NAME):
        generate_synthetic_telemetry(DB_NAME)
    raw_cohort = run_deltapulse_velocity_engine(DB_NAME, patient_id=None)
    scored_cohort, summary_df, feature_weights, rf_imp = run_multi_model_benchmark(raw_cohort)
    return scored_cohort, summary_df, feature_weights, rf_imp


cohort_df, summary_df, feature_weights, rf_importance_df = load_and_score_cohort()

if "selected_pid" not in st.session_state or st.session_state.selected_pid not in PATIENT_PROFILES:
    st.session_state.selected_pid = "VENT-01"
if "selected_min" not in st.session_state or st.session_state.selected_min >= SHIFT_MINUTES:
    st.session_state.selected_min = CHECKPOINT_LATE_MIN
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "play_step" not in st.session_state:
    st.session_state.play_step = 5
if "advance_tick" not in st.session_state:
    st.session_state.advance_tick = False

if st.session_state.advance_tick:
    st.session_state.advance_tick = False
    if st.session_state.is_playing:
        if st.session_state.selected_min < SHIFT_MINUTES - 1:
            st.session_state.selected_min = min(
                SHIFT_MINUTES - 1,
                st.session_state.selected_min + int(st.session_state.play_step)
            )
        else:
            st.session_state.is_playing = False

if "care_logs" not in st.session_state:
    st.session_state.care_logs = {}
for pid in PATIENT_PROFILES.keys():
    st.session_state.care_logs.setdefault(pid, [])


def set_patient(pid: str):
    st.session_state.selected_pid = pid


def jump_to_minute(m: int):
    st.session_state.is_playing = False
    st.session_state.advance_tick = False
    st.session_state.selected_min = int(min(max(30, m), SHIFT_MINUTES - 1))


def toggle_play():
    st.session_state.is_playing = not st.session_state.is_playing
    st.session_state.advance_tick = False


top_h1, top_h2, top_h3 = st.columns([2.6, 1.0, 0.9])
top_h1.markdown("### ⚡ DeltaPulse ICU Central Station")
with top_h2:
    philips_mode = st.toggle("🖥️ IntelliVue Neon", value=False)
with top_h3:
    if st.button("🔄 Reset Data", use_container_width=True):
        st.session_state.is_playing = False
        st.session_state.advance_tick = False
        generate_synthetic_telemetry(DB_NAME)
        st.cache_data.clear()
        st.rerun()

if philips_mode:
    app_bg, card_bg, card_border = "#000000", "#050505", "#1F1F1F"
    active_border, active_glow = "#00FF66", "rgba(0, 255, 102, 0.30)"
    c_norm, c_drift, c_crit = "#00FF66", "#FFD600", "#FF2A2A"
    c_cyan, c_purple, c_pink = "#00E5FF", "#D500F9", "#FF4081"
    plot_bg = "rgba(5, 5, 5, 0.92)"
else:
    app_bg, card_bg, card_border = "#0B0F19", "#111827", "#1F2937"
    active_border, active_glow = "#38BDF8", "rgba(56, 189, 248, 0.25)"
    c_norm, c_drift, c_crit = "#10B981", "#F59E0B", "#EF4444"
    c_cyan, c_purple, c_pink = "#38BDF8", "#A855F7", "#EC4899"
    plot_bg = "rgba(17, 24, 39, 0.60)"

st.markdown(f"""
<style>
    .stApp {{ background-color: {app_bg}; color: #F9FAFB; }}
    .stApp p, .stApp label, .stApp span {{ color: #F9FAFB; }}
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
        background-color: #1F2937 !important;
        color: #F9FAFB !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #374151 !important;
        border-color: {active_border} !important;
        color: #FFFFFF !important;
    }}
    .bed-tile {{ background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 10px 12px; margin-bottom: 5px; }}
    .bed-active {{ border: 2px solid {active_border} !important; box-shadow: 0 0 12px {active_glow}; }}
    .vital-card {{ background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 9px; padding: 12px 14px; margin-bottom: 10px; }}
    .status-normal {{ border-left: 5px solid {c_norm}; }}
    .status-drift {{ border-left: 5px solid {c_drift}; background-color: rgba(245, 158, 11, 0.07); }}
    .status-critical {{ border-left: 5px solid {c_crit}; background-color: rgba(239, 68, 68, 0.14); }}
    .vital-header {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #9CA3AF; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }}
    .vital-value {{ font-family: monospace; font-size: 1.85rem; font-weight: 800; color: #FFFFFF; margin: 4px 0; }}
    .vital-sub {{ font-size: 0.80rem; color: #D1D5DB; }}
    .pill-normal {{ background-color: rgba(16, 185, 129, 0.20); color: {c_norm}; padding: 2px 8px; border-radius: 999px; font-size: 0.70rem; font-weight: 800; }}
    .pill-drift {{ background-color: rgba(245, 158, 11, 0.22); color: {c_drift}; padding: 2px 8px; border-radius: 999px; font-size: 0.70rem; font-weight: 800; }}
    .pill-critical {{ background-color: rgba(239, 68, 68, 0.28); color: {c_crit}; padding: 2px 8px; border-radius: 999px; font-size: 0.70rem; font-weight: 800; }}
</style>
""", unsafe_allow_html=True)

current_min = st.session_state.selected_min
snapshot_rows = {}
alert_count, watch_count, stable_count = 0, 0, 0

for pid in PATIENT_PROFILES.keys():
    p_row = cohort_df[(cohort_df["patient_id"] == pid) & (cohort_df["minute_index"] == current_min)].iloc[0]
    snapshot_rows[pid] = p_row
    if p_row["pre_ehr_alert"]:
        alert_count += 1
    elif bool(p_row.get("phys_squeeze_active", False)) or bool(p_row.get("artifact_suppressed", False)):
        watch_count += 1
    else:
        stable_count += 1

pod_f1, pod_f2 = st.columns([2.1, 1.4])
with pod_f1:
    pod_filter = st.radio(
        "Unit Filter:",
        ["All 12 Beds", "Pod A (01–06)", "Pod B (07–12)", "🚨 Alerts Only"],
        horizontal=True,
        label_visibility="collapsed"
    )
with pod_f2:
    st.markdown(
        f"<div style='text-align:right; font-size:0.90rem; padding-top:4px;'>"
        f"<span style='color:{c_crit}; font-weight:800;'>🚨 {alert_count} Alert</span> &nbsp;•&nbsp; "
        f"<span style='color:{c_drift}; font-weight:800;'>🌀 {watch_count} Watch</span> &nbsp;•&nbsp; "
        f"<span style='color:{c_norm}; font-weight:800;'>🟢 {stable_count} Stable</span></div>",
        unsafe_allow_html=True
    )

visible_pids = []
for pid, meta in PATIENT_PROFILES.items():
    if pod_filter == "Pod A (01–06)" and meta.get("pod") != "Pod A":
        continue
    if pod_filter == "Pod B (07–12)" and meta.get("pod") != "Pod B":
        continue
    if pod_filter == "🚨 Alerts Only" and not bool(snapshot_rows[pid]["pre_ehr_alert"]):
        continue
    visible_pids.append(pid)

if not visible_pids:
    st.info("No active alerts at this time. Switch to **All 12 Beds** or tap **🚨 Late Crash (06:05 AM)**.")
else:
    for row_start in range(0, len(visible_pids), 4):
        row_pids = visible_pids[row_start:row_start + 4]
        bed_cols = st.columns(4)
        for col_idx, pid in enumerate(row_pids):
            meta = PATIENT_PROFILES[pid]
            p_row = snapshot_rows[pid]
            is_active = (pid == st.session_state.selected_pid)
            active_cls = "bed-active" if is_active else ""

            if p_row["pre_ehr_alert"]:
                badge_html = '<span class="pill-critical">🚨 ALERT</span>'
            elif bool(p_row.get("phys_squeeze_active", False)):
                badge_html = '<span class="pill-drift">🌀 COIL</span>'
            elif bool(p_row.get("artifact_suppressed", False)):
                badge_html = '<span class="pill-drift">🛡️ FILTERED</span>'
            else:
                badge_html = '<span class="pill-normal">🟢 STABLE</span>'

            short_label = meta["name"].split("—")[1].strip()
            with bed_cols[col_idx]:
                st.markdown(
                    f"""
                    <div class="bed-tile {active_cls}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="font-size:0.90rem; color:#FFFFFF;">{meta['bed']} • {pid}</strong>
                            {badge_html}
                        </div>
                        <div style="font-size:0.78rem; color:#9CA3AF; margin:3px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{short_label}</div>
                        <div style="font-family:monospace; font-size:0.84rem; color:#F3F4F6;">
                            RR:<b>{p_row['rr_clean']}</b> | PIP:<b>{p_row['pip_clean']}</b> | BP:<b>{int(p_row['sbp_clean'])}/{int(p_row['dbp_clean'])}</b> | Risk:<b>{p_row['prob_random']}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.button(
                    f"{'✅ Viewing ' if is_active else 'Open '}{meta['bed']}",
                    key=f"btn_{pid}",
                    on_click=set_patient,
                    args=(pid,),
                    use_container_width=True
                )

selected_pid = st.session_state.selected_pid
profile_meta = PATIENT_PROFILES[selected_pid]
df = cohort_df[cohort_df["patient_id"] == selected_pid].reset_index(drop=True)

with st.container(border=True):
    ctrl1, ctrl2, ctrl3 = st.columns([2.3, 1.5, 1.2])

    with ctrl1:
        selected_min = st.slider(
            "⏱️ 24-Hour Shift Timeline (07:00 AM – 06:59 AM)",
            min_value=30,
            max_value=SHIFT_MINUTES - 1,
            key="selected_min"
        )
        current_row = df.iloc[selected_min]
        shift_clock = current_row["timestamp"].strftime("%I:%M %p")
        is_day_shift = selected_min < 720
        shift_badge = "☀️ DAY SHIFT" if is_day_shift else "🌙 NIGHT SHIFT"
        shift_elapsed = selected_min if is_day_shift else (selected_min - 720)
        st.caption(
            f"**{shift_badge}** • **{shift_clock}** ({shift_elapsed // 60}h {shift_elapsed % 60:02d}m into shift)"
        )

        p_col1, p_col2, p_col3 = st.columns([1.1, 1.1, 1.2])
        play_label = "⏸️ Pause" if st.session_state.is_playing else "▶️ Play"
        p_col1.button(play_label, on_click=toggle_play, use_container_width=True)
        speed_choice = p_col2.selectbox(
            "Speed",
            options=[1, 5, 15],
            format_func=lambda x: f"⏩ {x}m / tick",
            key="play_step",
            label_visibility="collapsed"
        )
        p_col3.button("🔄 07:30 AM", on_click=jump_to_minute, args=(30,), use_container_width=True)

    with ctrl2:
        st.markdown("**⚡ Quick Jumps**")
        artifact_target = profile_meta["artifacts"][0] + 1
        b_row1_c1, b_row1_c2 = st.columns(2)
        b_row1_c1.button("☀️ Morning (07:30)", on_click=jump_to_minute, args=(30,), use_container_width=True)
        b_row1_c2.button("🌀 Coil (04:50 AM)", on_click=jump_to_minute, args=(CRASH_START_MIN - 10,), use_container_width=True)
        b_row2_c1, b_row2_c2 = st.columns(2)
        b_row2_c1.button(f"🛡️ Spike (m{artifact_target})", on_click=jump_to_minute, args=(artifact_target,), use_container_width=True)
        b_row2_c2.button("🚨 Crash (06:05 AM)", on_click=jump_to_minute, args=(CHECKPOINT_LATE_MIN,), use_container_width=True)

    with ctrl3:
        beginner_mode = st.toggle("🩺 Plain-English Mode", value=True)
        legacy_fp = int(df.iloc[:CRASH_START_MIN]["legacy_static_alert"].sum())
        deltapulse_fp = int(df.iloc[:CRASH_START_MIN]["pre_ehr_alert"].sum())
        clicks_saved = int((selected_min / 60.0) * 12 * len(PATIENT_PROFILES))
        st.caption(
            f"🛡️ **False Alarms (22h):** `{deltapulse_fp}` vs `{legacy_fp}` Legacy\n\n"
            f"⚡ **EHR Clicks Saved:** `{clicks_saved}`"
        )

st.markdown(
    f"#### {profile_meta['bed']} ({selected_pid}) — {profile_meta['name']}\n"
    f"**Dx:** {profile_meta['diagnosis']} &nbsp;•&nbsp; **Vent:** `{profile_meta['vent_mode']}` &nbsp;•&nbsp; **Lines:** `{profile_meta.get('line_status', 'Patent')}`"
)


def render_vital_card(title: str, value_str: str, z_val: float, vel_str: str, base_mean: float, unit: str, beginner: bool, invert_low: bool = False):
    abs_z = abs(z_val) if pd.notna(z_val) else 0.0
    if invert_low:
        is_crit = z_val <= -2.0
        is_drift = -2.0 < z_val <= -1.5
    else:
        is_crit = abs_z >= 2.5
        is_drift = 1.8 <= abs_z < 2.5

    if is_crit:
        card_cls, pill_cls, pill_txt = "status-critical", "pill-critical", "🔴 HIGH" if beginner else f"{z_val:+g} SD"
    elif is_drift:
        card_cls, pill_cls, pill_txt = "status-drift", "pill-drift", "🟡 DRIFT" if beginner else f"{z_val:+g} SD"
    else:
        card_cls, pill_cls, pill_txt = "status-normal", "pill-normal", "🟢 NORMAL" if beginner else f"{z_val:+g} SD"

    sub_text = f"Baseline: <b>{base_mean} {unit}</b> • {vel_str}" if beginner else f"4h Mean: <b>{base_mean} {unit}</b> • {vel_str} • Z: <b>{z_val:+g} SD</b>"
    return f"""
    <div class="vital-card {card_cls}">
        <div class="vital-header"><span>{title}</span><span class="{pill_cls}">{pill_txt}</span></div>
        <div class="vital-value">{value_str}</div>
        <div class="vital-sub">{sub_text}</div>
    </div>
    """


left_col, right_col = st.columns([1.08, 1.42], gap="large")
active_alerts = evaluate_bedside_context(current_row)
suppressed_so_far = int(df.iloc[:selected_min + 1]["artifact_suppressed"].sum())
patient_care_logs = st.session_state.care_logs.get(selected_pid, [])

with left_col:
    if current_row["pre_ehr_alert"] and active_alerts:
        trigger_lines = [f"- **{a['title']}:** {a['signal_read']}" for a in active_alerts]
        deduped_cues = list(dict.fromkeys(cue for a in active_alerts for cue in a["bedside_cues"]))
        action_md = ("\n\n**🩺 Bedside Action Check:**\n" + "\n".join([f"- {c}" for c in deduped_cues[:3]])) if beginner_mode and deduped_cues else ""
        st.error(f"🚨 **TREND ALERT AT {shift_clock}**\n\n" + "\n".join(trigger_lines) + action_md)
    elif bool(current_row.get("phys_squeeze_active", False)) and active_alerts:
        trigger_lines = [f"- **{a['title']}:** {a['signal_read']}" for a in active_alerts]
        deduped_cues = list(dict.fromkeys(cue for a in active_alerts for cue in a["bedside_cues"]))
        action_md = ("\n\n**🩺 Early Check:**\n" + "\n".join([f"- {c}" for c in deduped_cues[:3]])) if beginner_mode and deduped_cues else ""
        st.warning(f"🌀 **PRE-BREAKOUT COIL AT {shift_clock}**\n\n" + "\n".join(trigger_lines) + action_md)
    elif bool(current_row.get("artifact_suppressed", False)):
        st.warning(f"🛡️ **CARE SPIKE FILTERED AT {shift_clock}**\n\nBrief sensor blip damped by 5-minute median—false alarm prevented.")
    else:
        st.success(f"✅ **STABLE AT {shift_clock} — Within Personal Baseline**")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(render_vital_card("Resp Rate (RR)", f"{current_row['rr_clean']} bpm", current_row["rr_zscore"], f"{current_row['dRR_dt_30m']:+g}/30m", current_row["rr_base_mean"], "bpm", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Peak Pressure (PIP)", f"{current_row['pip_clean']} cmH₂O", current_row["pip_zscore"], f"{current_row['dPIP_dt_30m']:+g}/30m", current_row["pip_base_mean"], "cmH₂O", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Blood Pressure (BP)", f"{int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])}", current_row["sbp_zscore"], f"{current_row['dSBP_dt_30m']:+g}/30m", current_row["sbp_base_mean"], "mmHg", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Heart Rate (HR)", f"{current_row['hr_clean']} bpm", current_row["hr_zscore"], f"Raw: {current_row['hr']}", current_row["hr_base_mean"], "bpm", beginner_mode), unsafe_allow_html=True)

    with g_col2:
        st.markdown(render_vital_card("End-Tidal CO₂", f"{current_row['etco2_clean']} mmHg", current_row["etco2_zscore"], f"{current_row['dEtCO2_dt_30m']:+g}/30m", current_row["etco2_base_mean"], "mmHg", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Pulse Pressure", f"{current_row['pulse_pressure_clean']} mmHg", current_row["pulse_pressure_zscore"], f"{current_row['dPP_dt_30m']:+g}/30m", current_row["pulse_pressure_base_mean"], "mmHg", beginner_mode, invert_low=True), unsafe_allow_html=True)
        st.markdown(render_vital_card("Temp Gap (ΔT)", f"{current_row['temp_gradient_clean']} °C", current_row["temp_gradient_zscore"], f"Coil: {current_row['phys_squeeze_score']}%", current_row["temp_gradient_base_mean"], "°C", beginner_mode), unsafe_allow_html=True)

        uo_z = (current_row["urine_output_clean"] - 45.0) / 8.0
        spo2_uo_z = min(uo_z, float(current_row["spo2_zscore"])) if pd.notna(current_row["spo2_zscore"]) else uo_z

        st.markdown(render_vital_card("SpO₂ & Urine Out", f"{current_row['spo2_clean']}% | {int(current_row['urine_output_clean'])} mL/h", spo2_uo_z, f"SpO₂ Z: {current_row['spo2_zscore']:+g}", 48.0, "mL/h", beginner_mode, invert_low=True), unsafe_allow_html=True)

with right_col:
    window_df = df.iloc[max(0, selected_min - 120):selected_min + 1].copy()
    r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs(["🫁 Waveforms", "🎯 Z-Scores & Coil", "🧠 ML Forecast", "📝 Care Log"])

    with r_tab1:
        st.caption("👆 Drag with 1 finger to pan • Pinch to zoom • Double-tap to reset")
        fig_air = go.Figure()
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["rr"], mode="lines", name="Raw RR", line=dict(color="rgba(156,163,175,0.45)", width=1.5, dash="dot")))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["rr_clean"], mode="lines", name="Clean RR", line=dict(color=c_cyan, width=3)))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pip"], mode="lines", name="Raw PIP", line=dict(color="rgba(248,113,113,0.40)", width=1.5, dash="dot")))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pip_clean"], mode="lines", name="Clean PIP", line=dict(color=c_drift, width=3)))
        fig_air.update_layout(
            title="Airway Trends (RR & PIP)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg,
            dragmode="pan",
            height=300, margin=dict(l=10, r=10, t=36, b=52),
            legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_air, use_container_width=True, config=TOUCH_CHART_CONFIG)

        fig_hemo = go.Figure()
        fig_hemo.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["sbp_clean"], mode="lines", name="Systolic BP", line=dict(color=c_crit, width=2.5)))
        fig_hemo.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["etco2_clean"], mode="lines", name="EtCO₂", line=dict(color=c_purple, width=2.5)))
        fig_hemo.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pulse_pressure_clean"], mode="lines", name="Pulse Pressure", line=dict(color=c_norm, width=2.5)))
        fig_hemo.update_layout(
            title="Perfusion & CO₂ Trends",
            template="plotly_dark", pape
