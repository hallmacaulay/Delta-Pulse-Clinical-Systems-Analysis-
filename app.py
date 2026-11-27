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
    build_12h_shift_handoff,
    is_in_care_mute_window,
    PATIENT_PROFILES,
    ARTIFACT_TAGS,
    DB_NAME,
    SHIFT_MINUTES,
    CRASH_START_MIN,
    CHECKPOINT_LATE_MIN
)
from ml_benchmark import run_multi_model_benchmark, render_ml_benchmark_tab, TOUCH_CHART_CONFIG

st.set_page_config(
    page_title="DeltaPulse | 12-Bed ICU & RT Central Station",
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
if "ack_logs" not in st.session_state:
    st.session_state.ack_logs = {}
for pid in PATIENT_PROFILES.keys():
    st.session_state.care_logs.setdefault(pid, [])
if "latched_eta_memory" not in st.session_state:
    st.session_state.latched_eta_memory = {}


def set_patient(pid: str):
    st.session_state.selected_pid = pid


def jump_to_minute(m: int):
    st.session_state.is_playing = False
    st.session_state.advance_tick = False
    st.session_state.latched_eta_memory.pop(st.session_state.selected_pid, None)
    st.session_state.selected_min = int(min(max(30, m), SHIFT_MINUTES - 1))


def toggle_play():
    st.session_state.is_playing = not st.session_state.is_playing
    st.session_state.advance_tick = False


top_h1, top_h2, top_h3 = st.columns([2.6, 1.0, 0.9])
top_h1.markdown("### ⚡ DeltaPulse ICU & RT Central Station")
with top_h2:
    philips_mode = st.toggle("🖥️ IntelliVue Neon", value=False)
with top_h3:
    if st.button("🔄 Reset Data", use_container_width=True):
        st.session_state.is_playing = False
        st.session_state.advance_tick = False
        st.session_state.latched_eta_memory = {}
        st.session_state.ack_logs = {}
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
    div[data-testid="stMetricDelta"] > div {{
        background-color: #1F2937 !important;
        padding: 2px 8px !important;
        border-radius: 999px !important;
    }}
    .bed-tile {{ background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 10px 12px; margin-bottom: 5px; }}
    .bed-active {{ border: 2px solid {active_border} !important; box-shadow: 0 0 12px {active_glow}; }}
    .vital-card {{ background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 9px; padding: 12px 14px; margin-bottom: 10px; }}
    .status-normal {{ border-left: 5px solid {c_norm}; }}
    .status-drift {{ border-left: 5px solid {c_drift}; background-color: rgba(245, 158, 11, 0.07); }}
    .status-critical {{ border-left: 5px solid {c_crit}; background-color: rgba(239, 68, 68, 0.14); }}
    .vital-header {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.80rem; color: #9CA3AF; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }}
    .vital-value {{ font-family: monospace; font-size: 1.78rem; font-weight: 800; color: #FFFFFF; margin: 4px 0; }}
    .vital-sub {{ font-size: 0.78rem; color: #D1D5DB; }}
    .pill-normal {{ background-color: rgba(16, 185, 129, 0.20); color: {c_norm}; padding: 2px 8px; border-radius: 999px; font-size: 0.68rem; font-weight: 800; }}
    .pill-drift {{ background-color: rgba(245, 158, 11, 0.22); color: {c_drift}; padding: 2px 8px; border-radius: 999px; font-size: 0.68rem; font-weight: 800; }}
    .pill-critical {{ background-color: rgba(239, 68, 68, 0.28); color: {c_crit}; padding: 2px 8px; border-radius: 999px; font-size: 0.68rem; font-weight: 800; }}
    .pill-ack {{ background-color: rgba(56, 189, 248, 0.22); color: #38BDF8; padding: 2px 8px; border-radius: 999px; font-size: 0.68rem; font-weight: 800; }}
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


def render_status_banner(kind: str, title: str, body_lines: list[str] | None = None) -> str:
    palette = {
        "success": ("#10B981", "rgba(16, 185, 129, 0.14)"),
        "info":    ("#38BDF8", "rgba(56, 189, 248, 0.14)"),
        "warning": ("#F59E0B", "rgba(245, 158, 11, 0.16)"),
        "error":   ("#EF4444", "rgba(239, 68, 68, 0.18)"),
    }
    border_col, bg_col = palette.get(kind, ("#38BDF8", "#111827"))
    body_html = ""
    if body_lines:
        items = "".join(f"<div style='margin-top:4px; font-size:0.88rem; color:#F3F4F6;'>{line}</div>" for line in body_lines)
        body_html = f"<div style='margin-top:6px;'>{items}</div>"
    return f"""
    <div style="background-color:{bg_col}; border:1px solid #374151; border-left:5px solid {border_col};
                border-radius:9px; padding:12px 14px; margin-bottom:12px; color:#FFFFFF;">
        <div style="font-weight:800; font-size:0.95rem; color:#FFFFFF;">{title}</div>
        {body_html}
    </div>
    """


if not visible_pids:
    st.markdown(render_status_banner("info", "ℹ️ No active alerts at this time", ["Switch to <b>All 12 Beds</b> or tap <b>🚨 Crash (06:05 AM)</b>."]), unsafe_allow_html=True)
else:
    for row_start in range(0, len(visible_pids), 4):
        row_pids = visible_pids[row_start:row_start + 4]
        bed_cols = st.columns(4)
        for col_idx, pid in enumerate(row_pids):
            meta = PATIENT_PROFILES[pid]
            p_row = snapshot_rows[pid]
            is_active = (pid == st.session_state.selected_pid)
            active_cls = "bed-active" if is_active else ""
            ack_state = st.session_state.ack_logs.get(pid)

            if p_row["pre_ehr_alert"]:
                badge_html = '<span class="pill-ack">✅ ACKED</span>' if ack_state else '<span class="pill-critical">🚨 ALERT</span>'
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
                        <div style="font-family:monospace; font-size:0.82rem; color:#F3F4F6;">
                            RR:<b>{p_row['rr_clean']}</b> | VTe:<b>{int(p_row['vte_clean'])}</b> | PIP:<b>{p_row['pip_clean']}</b> | Risk:<b>{p_row['prob_random']}%</b>
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
        sqi_now = str(current_row.get("sqi_status", "GOOD"))
        st.caption(
            f"🛡️ **False Alarms (22h):** `{deltapulse_fp}` vs `{legacy_fp}` Legacy\n\n"
            f"📡 **Signal Quality:** `{sqi_now}`"
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
active_care_mute = is_in_care_mute_window(patient_care_logs, selected_min, window_mins=5)

with left_col:
    if str(current_row.get("sqi_status", "GOOD")) == "CHECK PROBE":
        st.markdown(render_status_banner("warning", f"📡 CHECK SPO₂ PROBE AT {shift_clock}", ["Sudden sensor drop detected; holding last valid SpO₂ to prevent false alarm."]), unsafe_allow_html=True)
    elif active_care_mute and current_row["pre_ehr_alert"]:
        st.markdown(render_status_banner("info", f"🫁 EXPECTED CARE VARIANCE AT {shift_clock}", [f"Active RT/airway care window: <b>{active_care_mute}</b> — alert held for 5 min."]), unsafe_allow_html=True)
    elif current_row["pre_ehr_alert"] and active_alerts:
        lines = [f"• <b>{a['title']}:</b> {a['signal_read'].replace('`', '')}" for a in active_alerts]
        deduped_cues = list(dict.fromkeys(cue for a in active_alerts for cue in a["bedside_cues"]))
        if beginner_mode and deduped_cues:
            lines.append("<div style='margin-top:6px; font-weight:700; color:#FCA5A5;'>🩺 Bedside Action Check:</div>")
            lines.extend([f"→ {c}" for c in deduped_cues[:3]])
        st.markdown(render_status_banner("error", f"🚨 TREND ALERT AT {shift_clock}", lines), unsafe_allow_html=True)

        # Closed-Loop Alert Acknowledgment & Escalation Bar
        ack_now = st.session_state.ack_logs.get(selected_pid)
        if ack_now:
            st.markdown(render_status_banner("info", f"✅ Acknowledged ({ack_now['role']} • {ack_now['time_str']})", [f"Status: <b>{ack_now['status']}</b>"]), unsafe_allow_html=True)
        ack_c1, ack_c2, ack_c3 = st.columns(3)
        if ack_c1.button("✅ Ack (RN)", key=f"ack_rn_{selected_pid}", use_container_width=True):
            st.session_state.ack_logs[selected_pid] = {"role": "RN", "time_str": shift_clock, "status": "At Bedside Assessing"}
            st.session_state.care_logs[selected_pid].append({"minute_index": int(selected_min), "time_str": shift_clock, "type": "Alert Ack", "category": "Closed-Loop", "label": "RN At Bedside", "detail": "Alert acknowledged by RN"})
            st.rerun()
        if ack_c2.button("🫁 Ack (RT)", key=f"ack_rt_{selected_pid}", use_container_width=True):
            st.session_state.ack_logs[selected_pid] = {"role": "RT", "time_str": shift_clock, "status": "RT Assessing Airway/Vent"}
            st.session_state.care_logs[selected_pid].append({"minute_index": int(selected_min), "time_str": shift_clock, "type": "Alert Ack", "category": "Closed-Loop", "label": "RT At Bedside", "detail": "Alert acknowledged by RT"})
            st.rerun()
        if ack_c3.button("📟 Escalate MD", key=f"esc_md_{selected_pid}", use_container_width=True):
            st.session_state.ack_logs[selected_pid] = {"role": "RN/RT", "time_str": shift_clock, "status": "Escalated to ICU MD"}
            st.session_state.care_logs[selected_pid].append({"minute_index": int(selected_min), "time_str": shift_clock, "type": "Escalation", "category": "Closed-Loop", "label": "Escalated to MD", "detail": "Provider notified of sustained trend breach"})
            st.rerun()
    elif bool(current_row.get("phys_squeeze_active", False)) and active_alerts:
        lines = [f"• <b>{a['title']}:</b> {a['signal_read'].replace('`', '')}" for a in active_alerts]
        deduped_cues = list(dict.fromkeys(cue for a in active_alerts for cue in a["bedside_cues"]))
        if beginner_mode and deduped_cues:
            lines.append("<div style='margin-top:6px; font-weight:700; color:#FCD34D;'>🩺 Early Check:</div>")
            lines.extend([f"→ {c}" for c in deduped_cues[:3]])
        st.markdown(render_status_banner("warning", f"🌀 PRE-BREAKOUT COIL AT {shift_clock}", lines), unsafe_allow_html=True)
    elif bool(current_row.get("artifact_suppressed", False)):
        st.markdown(render_status_banner("warning", f"🛡️ CARE SPIKE FILTERED AT {shift_clock}", ["Brief sensor blip damped by 5-minute median — false alarm prevented."]), unsafe_allow_html=True)
    else:
        st.markdown(render_status_banner("success", f"✅ STABLE AT {shift_clock} — Within Personal Baseline"), unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(render_vital_card("Resp Rate & Min Vent", f"{current_row['rr_clean']} bpm | {current_row['mv_clean']} L/m", current_row["rr_zscore"], f"RR {current_row['dRR_dt_30m']:+g}/30m", current_row["rr_base_mean"], "bpm", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Peak Pressure (PIP)", f"{current_row['pip_clean']} cmH₂O", current_row["pip_zscore"], f"{current_row['dPIP_dt_30m']:+g}/30m", current_row["pip_base_mean"], "cmH₂O", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Blood Pressure (BP)", f"{int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])}", current_row["sbp_zscore"], f"{current_row['dSBP_dt_30m']:+g}/30m", current_row["sbp_base_mean"], "mmHg", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Heart Rate (HR)", f"{current_row['hr_clean']} bpm", current_row["hr_zscore"], f"Raw: {current_row['hr']}", current_row["hr_base_mean"], "bpm", beginner_mode), unsafe_allow_html=True)

    with g_col2:
        st.markdown(render_vital_card("Exhaled VTe & RSBI", f"{int(current_row['vte_clean'])} mL | RSBI {current_row['rsbi_clean']}", current_row["vte_zscore"], f"VTe {current_row['dVTe_dt_30m']:+g}/30m", current_row["vte_base_mean"], "mL", beginner_mode, invert_low=True), unsafe_allow_html=True)
        st.markdown(render_vital_card("End-Tidal CO₂", f"{current_row['etco2_clean']} mmHg", current_row["etco2_zscore"], f"{current_row['dEtCO2_dt_30m']:+g}/30m", current_row["etco2_base_mean"], "mmHg", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Pulse Pressure", f"{current_row['pulse_pressure_clean']} mmHg", current_row["pulse_pressure_zscore"], f"{current_row['dPP_dt_30m']:+g}/30m", current_row["pulse_pressure_base_mean"], "mmHg", beginner_mode, invert_low=True), unsafe_allow_html=True)

        uo_z = (current_row["urine_output_clean"] - 45.0) / 8.0
        spo2_uo_z = min(uo_z, float(current_row["spo2_zscore"])) if pd.notna(current_row["spo2_zscore"]) else uo_z

        st.markdown(render_vital_card("SpO₂ (S/F) & Urine", f"{current_row['spo2_clean']}% ({int(current_row['sf_ratio_clean'])}) | {int(current_row['urine_output_clean'])} mL/h", spo2_uo_z, f"Temp ΔT: {current_row['temp_gradient_clean']}°C", 48.0, "mL/h", beginner_mode, invert_low=True), unsafe_allow_html=True)

with right_col:
    window_df = df.iloc[max(0, selected_min - 120):selected_min + 1].copy()
    r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs(["🫁 Waveforms", "🎯 Z-Scores & Coil", "🧠 ML Forecast", "📝 Care & Med Log"])

    with r_tab1:
        st.caption("👆 Drag with 1 finger to pan • Pinch to zoom • Double-tap to reset")
        fig_air = go.Figure()
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["rr"], mode="lines", name="Raw RR", line=dict(color="rgba(156,163,175,0.45)", width=1.5, dash="dot")))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["rr_clean"], mode="lines", name="Clean RR", line=dict(color=c_cyan, width=3)))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pip"], mode="lines", name="Raw PIP", line=dict(color="rgba(248,113,113,0.40)", width=1.5, dash="dot")))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pip_clean"], mode="lines", name="Clean PIP", line=dict(color=c_drift, width=3)))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["rsbi_clean"], mode="lines", name="RSBI", line=dict(color=c_pink, width=2, dash="dash")))
        fig_air.update_layout(
            title="Airway & Weaning Trends (RR, PIP & RSBI)",
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
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg,
            dragmode="pan",
            height=290, margin=dict(l=10, r=10, t=36, b=52),
            legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_hemo, use_container_width=True, config=TOUCH_CHART_CONFIG)

    with r_tab2:
        fig_z = go.Figure()
        fig_z.add_hrect(y0=-1.8, y1=1.8, fillcolor="rgba(16, 185, 129, 0.10)", line_width=0, annotation_text="Normal Zone")
        fig_z.add_hline(y=2.5, line_dash="dash", line_color=c_crit, annotation_text="+2.5 SD")
        fig_z.add_hline(y=-2.5, line_dash="dash", line_color=c_drift, annotation_text="-2.5 SD (VTe/PP)")
        for col, label, color in [("rr_zscore", "RR", c_cyan), ("pip_zscore", "PIP", c_drift), ("vte_zscore", "VTe", "#34D399"), ("etco2_zscore", "EtCO₂", c_purple), ("sbp_zscore", "SBP", c_crit), ("pulse_pressure_zscore", "Pulse Press", c_pink)]:
            fig_z.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df[col], mode="lines", name=label, line=dict(color=color, width=2.2)))
        fig_z.update_layout(
            title="Personal Baseline Z-Scores (SD)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg,
            dragmode="pan",
            height=335, margin=dict(l=10, r=10, t=36, b=62), yaxis_title="Z-Score (SD)",
            legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_z, use_container_width=True, config=TOUCH_CHART_CONFIG)

        fig_sq = go.Figure()
        fig_sq.add_trace(go.Scatter(
            x=window_df["timestamp"], y=window_df["phys_squeeze_score"],
            mode="lines", name="Coil Index (%)", fill="tozeroy",
            line=dict(color=c_drift, width=2.5), fillcolor="rgba(245, 158, 11, 0.16)"
        ))
        fig_sq.add_hline(y=55.0, line_dash="dash", line_color=c_drift, annotation_text="55% Coil Watch")
        fig_sq.update_layout(
            title="Variability Coil Index (%)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg,
            dragmode="pan",
            height=215, margin=dict(l=10, r=10, t=36, b=28), yaxis=dict(range=[0, 100], title="Coil (%)")
        )
        st.plotly_chart(fig_sq, use_container_width=True, config=TOUCH_CHART_CONFIG)

    with r_tab3:
        render_ml_benchmark_tab(st, df, summary_df, rf_importance_df, selected_min, philips_mode=philips_mode)

    with r_tab4:
        st.markdown(f"**📝 Log Bedside Note, RT Care, or Med Titration (`{shift_clock}`)**")
        with st.form(key=f"note_form_{selected_pid}", clear_on_submit=True):
            nf_col1, nf_col2 = st.columns([3.0, 1.0])
            note_input = nf_col1.text_input(
                "Note",
                placeholder="Type brief non-PHI note (e.g., thick white secretions suctioned)...",
                label_visibility="collapsed"
            )
            save_clicked = nf_col2.form_submit_button("💾 Save Note", use_container_width=True)
            if save_clicked and note_input.strip():
                st.session_state.care_logs[selected_pid].append({
                    "minute_index": int(selected_min),
                    "time_str": shift_clock,
                    "type": "Bedside Note",
                    "category": "Clinical Note",
                    "label": "Bedside Note",
                    "detail": note_input.strip(),
                })
                st.rerun()

        tag_cols = st.columns(3)
        for cat_idx, (cat_name, tags) in enumerate(ARTIFACT_TAGS.items()):
            with tag_cols[cat_idx]:
                st.caption(f"**{cat_name}**")
                for tag_label in tags:
                    if st.button(f"+ {tag_label}", key=f"tag_{selected_pid}_{tag_label}", use_container_width=True):
                        st.session_state.care_logs[selected_pid].append({
                            "minute_index": int(selected_min),
                            "time_str": shift_clock,
                            "type": "Care / Med Tag",
                            "category": cat_name,
                            "label": tag_label,
                            "detail": cat_name,
                        })
                        st.rerun()

        h_strip1, h_strip2 = st.columns([3.0, 1.0])
        h_strip1.markdown(f"**🧭 Shift Event & Med Timeline** • Filtered Blips: `{suppressed_so_far}`")
        if h_strip2.button("🗑️ Clear", key=f"clr_{selected_pid}", use_container_width=True):
            st.session_state.care_logs[selected_pid] = []
            st.session_state.ack_logs.pop(selected_pid, None)
            st.rerun()

        strip_df = build_shift_event_strip(
            df,
            selected_min,
            st.session_state.care_logs[selected_pid],
            med_events=profile_meta.get("med_events", [])
        )
        if strip_df.empty:
            st.markdown(render_status_banner("info", "📝 No events or notes logged yet"), unsafe_allow_html=True)
        else:
            st.dataframe(strip_df, use_container_width=True, hide_index=True, height=255)

    with st.expander("📋 EHR Scribe & 12h Handoff (Epic / Cerner / RT Vent Check / SBAR)", expanded=False):
        ehr_system = st.radio(
            "Select Output Format:",
            ["🫁 RT Vent Check", "🔄 12h Shift Handoff (SBAR)", "Epic (.dotphrase)", "Cerner (iView)", "HL7 FHIR JSON"],
            horizontal=True
        )
        status_line = (
            f"🚨 TREND ALERT ACTIVE (ML Risk: {current_row['prob_random']}%)"
            if current_row["pre_ehr_alert"]
            else (
                f"🌀 PRE-BREAKOUT COIL ({current_row['phys_squeeze_score']}% | ML Risk: {current_row['prob_random']}%)"
                if current_row.get("phys_squeeze_active", False)
                else f"🟢 Within Personal Baseline (ML Risk: {current_row['prob_random']}%)"
            )
        )
        logged_tags_upto = [t for t in patient_care_logs if int(t["minute_index"]) <= selected_min]
        care_tag_summary = (
            "; ".join(f"{t['time_str']} {t['label']}: {t['detail']}" if t['label'] == 'Bedside Note' else f"{t['time_str']} {t['label']}" for t in logged_tags_upto[-5:])
            if logged_tags_upto
            else "None logged"
        )
        ack_info = st.session_state.ack_logs.get(selected_pid)

        if "RT Vent" in ehr_system:
            macro_text = (
                f"DELTAPULSE [RESPIRATORY THERAPY VENTILATOR CHECK]\n"
                f"Time: {shift_clock} ({shift_badge}) | Bed: {profile_meta['bed']} ({selected_pid}, {profile_meta['age_sex']})\n"
                f"• Mode / Settings: {profile_meta['vent_mode']}\n"
                f"• Mechanics & Volumes: PIP {current_row['pip_clean']} cmH2O (Base {current_row['pip_base_mean']}, Z={current_row['pip_zscore']:+g}) | "
                f"Exhaled VTe {int(current_row['vte_clean'])} mL (Base {int(current_row['vte_base_mean'])}, Z={current_row['vte_zscore']:+g}) | "
                f"Min Vent {current_row['mv_clean']} L/min\n"
                f"• Weaning & Gas Exchange: Actual RR {current_row['rr_clean']} bpm | RSBI {current_row['rsbi_clean']} bpm/L | "
                f"EtCO2 {current_row['etco2_clean']} mmHg (Z={current_row['etco2_zscore']:+g}) | SpO2 {current_row['spo2_clean']}% (S/F Ratio {int(current_row['sf_ratio_clean'])})\n"
                f"• Airway Care & Noise Context: {care_tag_summary} ({suppressed_so_far} suction/care blips damped)\n"
                f"• Surveillance Status: {status_line}"
            )
            st.code(macro_text, language="text")
        elif "Handoff" in ehr_system:
            handoff_text = build_12h_shift_handoff(df, selected_min, profile_meta, patient_care_logs, ack_info=ack_info)
            st.code(handoff_text, language="text")
        elif "Epic" in ehr_system:
            macro_text = (
                f".DELTAPULSE [EPIC SMARTPHRASE]\n"
                f"Time: {shift_clock} ({shift_badge}) | Bed: {profile_meta['bed']} ({selected_pid}, {profile_meta['age_sex']})\n"
                f"• Vent ({profile_meta['vent_mode']}): RR {current_row['rr_clean']} bpm, VTe {int(current_row['vte_clean'])} mL, MV {current_row['mv_clean']} L/m, RSBI {current_row['rsbi_clean']}, "
                f"PIP {current_row['pip_clean']} cmH2O, EtCO2 {current_row['etco2_clean']} mmHg, SpO2 {current_row['spo2_clean']}% (S/F {int(current_row['sf_ratio_clean'])})\n"
                f"• Hemodynamics: BP {int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])} mmHg (PP: {current_row['pulse_pressure_clean']}), "
                f"HR {current_row['hr_clean']} bpm, Temp ΔT {current_row['temp_gradient_clean']}°C, UO {int(current_row['urine_output_clean'])} mL/h\n"
                f"• Lines: {profile_meta.get('line_status', 'Patent')}\n"
                f"• Care Log: {care_tag_summary} ({suppressed_so_far} spikes filtered)\n"
                f"• Status: {status_line}"
            )
            st.code(macro_text, language="text")
        elif "Cerner" in ehr_system:
            macro_text = (
                f",,deltapulse [CERNER IVIEW AUTO-TEXT]\n"
                f"BED: {profile_meta['bed']} ({selected_pid}) | TIME: {shift_clock} ({shift_badge}) | VENT: {profile_meta['vent_mode']}\n"
                f"RESP: RR {current_row['rr_clean']} | VTe {int(current_row['vte_clean'])} mL | MV {current_row['mv_clean']} L/m | RSBI {current_row['rsbi_clean']} | "
                f"PIP {current_row['pip_clean']} | EtCO2 {current_row['etco2_clean']} | SpO2 {current_row['spo2_clean']}% (S/F {int(current_row['sf_ratio_clean'])})\n"
                f"HEMO: BP {int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])} (PP {current_row['pulse_pressure_clean']}) | HR {current_row['hr_clean']} | ΔT {current_row['temp_gradient_clean']}°C | UO {int(current_row['urine_output_clean'])} mL/h\n"
                f"LINES: {profile_meta.get('line_status', 'Patent')}\n"
                f"LOG: {care_tag_summary} ({suppressed_so_far} spikes filtered) | {status_line}"
            )
            st.code(macro_text, language="text")
        else:
            fhir_bundle = {
                "resourceType": "Bundle",
                "type": "transaction",
                "timestamp": str(current_row["timestamp"]),
                "subject": {"reference": f"Patient/{selected_pid}", "display": profile_meta["name"]},
                "metrics_clean_5m_median": {
                    "rr_bpm": float(current_row["rr_clean"]),
                    "vte_ml": float(current_row["vte_clean"]),
                    "mv_l_min": float(current_row["mv_clean"]),
                    "rsbi": float(current_row["rsbi_clean"]),
                    "pip_cmh2o": float(current_row["pip_clean"]),
                    "etco2_mmhg": float(current_row["etco2_clean"]),
                    "spo2_pct": float(current_row["spo2_clean"]),
                    "sf_ratio": float(current_row["sf_ratio_clean"]),
                    "bp_mmhg": {"sbp": float(current_row["sbp_clean"]), "dbp": float(current_row["dbp_clean"]), "pp": float(current_row["pulse_pressure_clean"])},
                    "hr_bpm": float(current_row["hr_clean"]),
                    "uo_ml_hr": float(current_row["urine_output_clean"])
                },
                "surveillance": {
                    "sqi_status": str(current_row.get("sqi_status", "GOOD")),
                    "alert_active": bool(current_row["pre_ehr_alert"]),
                    "acknowledged": ack_info,
                    "coil_active": bool(current_row.get("phys_squeeze_active", False)),
                    "coil_pct": float(current_row["phys_squeeze_score"]),
                    "ml_risk_pct": float(current_row["prob_random"]),
                    "care_log": [f"{t['time_str']} {t['label']}: {t['detail']}" if t['label'] == 'Bedside Note' else f"{t['time_str']} {t['label']}" for t in logged_tags_upto[-5:]]
                }
            }
            st.code(json.dumps(fhir_bundle, indent=2), language="json")

with st.expander("🔍 Raw Audit Data (Last 15 Min)", expanded=False):
    st.dataframe(window_df[["timestamp", "patient_id", "rr_clean", "vte_clean", "rsbi_clean", "pip_clean", "etco2_clean", "spo2_clean", "sf_ratio_clean", "sqi_status", "prob_random", "pre_ehr_alert"]].tail(15).sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

if st.session_state.is_playing:
    if st.session_state.selected_min < SHIFT_MINUTES - 1:
        time.sleep(0.8)
        st.session_state.advance_tick = True
        st.rerun()
    else:
        st.session_state.is_playing = False
        st.rerun()
