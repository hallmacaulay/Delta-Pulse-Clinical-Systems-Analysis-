import os
import json
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
    DB_NAME
)
from ml_benchmark import run_multi_model_benchmark, render_ml_benchmark_tab

st.set_page_config(
    page_title="DeltaPulse | 12-Bed ICU Central Station & Velocity Radar",
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
if "selected_min" not in st.session_state:
    st.session_state.selected_min = 305
if "care_logs" not in st.session_state:
    st.session_state.care_logs = {}
for pid in PATIENT_PROFILES.keys():
    st.session_state.care_logs.setdefault(pid, [])


def set_patient(pid: str):
    st.session_state.selected_pid = pid


def jump_to_minute(m: int):
    st.session_state.selected_min = int(m)


top_h1, top_h2, top_h3 = st.columns([2.5, 1.1, 0.9])
top_h1.markdown("### ⚡ DeltaPulse: 12-Bed ICU Central Station & Pre-EHR Velocity Radar")
with top_h2:
    philips_mode = st.toggle("🖥️ Philips IntelliVue Hardware Neon", value=False)
with top_h3:
    if st.button("🔄 Regenerate Telemetry", use_container_width=True):
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
    .bed-tile {{ background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 9px; padding: 9px 11px; margin-bottom: 5px; }}
    .bed-active {{ border: 2px solid {active_border} !important; box-shadow: 0 0 12px {active_glow}; }}
    .vital-card {{ background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }}
    .status-normal {{ border-left: 5px solid {c_norm}; }}
    .status-drift {{ border-left: 5px solid {c_drift}; background-color: rgba(245, 158, 11, 0.06); }}
    .status-critical {{ border-left: 5px solid {c_crit}; background-color: rgba(239, 68, 68, 0.12); }}
    .vital-header {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.80rem; color: #9CA3AF; font-weight: 600; text-transform: uppercase; }}
    .vital-value {{ font-family: monospace; font-size: 1.75rem; font-weight: 800; color: #F9FAFB; margin: 4px 0; }}
    .vital-sub {{ font-size: 0.78rem; color: #D1D5DB; }}
    .pill-normal {{ background-color: rgba(16, 185, 129, 0.18); color: {c_norm}; padding: 2px 7px; border-radius: 999px; font-size: 0.68rem; font-weight: 700; }}
    .pill-drift {{ background-color: rgba(245, 158, 11, 0.20); color: {c_drift}; padding: 2px 7px; border-radius: 999px; font-size: 0.68rem; font-weight: 700; }}
    .pill-critical {{ background-color: rgba(239, 68, 68, 0.25); color: {c_crit}; padding: 2px 7px; border-radius: 999px; font-size: 0.68rem; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

current_min = st.session_state.selected_min
snapshot_rows = {}
alert_count, artifact_count, stable_count = 0, 0, 0

for pid in PATIENT_PROFILES.keys():
    p_row = cohort_df[(cohort_df["patient_id"] == pid) & (cohort_df["minute_index"] == current_min)].iloc[0]
    snapshot_rows[pid] = p_row
    if p_row["pre_ehr_alert"]:
        alert_count += 1
    elif bool(p_row.get("artifact_suppressed", False)):
        artifact_count += 1
    else:
        stable_count += 1

pod_f1, pod_f2 = st.columns([2.1, 1.4])
with pod_f1:
    pod_filter = st.radio(
        "ICU Pod View:",
        ["All 12 Beds (3×4)", "Pod A (Beds 01–06)", "Pod B (Beds 07–12)", "🚨 Active Alerts Only"],
        horizontal=True,
        label_visibility="collapsed"
    )
with pod_f2:
    st.markdown(
        f"<div style='text-align:right; font-size:0.86rem; padding-top:4px;'>"
        f"🏥 <b>{len(PATIENT_PROFILES)} Beds</b> &nbsp;|&nbsp; "
        f"<span style='color:{c_crit}; font-weight:700;'>🚨 {alert_count} Alert</span> &nbsp;|&nbsp; "
        f"<span style='color:{c_drift}; font-weight:700;'>🛡️ {artifact_count} Filtered</span> &nbsp;|&nbsp; "
        f"<span style='color:{c_norm}; font-weight:700;'>🟢 {stable_count} Stable</span></div>",
        unsafe_allow_html=True
    )

visible_pids = []
for pid, meta in PATIENT_PROFILES.items():
    if pod_filter == "Pod A (Beds 01–06)" and meta.get("pod") != "Pod A":
        continue
    if pod_filter == "Pod B (Beds 07–12)" and meta.get("pod") != "Pod B":
        continue
    if pod_filter == "🚨 Active Alerts Only" and not bool(snapshot_rows[pid]["pre_ehr_alert"]):
        continue
    visible_pids.append(pid)

if not visible_pids:
    st.info("No beds have an active Pre-EHR alert at this shift minute. Switch to 'All 12 Beds' or scrub to Late Shift (04:05 AM).")
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
                            <strong style="font-size:0.86rem;">{meta['bed']} ({pid})</strong>
                            {badge_html}
                        </div>
                        <div style="font-size:0.75rem; color:#9CA3AF; margin:3px 0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{short_label}</div>
                        <div style="font-family:monospace; font-size:0.80rem; color:#E5E7EB;">
                            RR:<b>{p_row['rr_clean']}</b> | PIP:<b>{p_row['pip_clean']}</b> | BP:<b>{int(p_row['sbp_clean'])}/{int(p_row['dbp_clean'])}</b> | <b>{p_row['prob_random']}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.button(
                    f"{'✅ Viewing ' if is_active else 'Select '}{meta['bed']}",
                    key=f"btn_{pid}",
                    on_click=set_patient,
                    args=(pid,),
                    use_container_width=True
                )

selected_pid = st.session_state.selected_pid
profile_meta = PATIENT_PROFILES[selected_pid]
df = cohort_df[cohort_df["patient_id"] == selected_pid].reset_index(drop=True)

with st.container(border=True):
    ctrl1, ctrl2, ctrl3 = st.columns([2.2, 1.5, 1.3])

    with ctrl1:
        selected_min = st.slider("⏱️ Night-Shift Telemetry Timeline Scrubber (11:00 PM – 04:59 AM)", min_value=30, max_value=359, key="selected_min")
        current_row = df.iloc[selected_min]
        shift_clock = current_row["timestamp"].strftime("%I:%M %p")
        hours_in = selected_min // 60
        mins_in = selected_min % 60
        st.caption(f"**Active Shift Clock:** `{shift_clock}` ({hours_in}h {mins_in:02d}m into shift | Minute {selected_min})")

    with ctrl2:
        st.markdown("**⚡ Instant Scenario Checkpoints**")
        artifact_target = profile_meta["artifacts"][0] + 1
        b_col1, b_col2 = st.columns(2)
        b_col1.button(f"🛡️ Care Spike (m{artifact_target})", on_click=jump_to_minute, args=(artifact_target,), use_container_width=True)
        b_col2.button("🚨 Late Shift (04:05 AM)", on_click=jump_to_minute, args=(305,), use_container_width=True)

    with ctrl3:
        beginner_mode = st.toggle("🩺 New Clinician Mode (Plain-English)", value=True)
        legacy_fp = int(df.iloc[:240]["legacy_static_alert"].sum())
        deltapulse_fp = int(df.iloc[:240]["pre_ehr_alert"].sum())
        clicks_saved = int((selected_min / 60.0) * 12 * len(PATIENT_PROFILES))
        st.caption(
            f"🛡️ **4h Baseline False Alarms:** `{deltapulse_fp}` (DeltaPulse) vs. `{legacy_fp}` (Legacy)\n\n"
            f"⚡ **Manual EHR Flowsheet Clicks Bypassed ({len(PATIENT_PROFILES)}-Bed Unit):** `{clicks_saved} clicks`"
        )

st.markdown(
    f"**Active Bedside Feed:** `{profile_meta['bed']} ({selected_pid} • {profile_meta.get('pod', 'Pod A')})` — **{profile_meta['name']}** &nbsp;|&nbsp; "
    f"**Dx:** {profile_meta['diagnosis']} &nbsp;|&nbsp; **Vent:** `{profile_meta['vent_mode']}`\n\n"
    f"🔌 **ICU Line & Hardware Integrity:** `{profile_meta.get('line_status', 'All Lines Patent')}`"
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
        card_cls, pill_cls, pill_txt = "status-critical", "pill-critical", "🔴 SEVERE DEVIATION" if beginner else f"Z: {z_val:+g} SD"
    elif is_drift:
        card_cls, pill_cls, pill_txt = "status-drift", "pill-drift", "🟡 MILD DRIFT" if beginner else f"Z: {z_val:+g} SD"
    else:
        card_cls, pill_cls, pill_txt = "status-normal", "pill-normal", "🟢 AT BASELINE" if beginner else f"Z: {z_val:+g} SD"

    sub_text = f"Usual Baseline: <b>{base_mean} {unit}</b> | {vel_str}" if beginner else f"4h Mean: <b>{base_mean} {unit}</b> | Vel: <b>{vel_str}</b> | Z: <b>{z_val:+g} SD</b>"
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
        action_md = ("\n\n**🩺 Assessment-First Bedside Cues:**\n" + "\n".join([f"- {c}" for c in deduped_cues[:4]])) if beginner_mode and deduped_cues else ""
        st.error(f"🚨 **PRE-EHR DETERIORATION WARNING AT {shift_clock} (~2.5 HR LAG BYPASSED)**\n\n" + "\n".join(trigger_lines) + action_md)
    elif bool(current_row.get("artifact_suppressed", False)):
        st.warning(f"🛡️ **TRANSIENT CARE ARTIFACT SUPPRESSED AT {shift_clock}**\n\nRaw sensor spike/dip detected. 5-minute rolling median filter held baseline steady. **False alarm prevented.**")
    else:
        st.success(f"✅ **PHYSIOLOGICAL VELOCITY STABLE AT {shift_clock} — WITHIN PERSONAL BASELINE**")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown(render_vital_card("Clean Resp Rate (RR)", f"{current_row['rr_clean']} bpm", current_row["rr_zscore"], f"{current_row['dRR_dt_30m']:+g} bpm/30m", current_row["rr_base_mean"], "bpm", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Peak Inspiratory Press (PIP)", f"{current_row['pip_clean']} cmH₂O", current_row["pip_zscore"], f"{current_row['dPIP_dt_30m']:+g} cmH₂O/30m", current_row["pip_base_mean"], "cmH₂O", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Clean Systolic / Diastolic BP", f"{int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])} mmHg", current_row["sbp_zscore"], f"{current_row['dSBP_dt_30m']:+g} mmHg/30m", current_row["sbp_base_mean"], "mmHg", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Clean Heart Rate (HR)", f"{current_row['hr_clean']} bpm", current_row["hr_zscore"], f"Raw: {current_row['hr']} bpm", current_row["hr_base_mean"], "bpm", beginner_mode), unsafe_allow_html=True)

    with g_col2:
        st.markdown(render_vital_card("End-Tidal CO₂ (EtCO₂)", f"{current_row['etco2_clean']} mmHg", current_row["etco2_zscore"], f"{current_row['dEtCO2_dt_30m']:+g} mmHg/30m", current_row["etco2_base_mean"], "mmHg", beginner_mode), unsafe_allow_html=True)
        st.markdown(render_vital_card("Pulse Pressure (SBP - DBP)", f"{current_row['pulse_pressure_clean']} mmHg", current_row["pulse_pressure_zscore"], f"{current_row['dPP_dt_30m']:+g} mmHg/30m", current_row["pulse_pressure_base_mean"], "mmHg", beginner_mode, invert_low=True), unsafe_allow_html=True)
        st.markdown(render_vital_card("Core-Peripheral Temp ΔT", f"{current_row['temp_gradient_clean']} °C", current_row["temp_gradient_zscore"], "Micro-Shunting Index", current_row["temp_gradient_base_mean"], "°C", beginner_mode), unsafe_allow_html=True)

        uo_z = (current_row["urine_output_clean"] - 45.0) / 8.0
        spo2_uo_z = min(uo_z, float(current_row["spo2_zscore"])) if pd.notna(current_row["spo2_zscore"]) else uo_z

        st.markdown(render_vital_card("Clean SpO₂ & Urine Output", f"{current_row['spo2_clean']}% | {int(current_row['urine_output_clean'])} mL/h", spo2_uo_z, f"SpO₂ Z: {current_row['spo2_zscore']:+g} SD", 48.0, "mL/h", beginner_mode, invert_low=True), unsafe_allow_html=True)

with right_col:
    window_df = df.iloc[max(0, selected_min - 120):selected_min + 1].copy()
    r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs(["🫁 Waveforms", "🎯 Z-Score Tunnel", "🧠 ML Forecast", "📝 Care & Event Log"])

    with r_tab1:
        fig_air = go.Figure()
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["rr"], mode="lines", name="Raw RR", line=dict(color="rgba(156,163,175,0.45)", width=1.5, dash="dot")))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["rr_clean"], mode="lines", name="Clean RR (5m Median)", line=dict(color=c_cyan, width=3)))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pip"], mode="lines", name="Raw PIP", line=dict(color="rgba(248,113,113,0.40)", width=1.5, dash="dot")))
        fig_air.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pip_clean"], mode="lines", name="Clean PIP", line=dict(color=c_drift, width=3)))
        fig_air.update_layout(title="Airway Surveillance: Raw vs. Clean Median Filter", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg, height=295, margin=dict(l=10, r=10, t=35, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_air, use_container_width=True)

        fig_hemo = go.Figure()
        fig_hemo.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["sbp_clean"], mode="lines", name="Systolic BP", line=dict(color=c_crit, width=2.5)))
        fig_hemo.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["etco2_clean"], mode="lines", name="EtCO₂", line=dict(color=c_purple, width=2.5)))
        fig_hemo.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df["pulse_pressure_clean"], mode="lines", name="Pulse Pressure", line=dict(color=c_norm, width=2.5)))
        fig_hemo.update_layout(title="Hemodynamic & Alveolar Perfusion Trends", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg, height=285, margin=dict(l=10, r=10, t=35, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_hemo, use_container_width=True)

    with r_tab2:
        fig_z = go.Figure()
        fig_z.add_hrect(y0=-1.8, y1=1.8, fillcolor="rgba(16, 185, 129, 0.10)", line_width=0, annotation_text="Safe Baseline Zone")
        fig_z.add_hline(y=2.5, line_dash="dash", line_color=c_crit, annotation_text="+2.5 SD Alert")
        fig_z.add_hline(y=-2.0, line_dash="dash", line_color=c_drift, annotation_text="-2.0 SD Compression")
        for col, label, color in [("rr_zscore", "RR", c_cyan), ("pip_zscore", "PIP", c_drift), ("etco2_zscore", "EtCO₂", c_purple), ("sbp_zscore", "SBP (AD)", c_crit), ("pulse_pressure_zscore", "Pulse Pressure", c_norm), ("temp_gradient_zscore", "Temp ΔT", c_pink)]:
            fig_z.add_trace(go.Scatter(x=window_df["timestamp"], y=window_df[col], mode="lines", name=label, line=dict(color=color, width=2.2)))
        fig_z.update_layout(title="240-Minute Self-Calibrating Z-Score Trajectories", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg, height=540, margin=dict(l=10, r=10, t=35, b=10), yaxis_title="Z-Score (SD)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_z, use_container_width=True)

    with r_tab3:
        render_ml_benchmark_tab(st, df, summary_df, rf_importance_df, selected_min, philips_mode=philips_mode)

    with r_tab4:
        st.markdown(f"**🏷️ Quick-Tag Bedside Care / Artifact Event (`{shift_clock}`)**")
        note_col, clr_col = st.columns([3.0, 1.0])
        custom_note = note_col.text_input("Optional brief bedside note (non-PHI):", key=f"note_in_{selected_pid}", placeholder="e.g., Moderate thick secretions cleared; SpO2 rebounded")
        if clr_col.button("🗑️ Clear Tags", key=f"clr_{selected_pid}", use_container_width=True):
            st.session_state.care_logs[selected_pid] = []
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
                            "category": cat_name,
                            "label": tag_label,
                            "detail": custom_note.strip() if custom_note.strip() else f"Logged manually ({cat_name})",
                        })
                        st.rerun()

        st.markdown(f"**🧭 'What Just Happened' Shift Event Strip** &nbsp;|&nbsp; Stage 1 Suppressed Minutes: `{suppressed_so_far}`")
        strip_df = build_shift_event_strip(df, selected_min, st.session_state.care_logs[selected_pid])
        if strip_df.empty:
            st.info("No transient artifacts, trend alerts, or bedside care tags logged yet in this window.")
        else:
            st.dataframe(strip_df, use_container_width=True, hide_index=True, height=260)

    with st.expander("📋 Multi-EHR Interoperability Scribe: Epic / Cerner / HL7 FHIR (Bypass Click Fatigue)", expanded=False):
        st.caption("Vendor-agnostic documentation generator compatible with Epic Hyperspace, Oracle Health (Cerner PowerChart), and HL7 FHIR R4 Observation streams—scoped strictly to monitored parameters and logged care/artifact events.")
        ehr_system = st.radio(
            "Select Hospital EHR Target Format:",
            ["Epic Hyperspace (.dotphrase)", "Oracle Health / Cerner PowerChart (iView / Dynamic Doc)", "HL7 FHIR R4 JSON Payload (Universal API)"],
            horizontal=True
        )
        status_line = (
            f"🚨 PRE-EHR DETERIORATION ALERT ACTIVE (4h ML Risk: {current_row['prob_random']}%)"
            if current_row["pre_ehr_alert"]
            else f"🟢 Within 240-Min Personal Baseline (4h ML Risk: {current_row['prob_random']}%)"
        )
        logged_tags_upto = [t for t in patient_care_logs if int(t["minute_index"]) <= selected_min]
        care_tag_summary = (
            "; ".join(f"{t['time_str']} {t['label']}" for t in logged_tags_upto[-5:])
            if logged_tags_upto
            else "None manually tagged in window"
        )
        if "Epic" in ehr_system:
            macro_text = (
                f".DELTAPULSE [EPIC HYPERSPACE SMARTPHRASE — AUTO-FLOWSHEET]\n"
                f"Shift Timestamp: {shift_clock} | Unit Bed: {profile_meta['bed']} ({selected_pid} — {profile_meta['age_sex']})\n"
                f"• Resp / Vent ({profile_meta['vent_mode']}): Clean RR {current_row['rr_clean']} bpm (Z: {current_row['rr_zscore']:+g}, 30m Vel: {current_row['dRR_dt_30m']:+g}), "
                f"PIP {current_row['pip_clean']} cmH2O (Z: {current_row['pip_zscore']:+g}), EtCO2 {current_row['etco2_clean']} mmHg, SpO2 {current_row['spo2_clean']}%\n"
                f"• Hemodynamics: Clean BP {int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])} mmHg (PP: {current_row['pulse_pressure_clean']} mmHg, Z: {current_row['sbp_zscore']:+g}), "
                f"HR {current_row['hr_clean']} bpm, Core-Periph ΔT {current_row['temp_gradient_clean']}°C, UO {int(current_row['urine_output_clean'])} mL/hr\n"
                f"• Lines & Infusions: {profile_meta.get('line_status', 'Patent')}\n"
                f"• Noise & Care Context: {suppressed_so_far} transient raw noise minutes suppressed by 5m median | Care tags: {care_tag_summary}\n"
                f"• Velocity Radar: {status_line}"
            )
            st.code(macro_text, language="text")
        elif "Cerner" in ehr_system:
            macro_text = (
                f",,deltapulse [ORACLE HEALTH / CERNER POWERCHART AUTO-TEXT — IVIEW ADULT ICU]\n"
                f"========================================================================\n"
                f"BED/PT: {profile_meta['bed']} ({selected_pid}) | TIME: {shift_clock} | MODE: {profile_meta['vent_mode']}\n"
                f"[RESPIRATORY SUB-BAND] RR (5m Clean): {current_row['rr_clean']} bpm [Z={current_row['rr_zscore']:+g} SD] | "
                f"PIP: {current_row['pip_clean']} cmH2O [Z={current_row['pip_zscore']:+g} SD] | EtCO2: {current_row['etco2_clean']} mmHg | SpO2: {current_row['spo2_clean']}%\n"
                f"[HEMODYNAMIC SUB-BAND] ART/NIBP: {int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])} mmHg "
                f"(Pulse Press: {current_row['pulse_pressure_clean']} mmHg) | HR: {current_row['hr_clean']} bpm | ΔT: {current_row['temp_gradient_clean']}°C | I&O UO: {int(current_row['urine_output_clean'])} mL/hr\n"
                f"[LINES / TUBES / DRAINS] {profile_meta.get('line_status', 'Patent')}\n"
                f"[ARTIFACT / CARE CONTEXT] {suppressed_so_far} transient noise minutes suppressed | Logged tags: {care_tag_summary}\n"
                f"[CLINICAL DECISION SURVEILLANCE] {status_line}"
            )
            st.code(macro_text, language="text")
        else:
            fhir_bundle = {
                "resourceType": "Bundle",
                "type": "transaction",
                "timestamp": str(current_row["timestamp"]),
                "subject": {"reference": f"Patient/{selected_pid}", "display": profile_meta["name"]},
                "extension_ehr_targets": ["Epic Bridges / FHIR R4", "Oracle Health Cerner Millennium FHIR R4"],
                "metrics_clean_5m_median": {
                    "respiratory_rate_bpm": {"value": float(current_row["rr_clean"]), "z_score": float(current_row["rr_zscore"]), "velocity_30m": float(current_row["dRR_dt_30m"])},
                    "peak_inspiratory_pressure_cmh2o": {"value": float(current_row["pip_clean"]), "z_score": float(current_row["pip_zscore"])},
                    "etco2_mmhg": {"value": float(current_row["etco2_clean"]), "z_score": float(current_row["etco2_zscore"])},
                    "spo2_pct": {"value": float(current_row["spo2_clean"]), "z_score": float(current_row["spo2_zscore"])},
                    "blood_pressure_mmhg": {"sbp": float(current_row["sbp_clean"]), "dbp": float(current_row["dbp_clean"]), "pulse_pressure": float(current_row["pulse_pressure_clean"]), "sbp_z_score": float(current_row["sbp_zscore"])},
                    "heart_rate_bpm": float(current_row["hr_clean"]),
                    "urine_output_ml_hr": float(current_row["urine_output_clean"])
                },
                "clinical_surveillance": {
                    "pre_ehr_alert": bool(current_row["pre_ehr_alert"]),
                    "rf_4h_decompensation_risk_pct": float(current_row["prob_random"]),
                    "stage1_suppressed_artifact_minutes": suppressed_so_far,
                    "logged_bedside_care_tags": [f"{t['time_str']} {t['label']}" for t in logged_tags_upto[-5:]],
                    "line_integrity": profile_meta.get("line_status", "Patent")
                }
            }
            st.code(json.dumps(fhir_bundle, indent=2), language="json")

with st.expander("🔍 Audit Log (Last 15 Minutes)"):
    st.dataframe(window_df[["timestamp", "patient_id", "rr", "rr_clean", "rr_zscore", "sbp_clean", "sbp_zscore", "pip_clean", "pip_zscore", "artifact_suppressed", "prob_random", "pre_ehr_alert"]].tail(15).sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
