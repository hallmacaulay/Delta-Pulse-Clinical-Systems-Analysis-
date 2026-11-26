lxt="+2.5 SD")
        fig_z.add_hline(y=-2.0, line_dash="dash", line_color=c_drift, annotation_text="-2.0 SD")
        for col, label, color in [("rr_zscore", "RR", c_cyan), ("pip_zscore", "PIP", c_drift), ("etco2_zscore", "EtCO₂", c_purple), ("sbp_zscore", "SBP", c_crit), ("pulse_pressure_zscore", "Pulse Press", c_norm), ("temp_gradient_zscore", "Temp ΔT", c_pink)]:
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
        st.markdown(f"**📝 Add Bedside Note or Care Tag (`{shift_clock}`)**")
        with st.form(key=f"note_form_{selected_pid}", clear_on_submit=True):
            nf_col1, nf_col2 = st.columns([3.0, 1.0])
            note_input = nf_col1.text_input(
                "Note",
                placeholder="Type brief non-PHI bedside note...",
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
                            "type": "Care Tag",
                            "category": cat_name,
                            "label": tag_label,
                            "detail": cat_name,
                        })
                        st.rerun()

        h_strip1, h_strip2 = st.columns([3.0, 1.0])
        h_strip1.markdown(f"**🧭 Shift Event Timeline** • Filtered Blips: `{suppressed_so_far}`")
        if h_strip2.button("🗑️ Clear", key=f"clr_{selected_pid}", use_container_width=True):
            st.session_state.care_logs[selected_pid] = []
            st.rerun()

        strip_df = build_shift_event_strip(df, selected_min, st.session_state.care_logs[selected_pid])
        if strip_df.empty:
            st.info("No events or notes logged yet.")
        else:
            st.dataframe(strip_df, use_container_width=True, hide_index=True, height=255)

    with st.expander("📋 EHR Scribe (Epic / Cerner / FHIR)", expanded=False):
        ehr_system = st.radio(
            "Target Format:",
            ["Epic (.dotphrase)", "Cerner (iView)", "HL7 FHIR JSON"],
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
        if "Epic" in ehr_system:
            macro_text = (
                f".DELTAPULSE [EPIC SMARTPHRASE]\n"
                f"Time: {shift_clock} ({shift_badge}) | Bed: {profile_meta['bed']} ({selected_pid}, {profile_meta['age_sex']})\n"
                f"• Vent ({profile_meta['vent_mode']}): RR {current_row['rr_clean']} bpm (Z: {current_row['rr_zscore']:+g}), "
                f"PIP {current_row['pip_clean']} cmH2O (Z: {current_row['pip_zscore']:+g}), EtCO2 {current_row['etco2_clean']} mmHg, SpO2 {current_row['spo2_clean']}%\n"
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
                f"RESP: RR {current_row['rr_clean']} bpm (Z={current_row['rr_zscore']:+g}) | PIP {current_row['pip_clean']} cmH2O (Z={current_row['pip_zscore']:+g}) | EtCO2 {current_row['etco2_clean']} mmHg | SpO2 {current_row['spo2_clean']}%\n"
                f"HEMO: BP {int(current_row['sbp_clean'])}/{int(current_row['dbp_clean'])} mmHg (PP {current_row['pulse_pressure_clean']}) | HR {current_row['hr_clean']} bpm | ΔT {current_row['temp_gradient_clean']}°C | UO {int(current_row['urine_output_clean'])} mL/h\n"
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
                    "rr_bpm": {"value": float(current_row["rr_clean"]), "z": float(current_row["rr_zscore"])},
                    "pip_cmh2o": {"value": float(current_row["pip_clean"]), "z": float(current_row["pip_zscore"])},
                    "etco2_mmhg": {"value": float(current_row["etco2_clean"]), "z": float(current_row["etco2_zscore"])},
                    "spo2_pct": {"value": float(current_row["spo2_clean"]), "z": float(current_row["spo2_zscore"])},
                    "bp_mmhg": {"sbp": float(current_row["sbp_clean"]), "dbp": float(current_row["dbp_clean"]), "pp": float(current_row["pulse_pressure_clean"])},
                    "hr_bpm": float(current_row["hr_clean"]),
                    "uo_ml_hr": float(current_row["urine_output_clean"])
                },
                "surveillance": {
                    "alert_active": bool(current_row["pre_ehr_alert"]),
                    "coil_active": bool(current_row.get("phys_squeeze_active", False)),
                    "coil_pct": float(current_row["phys_squeeze_score"]),
                    "ml_risk_pct": float(current_row["prob_random"]),
                    "care_log": [f"{t['time_str']} {t['label']}: {t['detail']}" if t['label'] == 'Bedside Note' else f"{t['time_str']} {t['label']}" for t in logged_tags_upto[-5:]]
                }
            }
            st.code(json.dumps(fhir_bundle, indent=2), language="json")

with st.expander("🔍 Raw Audit Data (Last 15 Min)", expanded=False):
    st.dataframe(window_df[["timestamp", "patient_id", "rr", "rr_clean", "rr_zscore", "sbp_clean", "sbp_zscore", "pip_clean", "pip_zscore", "phys_squeeze_score", "artifact_suppressed", "prob_random", "pre_ehr_alert"]].tail(15).sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

# Live 24-Hour Clock Playback Loop
if st.session_state.is_playing:
    if st.session_state.selected_min < SHIFT_MINUTES - 1:
        time.sleep(0.8)
        st.session_state.advance_tick = 
        
