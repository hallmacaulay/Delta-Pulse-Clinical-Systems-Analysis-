import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from database_and_engine import CHECKPOINT_LATE_MIN
from ml_classifier import (
    FEATURE_COLUMNS,
    CLINICAL_FEATURE_LABELS,
    prepare_training_cohort,
    explain_minute_prediction,
    forecast_regression_trajectory,
    build_plain_english_why
)

TOUCH_CHART_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoom2d"]
}


def build_benchmark_models() -> dict:
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", random_state=42)
        )
    }

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        )
    else:
        models["Gradient Boosted"] = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        )

    return models


def run_multi_model_benchmark(cohort_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    X, y, clean_df = prepare_training_cohort(cohort_df)
    models = build_benchmark_models()
    cohort_df = cohort_df.copy()

    benchmark_rows = []
    feature_weights = {}
    rf_importance_df = None

    for name, model in models.items():
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.50).astype(int)

        col_name = f"prob_{name.split()[0].lower()}"
        cohort_df[col_name] = 0.0
        cohort_df.loc[clean_df.index, col_name] = np.round(probs * 100.0, 1)

        auroc = roc_auc_score(y, probs)
        ppv = precision_score(y, preds, zero_division=0)
        sensitivity = recall_score(y, preds, zero_division=0)
        brier = brier_score_loss(y, probs)
        nne = round(1.0 / ppv, 2) if ppv > 0 else np.nan

        v1_slice = cohort_df[cohort_df["patient_id"] == "VENT-01"]
        suction_prob = v1_slice.loc[v1_slice["minute_index"].isin([120, 121]), col_name].max()
        crash_prob = v1_slice.loc[v1_slice["minute_index"] == CHECKPOINT_LATE_MIN, col_name].values[0]

        benchmark_rows.append({
            "Model": name,
            "AUROC": round(auroc, 3),
            "Precision": f"{round(ppv * 100, 1)}%",
            "Sensitivity": f"{round(sensitivity * 100, 1)}%",
            "Alerts per Catch (NNE)": nne,
            "Brier Loss": round(brier, 4),
            "Suction Spike": f"{suction_prob}%",
            "Late Crash (06:05 AM)": f"{crash_prob}%",
            "prob_col": col_name
        })

        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
            norm_imp = (raw_imp / np.sum(raw_imp)) * 100.0
            if "Random Forest" in name:
                rf_importance_df = pd.DataFrame({
                    "feature_key": FEATURE_COLUMNS,
                    "Clinical Parameter": [CLINICAL_FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
                    "Importance (%)": np.round(norm_imp, 1)
                }).sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)
        elif hasattr(model, "named_steps"):
            coefs = np.abs(model.named_steps["logisticregression"].coef_[0])
            norm_imp = (coefs / np.sum(coefs)) * 100.0
        else:
            corr_imp = np.array([abs(np.corrcoef(X[c], probs)[0, 1]) for c in FEATURE_COLUMNS])
            corr_imp = np.nan_to_num(corr_imp, nan=0.0)
            denom = np.sum(corr_imp) if np.sum(corr_imp) > 0 else 1.0
            norm_imp = (corr_imp / denom) * 100.0

        feature_weights[name] = pd.DataFrame({
            "Clinical Parameter": [CLINICAL_FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
            "Relative Weight (%)": np.round(norm_imp, 1)
        }).sort_values(by="Relative Weight (%)", ascending=False).reset_index(drop=True)

    summary_df = pd.DataFrame(benchmark_rows)
    return cohort_df, summary_df, feature_weights, rf_importance_df


def render_ml_benchmark_tab(st, patient_df: pd.DataFrame, summary_df: pd.DataFrame, rf_importance_df: pd.DataFrame, selected_min: int, philips_mode: bool = False):
    current_row = patient_df.iloc[selected_min]
    current_ts = pd.to_datetime(current_row["timestamp"])
    pid = str(current_row.get("patient_id", "VENT-01"))

    fc = forecast_regression_trajectory(patient_df, selected_min, prob_col="prob_random", horizon_mins=120, critical_threshold=80.0)
    local_drivers = explain_minute_prediction(patient_df, rf_importance_df, selected_min).head(5)
    why_sentence = build_plain_english_why(current_row, local_drivers, fc)

    # Persist initial predicted ETA in session_state while active so it never vanishes on the next tick
    eta_store = st.session_state.setdefault("latched_eta_memory", {})
    if fc["latched_state"] == "STABLE":
        eta_store.pop(pid, None)
    elif fc["eta_mins"] is not None and fc["eta_mins"] > 0:
        if pid not in eta_store:
            eta_store[pid] = {
                "initial_eta_clock": fc["eta_clock"],
                "initial_eta_mins": fc["eta_mins"],
                "detected_at_clock": current_ts.strftime("%I:%M %p"),
            }
    saved_eta = eta_store.get(pid)

    # Clear, high-contrast Forecast Verdict + Simple "Why" + Persistent Predicted Time
    if fc["eta_mins"] == 0:
        prior_note = (
            f" *(First predicted at {saved_eta['detected_at_clock']} for ~{saved_eta['initial_eta_clock']})*"
            if saved_eta else ""
        )
        st.error(
            f"🚨 **HIGH RISK NOW ({fc['current_prob']}%) — Critical Threshold Crossed**{prior_note}\n\n"
            f"{why_sentence}"
        )
    elif fc["eta_mins"] is not None:
        st.warning(
            f"⏳ **ESTIMATED BREACH IN ~{fc['eta_mins']} MIN (BY {fc['eta_clock']})** — "
            f"Risk: `{fc['current_prob']}%` ({fc['trajectory_status']})\n\n"
            f"{why_sentence}"
        )
    elif "DRIFT" in fc["trajectory_status"]:
        st.info(
            f"📈 **EARLY DRIFT ({fc['trajectory_status']})** — "
            f"2h Projected Risk: `{fc['proj_mean'][-1]}%`\n\n"
            f"{why_sentence}"
        )
    else:
        st.success(
            f"🟢 **LOW RISK / STABLE ({fc['current_prob']}%)** — "
            f"No Breach Expected in Next 2 Hours\n\n"
            f"{why_sentence}"
        )

    m1, m2, m3 = st.columns(3)
    for idx, (_, m_row) in enumerate(summary_df.iterrows()):
        col_ui = [m1, m2, m3][idx]
        prob_val = current_row[m_row["prob_col"]]
        badge = "HIGH RISK" if prob_val >= 65.0 else ("MODERATE" if prob_val >= 35.0 else "STABLE")
        col_ui.metric(
            label=m_row["Model"],
            value=f"{prob_val}%",
            delta=badge,
            delta_color="inverse" if prob_val >= 50.0 else "normal"
        )

    window_df = patient_df.iloc[max(30, selected_min - 90):selected_min + 1]

    fig_prob = go.Figure()
    colors = ["#00FF66", "#00E5FF", "#FFD600"] if philips_mode else ["#10B981", "#38BDF8", "#F59E0B"]
    plot_bg = "rgba(5,5,5,0.92)" if philips_mode else "rgba(17,24,39,0.6)"
    alert_line = "#FF2A2A" if philips_mode else "#EF4444"
    bar_col = "#00E5FF" if philips_mode else "#38BDF8"

    for idx, (_, m_row) in enumerate(summary_df.iterrows()):
        fig_prob.add_trace(go.Scatter(
            x=window_df["timestamp"], y=window_df[m_row["prob_col"]],
            mode="lines", name=m_row["Model"],
            line=dict(color=colors[idx % len(colors)], width=2.5)
        ))

    fig_prob.add_trace(go.Scatter(
        x=fc["future_ts"] + fc["future_ts"][::-1],
        y=list(fc["proj_upper"]) + list(fc["proj_lower"][::-1]),
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.14)" if fc["eta_mins"] is not None else "rgba(56, 189, 248, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Confidence Range (+2h)"
    ))
    fig_prob.add_trace(go.Scatter(
        x=fc["future_ts"],
        y=fc["proj_mean"],
        mode="lines",
        name="2-Hour Forecast",
        line=dict(color="#FF2A2A" if philips_mode else "#F87171", width=2.5, dash="dot")
    ))

    fig_prob.add_vline(x=current_ts, line_dash="solid", line_color="#9CA3AF", line_width=1.5)
    if fc["eta_mins"] is not None and fc["eta_mins"] > 0:
        eta_ts = current_ts + pd.Timedelta(minutes=fc["eta_mins"])
        fig_prob.add_trace(go.Scatter(
            x=[eta_ts], y=[80.0],
            mode="markers+text",
            name=f"ETA {fc['eta_clock']}",
            text=[f"ETA {fc['eta_clock']}"],
            textposition="top center",
            marker=dict(color="#FF2A2A", size=11, symbol="diamond")
        ))

    fig_prob.add_hline(y=80.0, line_dash="dash", line_color=alert_line, annotation_text="80% Alert Line")
    fig_prob.update_layout(
        title="2-Hour Risk Forecast (%)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg,
        dragmode="pan",
        height=300, margin=dict(l=10, r=10, t=38, b=58), yaxis=dict(range=[0, 108], title="Risk (%)"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_prob, use_container_width=True, config=TOUCH_CHART_CONFIG)

    fig_imp = go.Figure(go.Bar(
        x=local_drivers["Active Driver Score"][::-1],
        y=local_drivers["Clinical Parameter"][::-1],
        orientation="h", marker_color=bar_col,
        text=local_drivers["Current Reading"][::-1], textposition="auto"
    ))
    fig_imp.update_layout(
        title="Top 5 Vitals Driving This Score",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=plot_bg,
        dragmode="pan",
        height=225, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="Driver Impact"
    )
    st.plotly_chart(fig_imp, use_container_width=True, config=TOUCH_CHART_CONFIG)

    with st.expander("📊 Model Accuracy Comparison (Cohort Stats)", expanded=False):
        display_cols = [c for c in summary_df.columns if c != "prob_col"]
        st.dataframe(summary_df[display_cols], use_container_width=True, hide_index=True)
