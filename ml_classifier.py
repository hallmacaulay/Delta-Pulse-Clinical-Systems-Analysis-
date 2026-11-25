import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from database_and_engine import PATIENT_PROFILES, CRASH_START_MIN

FEATURE_COLUMNS = [
    "rr_zscore",
    "dRR_dt_30m",
    "pulse_pressure_zscore",
    "dPP_dt_30m",
    "sbp_zscore",
    "pip_zscore",
    "dPIP_dt_30m",
    "etco2_zscore",
    "dEtCO2_dt_30m",
    "spo2_zscore",
    "temp_gradient_zscore",
    "hr_zscore",
    "urine_output_clean",
    "phys_squeeze_score"
]

CLINICAL_FEATURE_LABELS = {
    "rr_zscore": "Resp Rate (RR)",
    "dRR_dt_30m": "RR 30m Speed",
    "pulse_pressure_zscore": "Pulse Pressure",
    "dPP_dt_30m": "Pulse Press Speed",
    "sbp_zscore": "Systolic BP",
    "pip_zscore": "Airway Press (PIP)",
    "dPIP_dt_30m": "PIP 30m Speed",
    "etco2_zscore": "EtCO₂ Level",
    "dEtCO2_dt_30m": "EtCO₂ 30m Speed",
    "spo2_zscore": "SpO₂ Level",
    "temp_gradient_zscore": "Temp Gap (ΔT)",
    "hr_zscore": "Heart Rate (HR)",
    "urine_output_clean": "Urine Output",
    "phys_squeeze_score": "Variability Coil"
}

STABLE_CONTROL_PIDS = {
    pid for pid, meta in PATIENT_PROFILES.items() if meta.get("crash_type") == "none"
}


def prepare_training_cohort(cohort_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    clean_df = cohort_df.dropna(subset=FEATURE_COLUMNS).copy()
    y = (
        ((clean_df["minute_index"] >= CRASH_START_MIN + 15) & (~clean_df["patient_id"].isin(STABLE_CONTROL_PIDS))) |
        (clean_df["pre_ehr_alert"] == True)
    ).astype(int)
    X = clean_df[FEATURE_COLUMNS]
    return X, y, clean_df


def train_deltapulse_rf(cohort_df: pd.DataFrame) -> tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    X, y, clean_df = prepare_training_cohort(cohort_df)
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42
    )
    rf_model.fit(X, y)

    cohort_df = cohort_df.copy()
    cohort_df["ml_risk_prob"] = 0.0
    probs = rf_model.predict_proba(X)[:, 1] * 100.0
    cohort_df.loc[clean_df.index, "ml_risk_prob"] = np.round(probs, 1)

    importance_df = pd.DataFrame({
        "feature_key": FEATURE_COLUMNS,
        "Clinical Parameter": [CLINICAL_FEATURE_LABELS[col] for col in FEATURE_COLUMNS],
        "Importance (%)": np.round(rf_model.feature_importances_ * 100.0, 1)
    }).sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)

    return rf_model, cohort_df, importance_df


def explain_minute_prediction(patient_df: pd.DataFrame, importance_df: pd.DataFrame, minute_idx: int) -> pd.DataFrame:
    row = patient_df.iloc[minute_idx]
    records = []

    for _, imp_row in importance_df.iterrows():
        col = imp_row["feature_key"]
        val = float(row[col]) if pd.notna(row[col]) else 0.0
        global_weight = imp_row["Importance (%)"] / 100.0

        if "zscore" in col:
            strain_magnitude = abs(val)
        elif "dt_30m" in col:
            strain_magnitude = abs(val) / 2.0
        elif col == "phys_squeeze_score":
            strain_magnitude = max(0.0, val / 35.0)
        else:
            strain_magnitude = max(0.0, (45.0 - val) / 10.0)

        local_impact = np.round(global_weight * strain_magnitude * 100.0, 1)
        records.append({
            "feature_key": col,
            "Clinical Parameter": imp_row["Clinical Parameter"],
            "Current Reading": round(val, 2),
            "RF Global Weight (%)": imp_row["Importance (%)"],
            "Active Driver Score": local_impact
        })

    return pd.DataFrame(records).sort_values(by="Active Driver Score", ascending=False).reset_index(drop=True)


def build_plain_english_why(row: pd.Series, local_drivers: pd.DataFrame, fc: dict) -> str:
    """Generates a single, crystal-clear plain-English sentence explaining WHY the ML model made its forecast."""
    # Check active clinical patterns first for intuitive bedside clarity
    reasons = []
    if float(row.get("sbp_zscore", 0)) > 2.5 and float(row.get("hr_zscore", 0)) < -1.2:
        reasons.append(f"BP spiked to {int(row['sbp_clean'])}/{int(row['dbp_clean'])} while HR dropped to {row['hr_clean']} bpm")
    if float(row.get("pip_zscore", 0)) <= -2.2:
        reasons.append(f"airway pressure (PIP) fell to {row['pip_clean']} cmH₂O (possible leak)")
    if float(row.get("pip_zscore", 0)) >= 2.2:
        reasons.append(f"airway pressure (PIP) climbed to {row['pip_clean']} cmH₂O")
    if float(row.get("etco2_zscore", 0)) >= 2.2:
        reasons.append(f"EtCO₂ rose to {row['etco2_clean']} mmHg")
    if float(row.get("rr_zscore", 0)) >= 2.2:
        reasons.append(f"breathing rate sped up to {row['rr_clean']} bpm")
    elif float(row.get("rr_zscore", 0)) <= -2.0:
        reasons.append(f"breathing rate slowed to {row['rr_clean']} bpm")
    if float(row.get("pulse_pressure_zscore", 0)) <= -1.8:
        reasons.append(f"pulse pressure narrowed to {row['pulse_pressure_clean']} mmHg")
    if float(row.get("spo2_zscore", 0)) <= -2.0:
        reasons.append(f"SpO₂ drifted down to {row['spo2_clean']}%")
    if float(row.get("urine_output_clean", 50)) < 32.0:
        reasons.append(f"urine output fell to {int(row['urine_output_clean'])} mL/h")
    if bool(row.get("phys_squeeze_active", False)) and not reasons:
        reasons.append(f"natural vital variability flattened ({row['phys_squeeze_score']}% coil)")

    if reasons:
        joined = " and ".join(reasons[:2]) if len(reasons) == 2 else ", ".join(reasons[:3])
        formatted = joined[0].upper() + joined[1:]
        return f"**Why:** {formatted} compared to this patient's 4-hour resting baseline."

    # Calm / stable baseline explanation
    top_driver = local_drivers.iloc[0]["Clinical Parameter"] if not local_drivers.empty else "all vitals"
    if bool(row.get("artifact_suppressed", False)):
        return "**Why:** Brief care spike was filtered out by the 5-minute median—underlying vitals remain steady."
    return f"**Why:** All vitals (including {top_driver}) are staying close to this patient's personal 4-hour baseline."


def forecast_regression_trajectory(
    patient_df: pd.DataFrame,
    selected_min: int,
    prob_col: str = "prob_random",
    horizon_mins: int = 120,
    critical_threshold: float = 80.0
) -> dict:
    row = patient_df.iloc[selected_min]
    current_prob = float(row.get(prob_col, 0.0))
    current_ts = pd.to_datetime(row["timestamp"])

    lookback = min(15, selected_min)
    if lookback >= 3:
        recent_probs = patient_df.iloc[selected_min - lookback:selected_min + 1][prob_col].astype(float).values
        x_hist = np.arange(len(recent_probs))
        prob_slope_per_min = float(np.polyfit(x_hist, recent_probs, 1)[0])
    else:
        prob_slope_per_min = 0.0

    z_strain = max(
        0.0,
        float(row.get("rr_zscore", 0.0)),
        float(row.get("pip_zscore", 0.0)),
        float(row.get("etco2_zscore", 0.0)),
        float(row.get("sbp_zscore", 0.0)),
        -float(row.get("pip_zscore", 0.0)),
        -float(row.get("pulse_pressure_zscore", 0.0)),
    )
    vel_activity = (
        abs(float(row.get("dRR_dt_30m", 0.0))) / 2.5 +
        abs(float(row.get("dPIP_dt_30m", 0.0))) / 2.5 +
        abs(float(row.get("dEtCO2_dt_30m", 0.0))) / 2.5 +
        abs(float(row.get("dSBP_dt_30m", 0.0))) / 10.0
    )
    squeeze_active = bool(row.get("phys_squeeze_active", False))

    if z_strain >= 1.3 and vel_activity >= 0.8 and prob_slope_per_min < 0.35:
        effective_slope = max(prob_slope_per_min, 0.45 * min(z_strain, 2.5))
    elif squeeze_active and prob_slope_per_min < 0.30:
        effective_slope = max(prob_slope_per_min, 0.42)
    elif z_strain < 1.2 and current_prob < 25.0 and not squeeze_active:
        effective_slope = min(prob_slope_per_min, 0.0)
    else:
        effective_slope = prob_slope_per_min

    future_steps = np.arange(0, horizon_mins + 5, 5)
    future_ts = [current_ts + pd.Timedelta(minutes=int(m)) for m in future_steps]

    raw_proj = current_prob + effective_slope * future_steps
    proj_mean = np.clip(np.round(raw_proj, 1), 0.2, 99.5)
    band_spread = np.sqrt(future_steps / 10.0) * 3.5
    proj_upper = np.clip(np.round(proj_mean + band_spread, 1), 0.2, 99.8)
    proj_lower = np.clip(np.round(proj_mean - band_spread, 1), 0.1, 99.0)

    eta_mins = None
    eta_clock = None
    if current_prob >= critical_threshold or bool(row.get("pre_ehr_alert", False)):
        eta_mins = 0
        eta_clock = current_ts.strftime("%I:%M %p")
        trajectory_status = "BREACHING NOW"
    else:
        cross_indices = np.where(proj_mean >= critical_threshold)[0]
        if len(cross_indices) > 0:
            eta_mins = int(future_steps[cross_indices[0]])
            eta_clock = (current_ts + pd.Timedelta(minutes=eta_mins)).strftime("%I:%M %p")
            trajectory_status = f"RISING (+{round(effective_slope * 60, 1)}%/hr)"
        elif squeeze_active or (effective_slope > 0.12 and proj_mean[-1] >= 45.0):
            trajectory_status = f"EARLY DRIFT (+{round(effective_slope * 60, 1)}%/hr)"
        else:
            trajectory_status = "STABLE"

    return {
        "current_prob": round(current_prob, 1),
        "slope_per_hr": round(effective_slope * 60.0, 1),
        "eta_mins": eta_mins,
        "eta_clock": eta_clock,
        "trajectory_status": trajectory_status,
        "future_ts": future_ts,
        "proj_mean": proj_mean,
        "proj_upper": proj_upper,
        "proj_lower": proj_lower,
        "critical_threshold": critical_threshold,
    }
