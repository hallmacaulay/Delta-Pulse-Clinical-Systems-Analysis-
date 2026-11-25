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
    "rr_zscore": "Resp Rate Personal Z-Score (SD)",
    "dRR_dt_30m": "Resp Rate 30m Velocity (dRR/dt)",
    "pulse_pressure_zscore": "Pulse Pressure Compression Z-Score",
    "dPP_dt_30m": "Pulse Pressure 30m Velocity",
    "sbp_zscore": "Systolic BP Personal Z-Score (AD Radar)",
    "pip_zscore": "Peak Inspiratory Pressure (PIP) Z-Score",
    "dPIP_dt_30m": "PIP 30m Velocity (Airway Resistance)",
    "etco2_zscore": "End-Tidal CO₂ (EtCO₂) Z-Score",
    "dEtCO2_dt_30m": "EtCO₂ 30m Velocity (Alveolar Vent)",
    "spo2_zscore": "SpO₂ Personal Z-Score (Oxygenation)",
    "temp_gradient_zscore": "Core-Peripheral Temp Gradient (ΔT) Z-Score",
    "hr_zscore": "Heart Rate Personal Z-Score",
    "urine_output_clean": "Clean Hourly Urine Output (mL/hr)",
    "phys_squeeze_score": "Physiological Squeeze Index (Variability Coil %)"
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
            "Clinical Parameter": imp_row["Clinical Parameter"],
            "Current Reading": round(val, 2),
            "RF Global Weight (%)": imp_row["Importance (%)"],
            "Active Driver Score": local_impact
        })

    return pd.DataFrame(records).sort_values(by="Active Driver Score", ascending=False).reset_index(drop=True)


def forecast_regression_trajectory(
    patient_df: pd.DataFrame,
    selected_min: int,
    prob_col: str = "prob_random",
    horizon_mins: int = 120,
    critical_threshold: float = 80.0
) -> dict:
    """
    Projects forward decompensation risk over the next `horizon_mins` using
    recent 15-minute risk velocity + physiological Z-score drift & Squeeze momentum,
    estimating Time-to-Threshold (ETA) before critical regression.
    """
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
        trajectory_status = "ACTIVE REGRESSION BREACH"
    else:
        cross_indices = np.where(proj_mean >= critical_threshold)[0]
        if len(cross_indices) > 0:
            eta_mins = int(future_steps[cross_indices[0]])
            eta_clock = (current_ts + pd.Timedelta(minutes=eta_mins)).strftime("%I:%M %p")
            trajectory_status = f"ACCELERATING (+{round(effective_slope * 60, 1)}%/hr)"
        elif squeeze_active or (effective_slope > 0.12 and proj_mean[-1] >= 45.0):
            trajectory_status = f"PRE-BREAKOUT COIL / DRIFT (+{round(effective_slope * 60, 1)}%/hr)"
        else:
            trajectory_status = "STABLE / FLAT TRAJECTORY"

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
