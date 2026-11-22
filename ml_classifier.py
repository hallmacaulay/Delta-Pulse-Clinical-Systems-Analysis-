import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from database_and_engine import (
from database_and_engine import PATIENT_PROFILES

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
    "urine_output_clean"
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
    "urine_output_clean": "Clean Hourly Urine Output (mL/hr)"
}

STABLE_CONTROL_PIDS = {
    pid for pid, meta in PATIENT_PROFILES.items() if meta.get("crash_type") == "none"
}


def prepare_training_cohort(cohort_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    clean_df = cohort_df.dropna(subset=FEATURE_COLUMNS).copy()
    y = (
        ((clean_df["minute_index"] >= 255) & (~clean_df["patient_id"].isin(STABLE_CONTROL_PIDS))) |
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
