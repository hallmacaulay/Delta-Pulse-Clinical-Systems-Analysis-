import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

DB_NAME = "aeropulse_vitals.db"
SHIFT_MINUTES = 1440       # Full 24-hour cycle (07:00 AM - 06:59 AM)
CRASH_START_MIN = 1320     # 05:00 AM (22h into 24h cycle; 2h ramp to 06:59 AM)
CHECKPOINT_LATE_MIN = 1385 # 06:05 AM (65 mins into ramp)

MIN_STD_FLOOR: Dict[str, float] = {
    "rr": 0.25,
    "hr": 0.80,
    "spo2": 0.25,
    "sbp": 1.00,
    "pulse_pressure": 1.00,
    "etco2": 0.35,
    "pip": 0.35,
    "temp_gradient": 0.05,
}

ARTIFACT_SPIKE_THRESHOLD: Dict[str, float] = {
    "rr": 5.0,
    "hr": 12.0,
    "spo2": 3.5,
    "sbp": 18.0,
    "pip": 5.0,
}

ARTIFACT_TAGS: Dict[str, List[str]] = {
    "Airway / Circuit": [
        "Suctioning",
        "Circuit / Condensate Clear",
        "Cuff Check",
        "Inline Nebulizer",
    ],
    "Patient Activity": [
        "Turning / Repositioning",
        "Coughing / Asynchrony",
        "Oral Care",
    ],
    "Hardware": [
        "Sensor / Probe Blip",
    ],
}

PATIENT_PROFILES = {
    "VENT-01": {
        "bed": "Bed 01",
        "pod": "Pod A",
        "name": "Patient A — Chronic COPD / Trach-Vent",
        "age_sex": "68M",
        "diagnosis": "End-Stage COPD, Chronic Hypercapnic Resp Failure (Inline Trach)",
        "vent_mode": "AC/VC | TV 450 | PEEP 8 | FiO2 35%",
        "line_status": "20G PIV (R Forearm) | Inline Trach Suction & Heated Circuit",
        "clinical_note": (
            "Chronically elevated baseline (RR ~22, EtCO2 ~52, SpO2 ~90.5%). "
            "Routine suctioning blips, pre-breakout coil at 04:30 AM, then mucus plugging & sepsis after 05:00 AM."
        ),
        "baseline": {
            "rr": (22.0, 0.7), "hr": (86.0, 1.8), "spo2": (90.5, 0.5),
            "sbp": (122.0, 2.0), "dbp": (78.0, 1.5), "uo": (45.0, 3.0),
            "etco2": (52.0, 0.9), "pip": (24.0, 0.9), "temp_grad": (1.8, 0.12)
        },
        "artifacts": [120, 480, 960, 1180],
        "crash_type": "mucus_sepsis"
    },
    "VENT-02": {
        "bed": "Bed 02",
        "pod": "Pod A",
        "name": "Patient B — C4 Tetraplegia / Neurogenic Baseline",
        "age_sex": "42M",
        "diagnosis": "High-Cervical SCI (C4 AIS-A), Trach-Vent Dependent",
        "vent_mode": "SIMV-PC | PIP 18 | PEEP 5 | FiO2 28%",
        "line_status": "Radial ART Line | Foley Catheter (Monitor for Kinks / AD Trigger)",
        "clinical_note": (
            "Low resting BP (~94/60) & HR (~56). Turning artifacts filtered; "
            "Autonomic Dysreflexia crisis (BP surge + reflex bradycardia) after 05:00 AM."
        ),
        "baseline": {
            "rr": (14.5, 0.5), "hr": (56.0, 1.4), "spo2": (97.0, 0.4),
            "sbp": (94.0, 1.8), "dbp": (60.0, 1.2), "uo": (52.0, 3.5),
            "etco2": (39.0, 0.7), "pip": (18.0, 0.7), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [110, 450, 920, 1150],
        "crash_type": "autonomic_dysreflexia"
    },
    "VENT-03": {
        "bed": "Bed 03",
        "pod": "Pod A",
        "name": "Patient C — Post-ICU Trach Weaning (PSV)",
        "age_sex": "59F",
        "diagnosis": "Post-ARDS Prolonged Ventilation, Diaphragmatic Weakness",
        "vent_mode": "PSV 12 / PEEP 6 | FiO2 30%",
        "line_status": "Radial ART Line | 20G PIV x2 (K+ Rider Infusing)",
        "clinical_note": (
            "Stable baseline with suction blips, followed by diaphragmatic fatigue "
            "and rapid shallow breathing creep after 05:00 AM."
        ),
        "baseline": {
            "rr": (16.0, 0.6), "hr": (78.0, 1.6), "spo2": (95.5, 0.4),
            "sbp": (116.0, 1.9), "dbp": (74.0, 1.4), "uo": (48.0, 2.8),
            "etco2": (41.0, 0.8), "pip": (17.0, 0.6), "temp_grad": (1.6, 0.11)
        },
        "artifacts": [130, 540, 1020],
        "crash_type": "weaning_fatigue"
    },
    "VENT-04": {
        "bed": "Bed 04",
        "pod": "Pod A",
        "name": "Patient D — Stable ALS Vent Control (High Care Noise)",
        "age_sex": "71F",
        "diagnosis": "Neuromuscular Weakness (ALS), Stable Chronic Trach-Vent",
        "vent_mode": "AC/PC | PIP 20 | PEEP 6 | FiO2 25%",
        "line_status": "Cuff Cycling Q15m | PEG Tube Feeds 55 mL/hr",
        "clinical_note": (
            "Stable 24h control with frequent suctioning, turning, and a 1-min circuit-open dip (m220). "
            "0% false alarms."
        ),
        "baseline": {
            "rr": (18.0, 0.6), "hr": (74.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (120.0, 1.8), "dbp": (76.0, 1.3), "uo": (50.0, 3.0),
            "etco2": (40.0, 0.7), "pip": (20.0, 0.7), "temp_grad": (1.7, 0.10)
        },
        "artifacts": [90, 220, 510, 880, 1210],
        "crash_type": "none"
    },
    "VENT-05": {
        "bed": "Bed 05",
        "pod": "Pod A",
        "name": "Patient E — Post-Op AAA / Silent Opioid Hypoventilation",
        "age_sex": "74M",
        "diagnosis": "Post-Op Open AAA Repair, PCA Hydromorphone / Epidural",
        "vent_mode": "PSV 8 / PEEP 5 | FiO2 35%",
        "line_status": "R IJ Central Line | Radial ART Line | PCA & Epidural Active",
        "clinical_note": (
            "Silent CO2 retention after late-night opioid stacking: RR falls and EtCO2 climbs "
            "while supplemental O2 keeps SpO2 near 97%."
        ),
        "baseline": {
            "rr": (15.5, 0.5), "hr": (74.0, 1.5), "spo2": (97.5, 0.4),
            "sbp": (124.0, 1.8), "dbp": (76.0, 1.2), "uo": (50.0, 3.0),
            "etco2": (40.0, 0.7), "pip": (18.0, 0.6), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [105, 490, 950],
        "crash_type": "silent_hypovent"
    },
    "VENT-06": {
        "bed": "Bed 06",
        "pod": "Pod A",
        "name": "Patient F — Severe CAP / Cuff & Circuit Leak",
        "age_sex": "63F",
        "diagnosis": "Severe Community-Acquired Pneumonia, Trach-Vent Day 9",
        "vent_mode": "AC/VC | TV 420 | PEEP 10 | FiO2 40%",
        "line_status": "L PICC (Double Lumen) | Heated Wire Circuit & Subglottic Trach",
        "clinical_note": (
            "Gradual trach cuff deflation / circuit leak after 05:00 AM causing a sustained PIP drop."
        ),
        "baseline": {
            "rr": (16.5, 0.5), "hr": (80.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (118.0, 1.8), "dbp": (74.0, 1.3), "uo": (46.0, 2.8),
            "etco2": (41.0, 0.7), "pip": (22.0, 0.7), "temp_grad": (1.6, 0.10)
        },
        "artifacts": [140, 600, 1040],
        "crash_type": "circuit_leak"
    },
    "VENT-07": {
        "bed": "Bed 07",
        "pod": "Pod B",
        "name": "Patient G — Urosepsis / Early Septic Shock Drift",
        "age_sex": "79F",
        "diagnosis": "Obstructive Urosepsis, Low-Dose Norepinephrine (0.04 mcg/kg/min)",
        "vent_mode": "AC/PC | PIP 19 | PEEP 6 | FiO2 35%",
        "line_status": "R IJ Triple-Lumen CVC (Pressor) | Foley with Urometer",
        "clinical_note": (
            "Pre-breakout coil at 04:30 AM, then early septic shock drift: rising RR/HR, "
            "narrowing pulse pressure, widening Temp ΔT, and falling urine output."
        ),
        "baseline": {
            "rr": (17.0, 0.5), "hr": (82.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (114.0, 1.8), "dbp": (68.0, 1.3), "uo": (42.0, 2.5),
            "etco2": (36.0, 0.7), "pip": (19.0, 0.6), "temp_grad": (1.9, 0.11)
        },
        "artifacts": [115, 520, 990],
        "crash_type": "septic_shock_drift"
    },
    "VENT-08": {
        "bed": "Bed 08",
        "pod": "Pod B",
        "name": "Patient H — Status Asthmaticus / Secretion & Resistance",
        "age_sex": "34F",
        "diagnosis": "Refractory Status Asthmaticus, Intubated with Inline Bronchodilators",
        "vent_mode": "AC/VC | TV 440 | PEEP 5 | FiO2 45%",
        "line_status": "18G PIV x2 (MgSO4 & Steroids) | Inline Aerogen Nebulizer",
        "clinical_note": (
            "Worsening bronchospasm and thick secretion plugging after 05:00 AM: "
            "rising PIP, falling SpO2, and tachycardia."
        ),
        "baseline": {
            "rr": (18.5, 0.6), "hr": (84.0, 1.6), "spo2": (96.5, 0.4),
            "sbp": (118.0, 1.8), "dbp": (75.0, 1.2), "uo": (55.0, 3.0),
            "etco2": (38.0, 0.7), "pip": (23.0, 0.8), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [85, 430, 910],
        "crash_type": "bronchospasm"
    },
    "VENT-09": {
        "bed": "Bed 09",
        "pod": "Pod B",
        "name": "Patient I — Post-CABG Day 1 (Stable Recovery)",
        "age_sex": "66M",
        "diagnosis": "CABG x3 POD #1, Weaning Off Propofol for Extubation",
        "vent_mode": "PSV 10 / PEEP 5 | FiO2 30%",
        "line_status": "L Radial ART Line | R IJ Cordis | Chest Tubes (<40 mL/h)",
        "clinical_note": (
            "Stable surgical recovery control with routine coughing/turning artifacts suppressed. 0% false alarms."
        ),
        "baseline": {
            "rr": (15.5, 0.5), "hr": (78.0, 1.5), "spo2": (97.5, 0.4),
            "sbp": (118.0, 1.9), "dbp": (66.0, 1.3), "uo": (60.0, 3.5),
            "etco2": (38.0, 0.7), "pip": (17.5, 0.6), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [95, 460, 890, 1240],
        "crash_type": "none"
    },
    "VENT-10": {
        "bed": "Bed 10",
        "pod": "Pod B",
        "name": "Patient J — DKA Step-Down (Compensated Tachypnea)",
        "age_sex": "29M",
        "diagnosis": "Severe DKA Resolving on Insulin Infusion",
        "vent_mode": "HFNC 35 L/min | FiO2 28% (Monitored Capnography)",
        "line_status": "18G PIV (Insulin Drip + D5 1/2NS) | Q1h Glucose Checks",
        "clinical_note": (
            "Compensated Kussmaul baseline (RR ~24, EtCO2 ~28). Trips legacy RR alarms all shift; 0% false alarms here."
        ),
        "baseline": {
            "rr": (24.0, 0.6), "hr": (92.0, 1.8), "spo2": (98.5, 0.3),
            "sbp": (116.0, 1.8), "dbp": (72.0, 1.3), "uo": (75.0, 4.0),
            "etco2": (28.0, 0.7), "pip": (12.0, 0.5), "temp_grad": (1.4, 0.09)
        },
        "artifacts": [100, 500, 940, 1270],
        "crash_type": "none"
    },
    "VENT-11": {
        "bed": "Bed 11",
        "pod": "Pod B",
        "name": "Patient K — Post-Craniotomy Neuro ICU (Stable Control)",
        "age_sex": "51M",
        "diagnosis": "Subdural Hematoma Evacuation POD #2, EVD Transduced (ICP 9)",
        "vent_mode": "AC/VC | TV 480 | PEEP 5 | FiO2 30%",
        "line_status": "EVD Clear CSF | Radial ART Line | 3% NaCl Hold",
        "clinical_note": (
            "Stable neuro-protective control with Q1h neuro checks and suction blips damped. 0% false alarms."
        ),
        "baseline": {
            "rr": (16.0, 0.5), "hr": (68.0, 1.4), "spo2": (98.0, 0.4),
            "sbp": (132.0, 2.0), "dbp": (74.0, 1.4), "uo": (55.0, 3.0),
            "etco2": (36.5, 0.6), "pip": (19.5, 0.6), "temp_grad": (1.4, 0.09)
        },
        "artifacts": [80, 420, 860, 1200],
        "crash_type": "none"
    },
    "VENT-12": {
        "bed": "Bed 12",
        "pod": "Pod B",
        "name": "Patient L — Chronic HFrEF / Controlled AFib (Low-Normal BP)",
        "age_sex": "76F",
        "diagnosis": "Decompensated HFrEF (EF 25%), Rate-Controlled AFib",
        "vent_mode": "PSV 8 / PEEP 5 | FiO2 30% (SBT 07:00)",
        "line_status": "L Forearm 20G PIV (Lasix Drip 10 mg/h) | Foley Catheter",
        "clinical_note": (
            "Chronic low-normal BP (~98/62) & AFib HR (~90). Trips legacy SBP < 100 alarms; 0% false alarms here."
        ),
        "baseline": {
            "rr": (17.5, 0.5), "hr": (90.0, 2.0), "spo2": (95.0, 0.4),
            "sbp": (98.0, 1.8), "dbp": (62.0, 1.2), "uo": (85.0, 4.5),
            "etco2": (37.0, 0.7), "pip": (16.5, 0.6), "temp_grad": (1.8, 0.10)
        },
        "artifacts": [115, 530, 970, 1280],
        "crash_type": "none"
    }
}


def generate_synthetic_telemetry(db_path: str = DB_NAME, minutes: int = SHIFT_MINUTES) -> pd.DataFrame:
    start_time = datetime(2026, 11, 10, 7, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(minutes)]
    all_dfs = []

    for seed_offset, (pid, meta) in enumerate(PATIENT_PROFILES.items()):
        np.random.seed(42 + seed_offset)
        b = meta["baseline"]

        rr = np.random.normal(*b["rr"], size=minutes)
        hr = np.random.normal(*b["hr"], size=minutes)
        spo2 = np.random.normal(*b["spo2"], size=minutes)
        sbp = np.random.normal(*b["sbp"], size=minutes)
        dbp = np.random.normal(*b["dbp"], size=minutes)
        uo = np.random.normal(*b["uo"], size=minutes)
        etco2 = np.random.normal(*b["etco2"], size=minutes)
        pip = np.random.normal(*b["pip"], size=minutes)
        temp_grad = np.random.normal(*b["temp_grad"], size=minutes)

        for idx in meta["artifacts"]:
            if idx >= minutes - 2:
                continue
            if pid == "VENT-04" and idx == 220:
                pip[idx] -= 10.5
                rr[idx] -= 6.5
            else:
                rr[idx:idx + 2] += 11.5
                hr[idx:idx + 2] += 18.0
                pip[idx:idx + 2] += 14.5
                spo2[idx:idx + 2] -= 5.0

        c_start = CRASH_START_MIN
        w = minutes - c_start
        crash = meta["crash_type"]

        # Pre-breakout autonomic variability compression ("Physiological Squeeze" coil: 04:30-04:59 AM)
        if crash in ("mucus_sepsis", "septic_shock_drift"):
            coil_s, coil_e = c_start - 30, c_start
            rr[coil_s:coil_e] = np.random.normal(b["rr"][0], 0.11, size=30)
            hr[coil_s:coil_e] = np.random.normal(b["hr"][0], 0.28, size=30)
            sbp[coil_s:coil_e] = np.random.normal(b["sbp"][0], 0.30, size=30)
            dbp[coil_s:coil_e] = np.random.normal(b["dbp"][0], 0.22, size=30)

        if crash == "mucus_sepsis":
            rr[c_start:] += np.linspace(0, 8.5, w)
            hr[c_start:] += np.linspace(0, 22.0, w)
            spo2[c_start:] -= np.linspace(0, 4.5, w)
            sbp[c_start:] -= np.linspace(0, 6.0, w)
            dbp[c_start:] += np.linspace(0, 9.5, w)
            uo[c_start:] -= np.linspace(0, 25.0, w)
            etco2[c_start:] += np.linspace(0, 9.5, w)
            pip[c_start:] += np.linspace(0, 10.5, w)
            temp_grad[c_start:] += np.linspace(0, 2.1, w)

        elif crash == "autonomic_dysreflexia":
            rr[c_start:] += np.linspace(0, 8.0, w)
            hr[c_start:] -= np.linspace(0, 10.5, w)
            spo2[c_start:] -= np.linspace(0, 3.5, w)
            sbp[c_start:] += np.linspace(0, 36.0, w)
            dbp[c_start:] += np.linspace(0, 12.0, w)
            uo[c_start:] -= np.linspace(0, 22.0, w)
            etco2[c_start:] += np.linspace(0, 8.5, w)
            pip[c_start:] += np.linspace(0, 10.0, w)
            temp_grad[c_start:] += np.linspace(0, 1.9, w)

        elif crash == "weaning_fatigue":
            rr[c_start:] += np.linspace(0, 11.0, w)
            hr[c_start:] += np.linspace(0, 20.0, w)
            spo2[c_start:] -= np.linspace(0, 4.0, w)
            sbp[c_start:] -= np.linspace(0, 4.0, w)
            dbp[c_start:] += np.linspace(0, 9.0, w)
            uo[c_start:] -= np.linspace(0, 18.0, w)
            etco2[c_start:] += np.linspace(0, 10.0, w)
            pip[c_start:] += np.linspace(0, 7.5, w)
            temp_grad[c_start:] += np.linspace(0, 1.7, w)

        elif crash == "silent_hypovent":
            rr[c_start:] -= np.linspace(0, 6.5, w)
            etco2[c_start:] += np.linspace(0, 16.0, w)
            hr[c_start:] -= np.linspace(0, 5.0, w)
            spo2[c_start:] -= np.linspace(0, 0.5, w)

        elif crash == "circuit_leak":
            pip[c_start:] -= np.linspace(0, 14.0, w)
            rr[c_start:] += np.linspace(0, 5.0, w)
            spo2[c_start:] -= np.linspace(0, 2.5, w)
            hr[c_start:] += np.linspace(0, 10.0, w)

        elif crash == "septic_shock_drift":
            rr[c_start:] += np.linspace(0, 11.5, w)
            hr[c_start:] += np.linspace(0, 26.0, w)
            sbp[c_start:] -= np.linspace(0, 14.0, w)
            dbp[c_start:] += np.linspace(0, 4.0, w)
            uo[c_start:] -= np.linspace(0, 28.0, w)
            temp_grad[c_start:] += np.linspace(0, 2.2, w)
            spo2[c_start:] -= np.linspace(0, 2.0, w)

        elif crash == "bronchospasm":
            pip[c_start:] += np.linspace(0, 13.0, w)
            spo2[c_start:] -= np.linspace(0, 6.5, w)
            hr[c_start:] += np.linspace(0, 22.0, w)
            rr[c_start:] += np.linspace(0, 7.0, w)
            etco2[c_start:] += np.linspace(0, 5.0, w)

        p_df = pd.DataFrame({
            "patient_id": pid,
            "minute_index": np.arange(minutes),
            "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
            "rr": np.clip(np.round(rr, 1), 4.0, None),
            "hr": np.clip(np.round(hr, 1), 30.0, None),
            "spo2": np.clip(np.round(spo2, 1), 70.0, 100.0),
            "sbp": np.round(sbp, 1),
            "dbp": np.round(dbp, 1),
            "pulse_pressure": np.round(sbp - dbp, 1),
            "urine_output": np.clip(np.round(uo, 1), 0.0, None),
            "etco2": np.round(etco2, 1),
            "pip": np.clip(np.round(pip, 1), 4.0, None),
            "temp_gradient": np.round(temp_grad, 2)
        })
        all_dfs.append(p_df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    with sqlite3.connect(db_path) as conn:
        full_df.to_sql("vitals_stream", conn, if_exists="replace", index=False)

    return full_df


def compute_physiological_squeeze(df: pd.DataFrame, short_win: int = 15, base_win: int = 120) -> pd.DataFrame:
    channels = {
        "rr": ("rr_clean", 0.22),
        "pp": ("pulse_pressure_clean", 0.75),
        "hr": ("hr_clean", 0.65),
    }

    squeeze_flags = []
    for key, (col, floor) in channels.items():
        short_std = df[col].rolling(window=short_win, min_periods=10).std().fillna(floor)
        base_std = (
            df[col].shift(30).combine_first(df[col])
            .rolling(window=base_win, min_periods=30)
            .std()
            .clip(lower=floor)
            .fillna(floor)
        )
        ratio = np.round(short_std / base_std, 2)
        df[f"{key}_squeeze_ratio"] = ratio
        squeeze_flags.append(ratio < 0.45)

    active_squeezes = sum(f.astype(int) for f in squeeze_flags)
    raw_squeeze = active_squeezes >= 2

    df["phys_squeeze_active"] = raw_squeeze.rolling(window=3, min_periods=3).sum().fillna(0).eq(3)
    df["phys_squeeze_score"] = np.round(
        (1.0 - np.minimum(df[["rr_squeeze_ratio", "pp_squeeze_ratio", "hr_squeeze_ratio"]].mean(axis=1), 1.0)) * 100.0,
        1
    )
    return df


def _process_single_patient(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("minute_index").reset_index(drop=True)

    vitals_cols = [
        "rr", "hr", "spo2", "sbp", "dbp", "pulse_pressure",
        "urine_output", "etco2", "pip", "temp_gradient"
    ]
    for col in vitals_cols:
        df[f"{col}_clean"] = df[col].rolling(window=5, min_periods=1).median()

    artifact_mask = pd.Series(False, index=df.index)
    for col, thresh in ARTIFACT_SPIKE_THRESHOLD.items():
        artifact_mask |= (df[col] - df[f"{col}_clean"]).abs() >= thresh
    df["artifact_suppressed"] = artifact_mask

    df = compute_physiological_squeeze(df)

    z_features = ["rr", "hr", "spo2", "sbp", "pulse_pressure", "etco2", "pip", "temp_gradient"]
    for col in z_features:
        clean_col = f"{col}_clean"
        ref_series = df[clean_col].shift(60).combine_first(df[clean_col])
        roll_mean = ref_series.rolling(window=240, min_periods=30).mean()
        roll_std = (
            ref_series
            .rolling(window=240, min_periods=30)
            .std()
            .clip(lower=MIN_STD_FLOOR[col])
        )
        df[f"{col}_base_mean"] = np.round(roll_mean, 1)
        df[f"{col}_zscore"] = np.round((df[clean_col] - roll_mean) / roll_std, 2)

    df["dRR_dt_30m"] = np.round(df["rr_clean"].diff(periods=30), 2)
    df["dPIP_dt_30m"] = np.round(df["pip_clean"].diff(periods=30), 2)
    df["dEtCO2_dt_30m"] = np.round(df["etco2_clean"].diff(periods=30), 2)
    df["dPP_dt_30m"] = np.round(df["pulse_pressure_clean"].diff(periods=30), 2)
    df["dSBP_dt_30m"] = np.round(df["sbp_clean"].diff(periods=30), 2)
    df["dSpO2_dt_30m"] = np.round(df["spo2_clean"].diff(periods=30), 2)

    df["core_velocity_alert"] = (
        (df["rr_zscore"] > 2.5) &
        (df["dRR_dt_30m"] > 2.2) &
        (df["pulse_pressure_zscore"] < -2.0)
    )
    df["vent_mechanics_alert"] = (
        (df["pip_zscore"] > 2.5) &
        (df["etco2_zscore"] > 2.5) &
        (df["temp_gradient_zscore"] > 2.0)
    )
    df["autonomic_dysreflexia_alert"] = (
        (df["sbp_zscore"] > 3.0) &
        (df["pip_zscore"] > 2.5) &
        (df["hr_zscore"] < -1.5)
    )
    df["low_pressure_leak_alert"] = (
        (df["pip_zscore"] <= -2.5) &
        ((df["pip_clean"] - df["pip_base_mean"]) <= -3.0)
    )
    df["secretion_burden_alert"] = (
        ((df["pip_zscore"] >= 2.8) & (df["spo2_zscore"] <= -2.5) & (df["dPIP_dt_30m"] >= 1.5)) |
        ((df["pip_zscore"] >= 3.5) & (df["spo2_zscore"] <= -3.0))
    )
    df["silent_hypovent_alert"] = (
        ((df["etco2_zscore"] >= 2.8) & (df["rr_zscore"] <= -2.2) & (df["dEtCO2_dt_30m"] >= 1.5)) |
        ((df["etco2_zscore"] >= 3.5) & (df["rr_zscore"] <= -2.5))
    )
    raw_pre_alert = (
        df["core_velocity_alert"] |
        df["vent_mechanics_alert"] |
        df["autonomic_dysreflexia_alert"] |
        df["low_pressure_leak_alert"] |
        df["secretion_burden_alert"] |
        df["silent_hypovent_alert"]
    )
    df["pre_ehr_alert"] = raw_pre_alert.rolling(window=3, min_periods=3).sum().fillna(0).eq(3)
    df["legacy_static_alert"] = (
        (df["rr"] > 20.5) |
        (df["spo2"] < 92.0) |
        (df["sbp"] < 100.0) |
        (df["hr"] < 60.0)
    )
    return df


def evaluate_bedside_context(row: pd.Series) -> List[Dict]:
    alerts: List[Dict] = []

    if bool(row.get("autonomic_dysreflexia_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Autonomic Dysreflexia (BP Surge + Slow HR)",
            "signal_read": (
                f"SBP jumped to `{row['sbp_clean']} mmHg` (`{row['sbp_zscore']:+g} SD`) "
                f"with slow HR (`{row['hr_clean']} bpm`) and rising PIP (`{row['pip_clean']} cmH₂O`)."
            ),
            "bedside_cues": [
                "Sit patient upright (90° HOB) if safe; check for headache, flushing, or sweating.",
                "Check Foley catheter for kinks/retention and inspect lines or tight clothing.",
                "Notify MD/RT immediately per Autonomic Dysreflexia protocol.",
            ],
        })

    if bool(row.get("low_pressure_leak_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Sustained Airway Pressure Drop (Cuff / Circuit Leak)",
            "signal_read": (
                f"PIP dropped to `{row['pip_clean']} cmH₂O` (`{row['pip_zscore']:+g} SD`, "
                f"`{row['dPIP_dt_30m']:+g}`/30m) — sustained beyond routine care noise."
            ),
            "bedside_cues": [
                "Check patient chest rise, breathing effort, and SpO₂ first.",
                "Inspect trach/ETT cuff seal, tubing joints, and water traps.",
                "Call RT if leak continues or patient shows distress.",
            ],
        })

    if bool(row.get("vent_mechanics_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "High Airway Resistance & Perfusion Strain",
            "signal_read": (
                f"PIP `{row['pip_clean']} cmH₂O` (`{row['pip_zscore']:+g} SD`), "
                f"EtCO₂ `{row['etco2_clean']} mmHg` (`{row['etco2_zscore']:+g} SD`), "
                f"Temp ΔT `{row['temp_gradient_clean']}°C`."
            ),
            "bedside_cues": [
                "Listen for coarse secretions or tube kinking/biting.",
                "Suction inline per protocol and drain circuit condensate.",
                "Loop in RT if PIP or EtCO₂ stays elevated after suctioning.",
            ],
        })

    if bool(row.get("secretion_burden_alert", False)) and not bool(row.get("vent_mechanics_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Rising PIP + Falling SpO₂ (Secretions / Bronchospasm)",
            "signal_read": (
                f"PIP climbed to `{row['pip_clean']} cmH₂O` (`{row['pip_zscore']:+g} SD`) "
                f"while SpO₂ fell to `{row['spo2_clean']}%` (`{row['spo2_zscore']:+g} SD`)."
            ),
            "bedside_cues": [
                "Assess breath sounds (wheezes vs. coarse rhonchi) and oxygenation.",
                "Suction airway inline and check tubing for water or kinks.",
                "Notify RT if airway pressure remains high or bronchodilator is needed.",
            ],
        })

    if bool(row.get("core_velocity_alert", False)):
        alerts.append({
            "severity": "WATCH",
            "title": "Rising RR + Narrowing Pulse Pressure (Shock / Fatigue Creep)",
            "signal_read": (
                f"RR climbed to `{row['rr_clean']} bpm` (`{row['dRR_dt_30m']:+g}`/30m) "
                f"while Pulse Pressure narrowed to `{row['pulse_pressure_clean']} mmHg` (`{row['pulse_pressure_zscore']:+g} SD`)."
            ),
            "bedside_cues": [
                "Check work of breathing, capillary refill, urine output, and alertness.",
                "Review 2–4h trend direction rather than single-point vitals.",
                "Escalate early to Charge RN / MD / RT for sepsis or weaning review.",
            ],
        })

    if bool(row.get("silent_hypovent_alert", False)):
        alerts.append({
            "severity": "WATCH",
            "title": "Silent Hypoventilation (Rising EtCO₂ + Slowing RR)",
            "signal_read": (
                f"EtCO₂ rose to `{row['etco2_clean']} mmHg` (`{row['etco2_zscore']:+g} SD`) "
                f"as RR slowed to `{row['rr_clean']} bpm` (SpO₂ still `{row['spo2_clean']}%`)."
            ),
            "bedside_cues": [
                "Check level of consciousness / arousability and breathing effort.",
                "Review recent opioids/sedatives and verify vent backup rate.",
                "Notify RT and MD (do not rely on normal SpO₂ alone).",
            ],
        })

    if bool(row.get("phys_squeeze_active", False)) and not bool(row.get("pre_ehr_alert", False)):
        alerts.append({
            "severity": "WATCH",
            "title": "Pre-Breakout Coil (Loss of Natural Vital Variability)",
            "signal_read": (
                f"Vital micro-variability compressed to `{row['phys_squeeze_score']}%` Coil Index "
                f"while average vitals still look flat."
            ),
            "bedside_cues": [
                "Check alertness, skin perfusion, and breathing effort before vitals shift.",
                "Audit recent sedation, ventilator synchrony, and Foley/line patency.",
                "Watch closely over the next 30–60 min for directional drift.",
            ],
        })

    return alerts


def build_shift_event_strip(
    df: pd.DataFrame,
    selected_min: int,
    manual_tags: List[Dict],
) -> pd.DataFrame:
    upto_df = df.iloc[:selected_min + 1]
    records: List[Dict] = []

    art_rows = upto_df[upto_df["artifact_suppressed"]]
    for _, r in art_rows.iterrows():
        diffs = []
        for col, thresh in ARTIFACT_SPIKE_THRESHOLD.items():
            d = float(r[col] - r[f"{col}_clean"])
            if abs(d) >= thresh:
                direction = "spike" if d > 0 else "dip"
                diffs.append(f"{col.upper()} {direction} ({d:+.1f})")
        records.append({
            "minute_index": int(r["minute_index"]),
            "Time": r["timestamp"].strftime("%I:%M %p"),
            "Type": "Artifact Filtered",
            "Event": "Noise Blip Suppressed",
            "Detail": ", ".join(diffs) if diffs else "Transient spike damped",
        })

    squeeze_rows = upto_df[upto_df["phys_squeeze_active"]]
    if not squeeze_rows.empty:
        first_sq = squeeze_rows.iloc[0]
        records.append({
            "minute_index": int(first_sq["minute_index"]),
            "Time": first_sq["timestamp"].strftime("%I:%M %p"),
            "Type": "Variability Coil",
            "Event": "Pre-Breakout Squeeze Detected",
            "Detail": f"Coil Index {first_sq['phys_squeeze_score']}% (min {int(first_sq['minute_index'])})",
        })

    alert_rows = upto_df[upto_df["pre_ehr_alert"]]
    if not alert_rows.empty:
        first_alert = alert_rows.iloc[0]
        ctx = evaluate_bedside_context(first_alert)
        title = ctx[0]["title"] if ctx else "Sustained Trend Breach"
        records.append({
            "minute_index": int(first_alert["minute_index"]),
            "Time": first_alert["timestamp"].strftime("%I:%M %p"),
            "Type": "Trend Alert",
            "Event": title,
            "Detail": f"First sustained breach (min {int(first_alert['minute_index'])})",
        })

    for tag in manual_tags:
        if int(tag["minute_index"]) <= selected_min:
            records.append({
                "minute_index": int(tag["minute_index"]),
                "Time": tag["time_str"],
                "Type": tag.get("type", "Care Tag"),
                "Event": tag["label"],
                "Detail": tag["detail"],
            })

    if not records:
        return pd.DataFrame(columns=["Time", "Type", "Event", "Detail"])

    strip_df = pd.DataFrame(records).sort_values("minute_index", ascending=False)
    return strip_df[["Time", "Type", "Event", "Detail"]].reset_index(drop=True)


def run_deltapulse_velocity_engine(db_path: str = DB_NAME, patient_id: str | None = None) -> pd.DataFrame:
    expected_pids = set(PATIENT_PROFILES.keys())
    expected_rows = len(PATIENT_PROFILES) * SHIFT_MINUTES
    try:
        with sqlite3.connect(db_path) as conn:
            raw_df = pd.read_sql("SELECT * FROM vitals_stream", conn)
        has_all_beds = (
            "patient_id" in raw_df.columns
            and len(raw_df) == expected_rows
            and expected_pids.issubset(set(raw_df["patient_id"].unique()))
        )
        if not has_all_beds:
            raw_df = generate_synthetic_telemetry(db_path)
    except Exception:
        raw_df = generate_synthetic_telemetry(db_path)

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    processed_parts = []
    for pid in PATIENT_PROFILES.keys():
        slice_df = raw_df[raw_df["patient_id"] == pid].copy()
        processed_parts.append(_process_single_patient(slice_df))

    cohort_df = pd.concat(processed_parts, ignore_index=True)
    if patient_id is not None:
        return cohort_df[cohort_df["patient_id"] == patient_id].reset_index(drop=True)
    return cohort_df


if __name__ == "__main__":
    print("Running DeltaPulse V3 12-Bed 24-Hour Verification...")
    generate_synthetic_telemetry(DB_NAME)
    cohort = run_deltapulse_velocity_engine(DB_NAME, patient_id=None)

    assert len(cohort) == len(PATIENT_PROFILES) * SHIFT_MINUTES
    for pid, meta in PATIENT_PROFILES.items():
        p_df = cohort[cohort["patient_id"] == pid].reset_index(drop=True)
        base_fp = int(p_df.iloc[:CRASH_START_MIN]["pre_ehr_alert"].sum())
        m_late_alert = bool(p_df.iloc[CHECKPOINT_LATE_MIN]["pre_ehr_alert"])
        m_end_alert = bool(p_df.iloc[SHIFT_MINUTES - 1]["pre_ehr_alert"])

        if meta["crash_type"] == "none":
            assert int(p_df["pre_ehr_alert"].sum()) == 0
        else:
            assert base_fp == 0 and m_late_alert and m_end_alert
    print("[PASS] All 12 ICU beds verified across 17,280 patient-minutes.")
