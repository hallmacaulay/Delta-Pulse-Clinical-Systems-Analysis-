import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

DB_NAME = "aeropulse_vitals.db"

# Minimum rolling std floors prevent division-by-zero or Z-score explosions
# on flat-line windows without blunting true physiological shifts.
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

# Raw vs. 5-min rolling median delta thresholds for Stage 1 artifact flagging
ARTIFACT_SPIKE_THRESHOLD: Dict[str, float] = {
    "rr": 5.0,
    "hr": 12.0,
    "spo2": 3.5,
    "sbp": 18.0,
    "pip": 5.0,
}

# Bedside quick-tag menu for manual care / artifact context logging
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

# 12-Bed ICU Pod (7 Active Deterioration Archetypes + 5 Tricky-Baseline Stable Controls)
PATIENT_PROFILES = {
    "VENT-01": {
        "bed": "Bed 01",
        "pod": "Pod A",
        "name": "Patient A — Chronic COPD / Trach-Vent",
        "age_sex": "68M",
        "diagnosis": "End-Stage COPD, Chronic Hypercapnic Resp Failure (Inline Trach)",
        "vent_mode": "AC/VC | TV 450 | PEEP 8 | FiO2 35%",
        "line_status": "20G PIV (R Forearm): Patent | Inline Trach Suction & Heated Circuit (Compatible)",
        "clinical_note": (
            "Chronically elevated baseline (RR ~22 bpm, EtCO2 ~52 mmHg, SpO2 ~90.5%). "
            "Triggers continuous false alarms on static NEWS2 monitors. Simulates inline "
            "suctioning at 01:00 AM (m120), followed by mucus plugging & sepsis after 03:00 AM (m240)."
        ),
        "baseline": {
            "rr": (22.0, 0.7), "hr": (86.0, 1.8), "spo2": (90.5, 0.5),
            "sbp": (122.0, 2.0), "dbp": (78.0, 1.5), "uo": (45.0, 3.0),
            "etco2": (52.0, 0.9), "pip": (24.0, 0.9), "temp_grad": (1.8, 0.12)
        },
        "artifacts": [120, 195],
        "crash_type": "mucus_sepsis"
    },
    "VENT-02": {
        "bed": "Bed 02",
        "pod": "Pod A",
        "name": "Patient B — C4 Tetraplegia / Neurogenic Baseline",
        "age_sex": "42M",
        "diagnosis": "High-Cervical Spinal Cord Injury (C4 AIS-A), Trach-Vent Dependent",
        "vent_mode": "SIMV-PC | PIP 18 | PEEP 5 | FiO2 28%",
        "line_status": "Radial ART Line: Optimal Trace | Foley Catheter: Check Dependent Loop (AD Trigger Risk)",
        "clinical_note": (
            "Neurogenic baseline hypotension (BP ~94/60 mmHg) and bradycardia (HR ~56 bpm). "
            "Simulates turning artifact at 12:50 AM (m110), followed by acute Autonomic "
            "Dysreflexia (BP surge to ~130/72 mmHg + reflex bradycardia) & airway strain after 03:00 AM."
        ),
        "baseline": {
            "rr": (14.5, 0.5), "hr": (56.0, 1.4), "spo2": (97.0, 0.4),
            "sbp": (94.0, 1.8), "dbp": (60.0, 1.2), "uo": (52.0, 3.5),
            "etco2": (39.0, 0.7), "pip": (18.0, 0.7), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [110, 185],
        "crash_type": "autonomic_dysreflexia"
    },
    "VENT-03": {
        "bed": "Bed 03",
        "pod": "Pod A",
        "name": "Patient C — Post-ICU Trach Weaning (Pressure Support)",
        "age_sex": "59F",
        "diagnosis": "Post-ARDS Prolonged Mechanical Ventilation, Diaphragmatic Weakness",
        "vent_mode": "PSV 12 / PEEP 6 | FiO2 30%",
        "line_status": "Radial ART Line: Mild Positional Dampening (Flushed Q2h) | 20G PIV x2: K+ Rider Running",
        "clinical_note": (
            "Stable baseline during initial night shift support, with suction spike at 01:10 AM (m130), "
            "followed by progressive diaphragmatic fatigue and rapid shallow breathing creep after 03:00 AM."
        ),
        "baseline": {
            "rr": (16.0, 0.6), "hr": (78.0, 1.6), "spo2": (95.5, 0.4),
            "sbp": (116.0, 1.9), "dbp": (74.0, 1.4), "uo": (48.0, 2.8),
            "etco2": (41.0, 0.8), "pip": (17.0, 0.6), "temp_grad": (1.6, 0.11)
        },
        "artifacts": [130],
        "crash_type": "weaning_fatigue"
    },
    "VENT-04": {
        "bed": "Bed 04",
        "pod": "Pod A",
        "name": "Patient D — Stable ALS Vent Control (High Care Noise)",
        "age_sex": "71F",
        "diagnosis": "Neuromuscular Respiratory Weakness (ALS), Stable Chronic Trach-Vent",
        "vent_mode": "AC/PC | PIP 20 | PEEP 6 | FiO2 25%",
        "line_status": "Non-Invasive Cuff Cycling Q15m | PEG Tube: Nocturnal Feeds at 55 mL/hr",
        "clinical_note": (
            "Stable 6-hour control patient undergoing routine inline suctioning, coughing, "
            "repositioning (m90, m150, m290), and a brief circuit-open low-pressure dip (m220). "
            "Demonstrates bidirectional Stage 1 noise suppression with 0% false-positive alarms."
        ),
        "baseline": {
            "rr": (18.0, 0.6), "hr": (74.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (120.0, 1.8), "dbp": (76.0, 1.3), "uo": (50.0, 3.0),
            "etco2": (40.0, 0.7), "pip": (20.0, 0.7), "temp_grad": (1.7, 0.10)
        },
        "artifacts": [90, 150, 220, 290],
        "crash_type": "none"
    },
    "VENT-05": {
        "bed": "Bed 05",
        "pod": "Pod A",
        "name": "Patient E — Post-Op AAA / Silent Opioid Hypoventilation",
        "age_sex": "74M",
        "diagnosis": "Post-Op Open AAA Repair, PCA Hydromorphone / Thoracic Epidural",
        "vent_mode": "PSV 8 / PEEP 5 | FiO2 35% (Supplemental O2 Masking)",
        "line_status": "R IJ Central Line: Patent | Radial ART Line | PCA & Epidural Pumps Active",
        "clinical_note": (
            "Simulates silent CO2 retention after late-night opioid stacking: RR drifts down and "
            "EtCO2 climbs steadily while supplemental O2 keeps SpO2 deceptively parked near 97%."
        ),
        "baseline": {
            "rr": (15.5, 0.5), "hr": (74.0, 1.5), "spo2": (97.5, 0.4),
            "sbp": (124.0, 1.8), "dbp": (76.0, 1.2), "uo": (50.0, 3.0),
            "etco2": (40.0, 0.7), "pip": (18.0, 0.6), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [105, 190],
        "crash_type": "silent_hypovent"
    },
    "VENT-06": {
        "bed": "Bed 06",
        "pod": "Pod A",
        "name": "Patient F — Severe CAP / Cuff & Circuit Leak",
        "age_sex": "63F",
        "diagnosis": "Severe Community-Acquired Pneumonia, Trach-Vent Day 9",
        "vent_mode": "AC/VC | TV 420 | PEEP 10 | FiO2 40%",
        "line_status": "L PICC (Double Lumen): Abx Infusing | Heated Wire Circuit & Subglottic Trach",
        "clinical_note": (
            "Simulates a gradual trach cuff deflation / heated circuit leak after 03:00 AM, "
            "producing a sustained PIP drop and negative 30m pressure velocity beyond transient blips."
        ),
        "baseline": {
            "rr": (16.5, 0.5), "hr": (80.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (118.0, 1.8), "dbp": (74.0, 1.3), "uo": (46.0, 2.8),
            "etco2": (41.0, 0.7), "pip": (22.0, 0.7), "temp_grad": (1.6, 0.10)
        },
        "artifacts": [140],
        "crash_type": "circuit_leak"
    },
    "VENT-07": {
        "bed": "Bed 07",
        "pod": "Pod B",
        "name": "Patient G — Urosepsis / Early Septic Shock Drift",
        "age_sex": "79F",
        "diagnosis": "Obstructive Urosepsis, Low-Dose Norepinephrine (0.04 mcg/kg/min)",
        "vent_mode": "AC/PC | PIP 19 | PEEP 6 | FiO2 35%",
        "line_status": "R IJ Triple-Lumen CVC (Pressor Dedicated) | Foley with Urometer",
        "clinical_note": (
            "Simulates early septic decompensation ('slow boil'): rising RR and HR, narrowing "
            "pulse pressure, widening Core-Periph Temp gradient, and falling hourly urine output."
        ),
        "baseline": {
            "rr": (17.0, 0.5), "hr": (82.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (114.0, 1.8), "dbp": (68.0, 1.3), "uo": (42.0, 2.5),
            "etco2": (36.0, 0.7), "pip": (19.0, 0.6), "temp_grad": (1.9, 0.11)
        },
        "artifacts": [115, 205],
        "crash_type": "septic_shock_drift"
    },
    "VENT-08": {
        "bed": "Bed 08",
        "pod": "Pod B",
        "name": "Patient H — Status Asthmaticus / Secretion & Resistance",
        "age_sex": "34F",
        "diagnosis": "Refractory Status Asthmaticus, Intubated with Inline Bronchodilators",
        "vent_mode": "AC/VC | TV 440 | PEEP 5 | FiO2 45%",
        "line_status": "18G PIV x2: MgSO4 & Steroids | Inline Aerogen Nebulizer Active",
        "clinical_note": (
            "Simulates worsening bronchospasm and thick secretion plugging after 03:00 AM: "
            "sustained PIP climb paired with progressive SpO2 erosion and compensatory tachycardia."
        ),
        "baseline": {
            "rr": (18.5, 0.6), "hr": (84.0, 1.6), "spo2": (96.5, 0.4),
            "sbp": (118.0, 1.8), "dbp": (75.0, 1.2), "uo": (55.0, 3.0),
            "etco2": (38.0, 0.7), "pip": (23.0, 0.8), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [85, 160],
        "crash_type": "bronchospasm"
    },
    "VENT-09": {
        "bed": "Bed 09",
        "pod": "Pod B",
        "name": "Patient I — Post-CABG Day 1 (Stable Recovery)",
        "age_sex": "66M",
        "diagnosis": "CABG x3 POD #1, Weaning Off Propofol for Morning Extubation",
        "vent_mode": "PSV 10 / PEEP 5 | FiO2 30%",
        "line_status": "L Radial ART Line | R IJ Cordis | Mediastinal & Pleural Chest Tubes (<40 mL/h)",
        "clinical_note": (
            "Stable cardiac surgical recovery control experiencing brief coughing and turning "
            "artifacts (m95, m175, m265) as sedation lightens — demonstrates calm false-alarm suppression."
        ),
        "baseline": {
            "rr": (15.5, 0.5), "hr": (78.0, 1.5), "spo2": (97.5, 0.4),
            "sbp": (118.0, 1.9), "dbp": (66.0, 1.3), "uo": (60.0, 3.5),
            "etco2": (38.0, 0.7), "pip": (17.5, 0.6), "temp_grad": (1.5, 0.10)
        },
        "artifacts": [95, 175, 265],
        "crash_type": "none"
    },
    "VENT-10": {
        "bed": "Bed 10",
        "pod": "Pod B",
        "name": "Patient J — DKA Step-Down (Compensated Tachypnea)",
        "age_sex": "29M",
        "diagnosis": "Severe DKA Resolving on Insulin Infusion (Anion Gap Closing)",
        "vent_mode": "High-Flow Nasal Cannula 35 L/min | FiO2 28% (Monitored Capnography)",
        "line_status": "18G PIV (Insulin Drip + D5 1/2NS) | Q1h Point-of-Care Glucose Protocol",
        "clinical_note": (
            "Compensated Kussmaul baseline (RR ~24 bpm, EtCO2 ~28 mmHg) that sets off static "
            "RR > 20.5 alarms all shift on legacy monitors, while DeltaPulse holds 0 false alerts."
        ),
        "baseline": {
            "rr": (24.0, 0.6), "hr": (92.0, 1.8), "spo2": (98.5, 0.3),
            "sbp": (116.0, 1.8), "dbp": (72.0, 1.3), "uo": (75.0, 4.0),
            "etco2": (28.0, 0.7), "pip": (12.0, 0.5), "temp_grad": (1.4, 0.09)
        },
        "artifacts": [100, 190, 270],
        "crash_type": "none"
    },
    "VENT-11": {
        "bed": "Bed 11",
        "pod": "Pod B",
        "name": "Patient K — Post-Craniotomy Neuro ICU (Stable Control)",
        "age_sex": "51M",
        "diagnosis": "Subdural Hematoma Evacuation POD #2, EVD Transduced (ICP 9 mmHg)",
        "vent_mode": "AC/VC | TV 480 | PEEP 5 | FiO2 30% (Target EtCO2 35–38)",
        "line_status": "EVD Draining Clear CSF | Radial ART Line Leveled to Tragus | 3% NaCl Hold",
        "clinical_note": (
            "Stable neuro-protective ventilation control undergoing Q1h neuro checks, oral care, "
            "and suctioning blips (m80, m145, m230, m300) without sustained intracranial or airway drift."
        ),
        "baseline": {
            "rr": (16.0, 0.5), "hr": (68.0, 1.4), "spo2": (98.0, 0.4),
            "sbp": (132.0, 2.0), "dbp": (74.0, 1.4), "uo": (55.0, 3.0),
            "etco2": (36.5, 0.6), "pip": (19.5, 0.6), "temp_grad": (1.4, 0.09)
        },
        "artifacts": [80, 145, 230, 300],
        "crash_type": "none"
    },
    "VENT-12": {
        "bed": "Bed 12",
        "pod": "Pod B",
        "name": "Patient L — Chronic HFrEF / Controlled AFib (Low-Normal BP)",
        "age_sex": "76F",
        "diagnosis": "Decompensated HFrEF (EF 25%), Rate-Controlled AFib, Extubation-Ready",
        "vent_mode": "PSV 8 / PEEP 5 | FiO2 30% (SBT Scheduled 07:00)",
        "line_status": "L Forearm 20G PIV (Lasix Drip 10 mg/hr) | Foley Catheter (Strict I&O)",
        "clinical_note": (
            "Chronic low-normal systolic BP (~98 mmHg) and faster AFib baseline HR (~90 bpm) "
            "that constantly trip static SBP < 100 alarms on legacy monitors; stable throughout shift."
        ),
        "baseline": {
            "rr": (17.5, 0.5), "hr": (90.0, 2.0), "spo2": (95.0, 0.4),
            "sbp": (98.0, 1.8), "dbp": (62.0, 1.2), "uo": (85.0, 4.5),
            "etco2": (37.0, 0.7), "pip": (16.5, 0.6), "temp_grad": (1.8, 0.10)
        },
        "artifacts": [115, 210, 280],
        "crash_type": "none"
    }
}


def generate_synthetic_telemetry(db_path: str = DB_NAME, minutes: int = 360) -> pd.DataFrame:
    start_time = datetime(2026, 11, 10, 23, 0, 0)
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
            if pid == "VENT-04" and idx == 220:
                # Brief downward circuit-opening / low-pressure blip (1 min -> suppressed)
                pip[idx] -= 10.5
                rr[idx] -= 6.5
            else:
                # Brief upward suctioning / coughing / turning spike (2 min -> suppressed)
                rr[idx:idx + 2] += 11.5
                hr[idx:idx + 2] += 18.0
                pip[idx:idx + 2] += 14.5
                spo2[idx:idx + 2] -= 5.0

        w = minutes - 240
        crash = meta["crash_type"]

        if crash == "mucus_sepsis":
            rr[240:] += np.linspace(0, 8.5, w)
            hr[240:] += np.linspace(0, 22.0, w)
            spo2[240:] -= np.linspace(0, 4.5, w)
            sbp[240:] -= np.linspace(0, 6.0, w)
            dbp[240:] += np.linspace(0, 9.5, w)
            uo[240:] -= np.linspace(0, 25.0, w)
            etco2[240:] += np.linspace(0, 9.5, w)
            pip[240:] += np.linspace(0, 10.5, w)
            temp_grad[240:] += np.linspace(0, 2.1, w)

        elif crash == "autonomic_dysreflexia":
            rr[240:] += np.linspace(0, 8.0, w)
            hr[240:] -= np.linspace(0, 10.5, w)
            spo2[240:] -= np.linspace(0, 3.5, w)
            sbp[240:] += np.linspace(0, 36.0, w)
            dbp[240:] += np.linspace(0, 12.0, w)
            uo[240:] -= np.linspace(0, 22.0, w)
            etco2[240:] += np.linspace(0, 8.5, w)
            pip[240:] += np.linspace(0, 10.0, w)
            temp_grad[240:] += np.linspace(0, 1.9, w)

        elif crash == "weaning_fatigue":
            rr[240:] += np.linspace(0, 11.0, w)
            hr[240:] += np.linspace(0, 20.0, w)
            spo2[240:] -= np.linspace(0, 4.0, w)
            sbp[240:] -= np.linspace(0, 4.0, w)
            dbp[240:] += np.linspace(0, 9.0, w)
            uo[240:] -= np.linspace(0, 18.0, w)
            etco2[240:] += np.linspace(0, 10.0, w)
            pip[240:] += np.linspace(0, 7.5, w)
            temp_grad[240:] += np.linspace(0, 1.7, w)

        elif crash == "silent_hypovent":
            # Opioid CO2 narcosis: RR drops, EtCO2 climbs, SpO2 remains deceptively steady
            rr[240:] -= np.linspace(0, 6.5, w)
            etco2[240:] += np.linspace(0, 16.0, w)
            hr[240:] -= np.linspace(0, 5.0, w)
            spo2[240:] -= np.linspace(0, 0.5, w)

        elif crash == "circuit_leak":
            # Cuff deflation / circuit leak: sustained PIP drop & negative 30m velocity
            pip[240:] -= np.linspace(0, 14.0, w)
            rr[240:] += np.linspace(0, 5.0, w)
            spo2[240:] -= np.linspace(0, 2.5, w)
            hr[240:] += np.linspace(0, 10.0, w)

        elif crash == "septic_shock_drift":
            # Early urosepsis decompensation: tachypnea, tachycardia, narrowing PP, oliguria
            rr[240:] += np.linspace(0, 11.5, w)
            hr[240:] += np.linspace(0, 26.0, w)
            sbp[240:] -= np.linspace(0, 14.0, w)
            dbp[240:] += np.linspace(0, 4.0, w)
            uo[240:] -= np.linspace(0, 28.0, w)
            temp_grad[240:] += np.linspace(0, 2.2, w)
            spo2[240:] -= np.linspace(0, 2.0, w)

        elif crash == "bronchospasm":
            # Status asthmaticus / secretion plugging: rising PIP + falling SpO2 + tachycardia
            pip[240:] += np.linspace(0, 13.0, w)
            spo2[240:] -= np.linspace(0, 6.5, w)
            hr[240:] += np.linspace(0, 22.0, w)
            rr[240:] += np.linspace(0, 7.0, w)
            etco2[240:] += np.linspace(0, 5.0, w)

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


def _process_single_patient(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("minute_index").reset_index(drop=True)

    # Stage 1: 5-Minute Rolling Median Filter (Bidirectional Artifact Suppression)
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

    # Stage 2: 240-Minute Self-Calibrating Personal Z-Score Baselines (with Std Floor)
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

    # Stage 3: 30-Minute Physiological Velocity & Co-Moving Clinical Rules
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
        (df["pip_zscore"] <= -2.2) &
        (df["dPIP_dt_30m"] <= -3.0)
    )
    df["secretion_burden_alert"] = (
        (df["pip_zscore"] >= 2.2) &
        (df["spo2_zscore"] <= -2.0)
    )
    df["silent_hypovent_alert"] = (
        (df["etco2_zscore"] >= 2.2) &
        (df["rr_zscore"] <= -1.8)
    )
    df["pre_ehr_alert"] = (
        df["core_velocity_alert"] |
        df["vent_mechanics_alert"] |
        df["autonomic_dysreflexia_alert"] |
        df["low_pressure_leak_alert"] |
        df["secretion_burden_alert"] |
        df["silent_hypovent_alert"]
    )
    df["legacy_static_alert"] = (
        (df["rr"] > 20.5) |
        (df["spo2"] < 92.0) |
        (df["sbp"] < 100.0) |
        (df["hr"] < 60.0)
    )
    return df


def evaluate_bedside_context(row: pd.Series) -> List[Dict]:
    """
    Translates active co-moving rule flags at a single shift minute into
    plain-language signal stories and 3 assessment-first bedside checks.
    """
    alerts: List[Dict] = []

    if bool(row.get("autonomic_dysreflexia_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Autonomic Dysreflexia Crisis (SBP Surge + Reflex Bradycardia)",
            "signal_read": (
                f"Clean SBP surged to `{row['sbp_clean']} mmHg` (`{row['sbp_zscore']:+g} SD`, "
                f"`{row['dSBP_dt_30m']:+g}`/30m) with reflex bradycardia (`{row['hr_clean']} bpm`) "
                f"and rising PIP (`{row['pip_clean']} cmH2O`)."
            ),
            "bedside_cues": [
                "Assess patient immediately: elevate HOB to 90 degrees if safe; check for headache/flushing/diaphoresis.",
                "Trace noxious triggers: inspect Foley catheter for kinks/retention, check bowel/lines/tight clothing.",
                "Verify airway patency and escalate immediately per unit AD protocol / MD notification.",
            ],
        })

    if bool(row.get("low_pressure_leak_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Sustained Peak Pressure Drop (Possible Leak / Circuit Disconnect)",
            "signal_read": (
                f"Clean PIP fell to `{row['pip_clean']} cmH2O` (`{row['pip_zscore']:+g} SD`, "
                f"`{row['dPIP_dt_30m']:+g}`/30m) — sustained beyond transient care noise."
            ),
            "bedside_cues": [
                "Assess patient first: check chest rise, work of breathing, comfort, and SpO2.",
                "Trace airway and circuit: verify trach/ETT securement, tubing joints, water traps, and cuff seal.",
                "Notify RT immediately if leak persists or patient shows distress.",
            ],
        })

    if bool(row.get("vent_mechanics_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Ventilator Airway Resistance & Perfusion Strain",
            "signal_read": (
                f"Clean PIP `{row['pip_clean']} cmH2O` (`{row['pip_zscore']:+g} SD`), "
                f"EtCO2 `{row['etco2_clean']} mmHg` (`{row['etco2_zscore']:+g} SD`), and "
                f"Core-Periph Temp Gradient `{row['temp_gradient_clean']} C`."
            ),
            "bedside_cues": [
                "Assess breath sounds and patient comfort for coarse secretions, biting/kinking, or dyssynchrony.",
                "Perform inline trach suctioning per unit protocol and inspect circuit for condensate.",
                "Loop in RT if elevated PIP or EtCO2 fails to normalize post-clearance.",
            ],
        })

    if bool(row.get("secretion_burden_alert", False)) and not bool(row.get("vent_mechanics_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Rising Airway Pressure + SpO2 Erosion (Secretion / Bronchospasm Pattern)",
            "signal_read": (
                f"Sustained PIP climb to `{row['pip_clean']} cmH2O` (`{row['pip_zscore']:+g} SD`) "
                f"paired with SpO2 erosion to `{row['spo2_clean']}%` (`{row['spo2_zscore']:+g} SD`)."
            ),
            "bedside_cues": [
                "Assess patient, breath sounds (wheezing vs. coarse rhonchi), and oxygenation immediately.",
                "Check airway patency and suction inline; inspect circuit tubing for water or kinks.",
                "Notify RT if airway pressure remains elevated or bronchodilator/vent adjustment is needed.",
            ],
        })

    if bool(row.get("core_velocity_alert", False)):
        alerts.append({
            "severity": "WATCH",
            "title": "Hemodynamic & Tachypneic Velocity Creep (Early Shock / Fatigue Watch)",
            "signal_read": (
                f"Clean RR climbed to `{row['rr_clean']} bpm` (`{row['rr_zscore']:+g} SD`, "
                f"`{row['dRR_dt_30m']:+g}`/30m) while Pulse Pressure narrowed to "
                f"`{row['pulse_pressure_clean']} mmHg` (`{row['pulse_pressure_zscore']:+g} SD`)."
            ),
            "bedside_cues": [
                "Assess work of breathing, perfusion (capillary refill, skin temp), urine output, and mental status.",
                "Review 2–4 hour trend trajectory rather than isolated point-in-time vitals.",
                "Escalate early to charge nurse / MD / RT for weaning or sepsis evaluation.",
            ],
        })

    if bool(row.get("silent_hypovent_alert", False)):
        alerts.append({
            "severity": "WATCH",
            "title": "Rising EtCO2 + Falling Respiratory Rate (Silent Hypoventilation)",
            "signal_read": (
                f"Clean EtCO2 rose to `{row['etco2_clean']} mmHg` (`{row['etco2_zscore']:+g} SD`) "
                f"while RR dropped to `{row['rr_clean']} bpm` (`{row['rr_zscore']:+g} SD`); "
                f"SpO2 reads `{row['spo2_clean']}%`."
            ),
            "bedside_cues": [
                "Assess level of consciousness (AVPU/arousability) and respiratory effort immediately.",
                "Review recent sedatives/opioids and verify vent backup rate & EtCO2 sensor integrity.",
                "Notify RT and MD for ventilation review (do not rely solely on reassuring SpO2).",
            ],
        })

    return alerts


def build_shift_event_strip(
    df: pd.DataFrame,
    selected_min: int,
    manual_tags: List[Dict],
) -> pd.DataFrame:
    """
    Builds a chronological 'What Just Happened' memory-offloading strip up to
    selected_min, combining Stage 1 suppressed artifacts, alert onset, and manual care tags.
    """
    upto_df = df.iloc[:selected_min + 1]
    records: List[Dict] = []

    # 1. Stage 1 Auto-Suppressed Artifacts
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
            "Event": "Stage 1 Median Suppressed Noise",
            "Detail": ", ".join(diffs) if diffs else "Transient raw noise damped",
        })

    # 2. First Pre-EHR Alert Onset (if active in window)
    alert_rows = upto_df[upto_df["pre_ehr_alert"]]
    if not alert_rows.empty:
        first_alert = alert_rows.iloc[0]
        ctx = evaluate_bedside_context(first_alert)
        title = ctx[0]["title"] if ctx else "Pre-EHR Co-Moving Trend Breach"
        records.append({
            "minute_index": int(first_alert["minute_index"]),
            "Time": first_alert["timestamp"].strftime("%I:%M %p"),
            "Type": "Trend Alert",
            "Event": title,
            "Detail": f"First sustained breach at minute {int(first_alert['minute_index'])}",
        })

    # 3. Manual Bedside Care Tags
    for tag in manual_tags:
        if int(tag["minute_index"]) <= selected_min:
            records.append({
                "minute_index": int(tag["minute_index"]),
                "Time": tag["time_str"],
                "Type": "Bedside Care Tag",
                "Event": tag["label"],
                "Detail": tag["detail"],
            })

    if not records:
        return pd.DataFrame(columns=["Time", "Type", "Event", "Detail"])

    strip_df = pd.DataFrame(records).sort_values("minute_index", ascending=False)
    return strip_df[["Time", "Type", "Event", "Detail"]].reset_index(drop=True)


def run_deltapulse_velocity_engine(db_path: str = DB_NAME, patient_id: str | None = None) -> pd.DataFrame:
    expected_pids = set(PATIENT_PROFILES.keys())
    expected_rows = len(PATIENT_PROFILES) * 360
    try:
        with sqlite3.connect(db_path) as conn:
            raw_df = pd.read_sql("SELECT * FROM vitals_stream", conn)
        has_all_beds = (
            "patient_id" in raw_df.columns
            and len(raw_df) >= expected_rows
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
    print("Running DeltaPulse 12-Bed ICU Pod End-to-End Verification...")
    generate_synthetic_telemetry(DB_NAME)
    cohort = run_deltapulse_velocity_engine(DB_NAME, patient_id=None)

    assert len(cohort) == len(PATIENT_PROFILES) * 360, f"Unexpected row count: {len(cohort)}"
    print(f"[PASS] Loaded {len(cohort)} patient-minutes across {cohort['patient_id'].nunique()} beds.")

    for pid, meta in PATIENT_PROFILES.items():
        p_df = cohort[cohort["patient_id"] == pid].reset_index(drop=True)
        base_fp = int(p_df.iloc[:240]["pre_ehr_alert"].sum())
        suppressed = int(p_df["artifact_suppressed"].sum())

        m305_alert = bool(p_df.iloc[305]["pre_ehr_alert"])
        m359_alert = bool(p_df.iloc[359]["pre_ehr_alert"])

        if meta["crash_type"] == "none":
            full_fp = int(p_df["pre_ehr_alert"].sum())
            assert full_fp == 0, f"[FAIL] Stable control {pid} triggered {full_fp} false alerts"
            status = f"0% FP across 6h | {suppressed} artifact mins suppressed"
        else:
            assert base_fp == 0, f"[FAIL] {pid} triggered {base_fp} baseline false alerts"

            assert m305_alert, f"[FAIL] {pid} ({meta['crash_type']}) did not trigger by m305"
            assert m359_alert, f"[FAIL] {pid} ({meta['crash_type']}) dropped out by m359"
            ctx = evaluate_bedside_context(p_df.iloc[305])

            status = f"0 baseline FP | m305–m359 LATCHED: {ctx[0]['title']}"

        print(f"  - {meta['bed']} ({pid}): {status}")

    print("[PASS] All 12 ICU beds passed deterministic signal, artifact, and co-moving rule validation.")
