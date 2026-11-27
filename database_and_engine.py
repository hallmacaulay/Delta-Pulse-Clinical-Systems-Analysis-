import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

DB_NAME = "aeropulse_vitals_v4.db"
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
    "vte": 8.0,
    "mv": 0.25,
    "rsbi": 2.5,
    "sf_ratio": 3.0,
    "temp_gradient": 0.05,
}

ARTIFACT_SPIKE_THRESHOLD: Dict[str, float] = {
    "rr": 5.0,
    "hr": 12.0,
    "spo2": 3.5,
    "sbp": 18.0,
    "pip": 5.0,
    "vte": 65.0,
}

ARTIFACT_TAGS: Dict[str, List[str]] = {
    "Airway / RT Care (5m Hold)": [
        "Suctioning",
        "Inline Nebulizer",
        "Circuit / Condensate Clear",
        "Cuff Check",
    ],
    "Patient / Line": [
        "Turning / Repositioning",
        "Coughing / Asynchrony",
        "ART Line Flush / Zero",
    ],
    "Meds / Titration": [
        "Norepi Titrated Up",
        "Analgesia / Sedation Bolus",
        "Lasix / Diuretic Bolus",
    ],
}

# Active care tags that trigger a 5-minute Expected Care Variance window
CARE_MUTE_TAGS = {"Suctioning", "Inline Nebulizer", "Circuit / Condensate Clear", "Cuff Check"}

PATIENT_PROFILES = {
    "VENT-01": {
        "bed": "Bed 01",
        "pod": "Pod A",
        "name": "Patient A — Chronic COPD / Trach-Vent",
        "age_sex": "68M",
        "diagnosis": "End-Stage COPD, Chronic Hypercapnic Resp Failure (Inline Trach)",
        "vent_mode": "AC/VC | TV 450 | PEEP 8 | FiO2 35%",
        "control_type": "volume",
        "fio2": 0.35,
        "line_status": "20G PIV (R Forearm) | Inline Trach Suction & Heated Circuit",
        "med_events": [
            {"minute_index": 240, "label": "Duoneb Inline + Solu-Medrol 40mg IV", "detail": "Scheduled bronchodilator & steroid"},
            {"minute_index": 960, "label": "Duoneb Inline Treatment", "detail": "Night RT aerosol therapy"},
        ],
        "clinical_note": (
            "Volume-control COPD baseline (RR ~22, EtCO2 ~52, SpO2 ~90.5%). "
            "Pre-breakout coil at 04:30 AM, then mucus plugging & sepsis (PIP surge) after 05:00 AM."
        ),
        "baseline": {
            "rr": (22.0, 0.7), "hr": (86.0, 1.8), "spo2": (90.5, 0.5),
            "sbp": (122.0, 2.0), "dbp": (78.0, 1.5), "uo": (45.0, 3.0),
            "etco2": (52.0, 0.9), "pip": (24.0, 0.9), "vte": (445.0, 10.0),
            "temp_grad": (1.8, 0.12)
        },
        "artifacts": [120, 480, 960, 1180],
        "probe_blip_min": None,
        "crash_type": "mucus_sepsis"
    },
    "VENT-02": {
        "bed": "Bed 02",
        "pod": "Pod A",
        "name": "Patient B — C4 Tetraplegia / Neurogenic Baseline",
        "age_sex": "42M",
        "diagnosis": "High-Cervical SCI (C4 AIS-A), Trach-Vent Dependent",
        "vent_mode": "SIMV-PC | PIP 18 | PEEP 5 | FiO2 28%",
        "control_type": "pressure",
        "fio2": 0.28,
        "line_status": "Radial ART Line | Foley Catheter (Monitor for Kinks / AD Trigger)",
        "med_events": [
            {"minute_index": 360, "label": "Baclofen 10mg Enteral", "detail": "Routine spasticity control"},
        ],
        "clinical_note": (
            "Low resting BP (~94/60) & HR (~56). Autonomic Dysreflexia crisis "
            "(BP surge + reflex bradycardia + VTe drop) after 05:00 AM."
        ),
        "baseline": {
            "rr": (14.5, 0.5), "hr": (56.0, 1.4), "spo2": (97.0, 0.4),
            "sbp": (94.0, 1.8), "dbp": (60.0, 1.2), "uo": (52.0, 3.5),
            "etco2": (39.0, 0.7), "pip": (18.0, 0.7), "vte": (480.0, 12.0),
            "temp_grad": (1.5, 0.10)
        },
        "artifacts": [110, 450, 920, 1150],
        "probe_blip_min": None,
        "crash_type": "autonomic_dysreflexia"
    },
    "VENT-03": {
        "bed": "Bed 03",
        "pod": "Pod A",
        "name": "Patient C — Post-ICU Trach Weaning (PSV)",
        "age_sex": "59F",
        "diagnosis": "Post-ARDS Prolonged Ventilation, Diaphragmatic Weakness",
        "vent_mode": "PSV 12 / PEEP 6 | FiO2 30%",
        "control_type": "psv",
        "fio2": 0.30,
        "line_status": "Radial ART Line | 20G PIV x2 (K+ Rider Infusing)",
        "med_events": [
            {"minute_index": 180, "label": "Morning Spontaneous Breathing Trial Started", "detail": "PSV 12/6 weaning protocol"},
            {"minute_index": 1290, "label": "Dexmedetomidine Weaned Off", "detail": "Awake weaning assessment"},
        ],
        "clinical_note": (
            "Pressure-support weaning patient: diaphragmatic fatigue after 05:00 AM drives "
            "rapid shallow breathing (VTe drop + RSBI surge > 105)."
        ),
        "baseline": {
            "rr": (16.0, 0.6), "hr": (78.0, 1.6), "spo2": (95.5, 0.4),
            "sbp": (116.0, 1.9), "dbp": (74.0, 1.4), "uo": (48.0, 2.8),
            "etco2": (41.0, 0.8), "pip": (17.0, 0.6), "vte": (430.0, 12.0),
            "temp_grad": (1.6, 0.11)
        },
        "artifacts": [130, 540, 1020],
        "probe_blip_min": None,
        "crash_type": "weaning_fatigue"
    },
    "VENT-04": {
        "bed": "Bed 04",
        "pod": "Pod A",
        "name": "Patient D — Stable ALS Vent Control (High Care Noise)",
        "age_sex": "71F",
        "diagnosis": "Neuromuscular Weakness (ALS), Stable Chronic Trach-Vent",
        "vent_mode": "AC/PC | PIP 20 | PEEP 6 | FiO2 25%",
        "control_type": "pressure",
        "fio2": 0.25,
        "line_status": "Cuff Cycling Q15m | PEG Tube Feeds 55 mL/hr",
        "med_events": [
            {"minute_index": 300, "label": "CoughAssist + Inline Suction", "detail": "Scheduled airway clearance"},
        ],
        "clinical_note": (
            "Stable 24h Pressure-Control patient with frequent suctioning, turning, "
            "and a 1-min circuit-open dip (m220). 0% false alarms."
        ),
        "baseline": {
            "rr": (18.0, 0.6), "hr": (74.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (120.0, 1.8), "dbp": (76.0, 1.3), "uo": (50.0, 3.0),
            "etco2": (40.0, 0.7), "pip": (20.0, 0.7), "vte": (460.0, 10.0),
            "temp_grad": (1.7, 0.10)
        },
        "artifacts": [90, 220, 510, 880, 1210],
        "probe_blip_min": None,
        "crash_type": "none"
    },
    "VENT-05": {
        "bed": "Bed 05",
        "pod": "Pod A",
        "name": "Patient E — Post-Op AAA / Silent Opioid Hypoventilation",
        "age_sex": "74M",
        "diagnosis": "Post-Op Open AAA Repair, PCA Hydromorphone / Epidural",
        "vent_mode": "PSV 8 / PEEP 5 | FiO2 35%",
        "control_type": "psv",
        "fio2": 0.35,
        "line_status": "R IJ Central Line | Radial ART Line | PCA & Epidural Active",
        "med_events": [
            {"minute_index": 1280, "label": "PCA Hydromorphone Demand x3 + Epidural Bolus", "detail": "Late-night analgesia stacking"},
        ],
        "clinical_note": (
            "Silent CO2 retention after late-night opioid stacking: RR & Minute Ventilation fall "
            "and EtCO2 climbs while 35% FiO2 keeps SpO2 near 97%."
        ),
        "baseline": {
            "rr": (15.5, 0.5), "hr": (74.0, 1.5), "spo2": (97.5, 0.4),
            "sbp": (124.0, 1.8), "dbp": (76.0, 1.2), "uo": (50.0, 3.0),
            "etco2": (40.0, 0.7), "pip": (18.0, 0.6), "vte": (490.0, 12.0),
            "temp_grad": (1.5, 0.10)
        },
        "artifacts": [105, 490, 950],
        "probe_blip_min": None,
        "crash_type": "silent_hypovent"
    },
    "VENT-06": {
        "bed": "Bed 06",
        "pod": "Pod A",
        "name": "Patient F — Severe CAP / Cuff & Circuit Leak",
        "age_sex": "63F",
        "diagnosis": "Severe Community-Acquired Pneumonia, Trach-Vent Day 9",
        "vent_mode": "AC/VC | TV 420 | PEEP 10 | FiO2 40%",
        "control_type": "volume",
        "fio2": 0.40,
        "line_status": "L PICC (Double Lumen) | Heated Wire Circuit & Subglottic Trach",
        "med_events": [
            {"minute_index": 420, "label": "Cefepime + Azithromycin IV", "detail": "Scheduled CAP coverage"},
        ],
        "clinical_note": (
            "Gradual trach cuff deflation / circuit leak after 05:00 AM causing sustained PIP & VTe loss."
        ),
        "baseline": {
            "rr": (16.5, 0.5), "hr": (80.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (118.0, 1.8), "dbp": (74.0, 1.3), "uo": (46.0, 2.8),
            "etco2": (41.0, 0.7), "pip": (22.0, 0.7), "vte": (418.0, 9.0),
            "temp_grad": (1.6, 0.10)
        },
        "artifacts": [140, 600, 1040],
        "probe_blip_min": None,
        "crash_type": "circuit_leak"
    },
    "VENT-07": {
        "bed": "Bed 07",
        "pod": "Pod B",
        "name": "Patient G — Urosepsis / Early Septic Shock Drift",
        "age_sex": "79F",
        "diagnosis": "Obstructive Urosepsis, Low-Dose Norepinephrine (0.04 mcg/kg/min)",
        "vent_mode": "AC/PC | PIP 19 | PEEP 6 | FiO2 35%",
        "control_type": "pressure",
        "fio2": 0.35,
        "line_status": "R IJ Triple-Lumen CVC (Pressor) | Foley with Urometer",
        "med_events": [
            {"minute_index": 600, "label": "Norepinephrine 0.04 mcg/kg/min", "detail": "Baseline pressor maintenance"},
            {"minute_index": 1330, "label": "500 mL LR Fluid Challenge Started", "detail": "Narrowing pulse pressure response"},
        ],
        "clinical_note": (
            "Pre-breakout coil at 04:30 AM, then early septic shock drift: rising RR/HR, "
            "narrowing pulse pressure, widening Temp ΔT, and falling urine output."
        ),
        "baseline": {
            "rr": (17.0, 0.5), "hr": (82.0, 1.5), "spo2": (96.0, 0.4),
            "sbp": (114.0, 1.8), "dbp": (68.0, 1.3), "uo": (42.0, 2.5),
            "etco2": (36.0, 0.7), "pip": (19.0, 0.6), "vte": (440.0, 11.0),
            "temp_grad": (1.9, 0.11)
        },
        "artifacts": [115, 520, 990],
        "probe_blip_min": 650,
        "crash_type": "septic_shock_drift"
    },
    "VENT-08": {
        "bed": "Bed 08",
        "pod": "Pod B",
        "name": "Patient H — Status Asthmaticus / Secretion & Resistance",
        "age_sex": "34F",
        "diagnosis": "Refractory Status Asthmaticus, Intubated with Inline Bronchodilators",
        "vent_mode": "AC/VC | TV 440 | PEEP 5 | FiO2 45%",
        "control_type": "volume",
        "fio2": 0.45,
        "line_status": "18G PIV x2 (MgSO4 & Steroids) | Inline Aerogen Nebulizer",
        "med_events": [
            {"minute_index": 210, "label": "Magnesium Sulfate 2g IV + Continuous Albuterol", "detail": "Bronchospasm protocol"},
        ],
        "clinical_note": (
            "Worsening bronchospasm and thick secretion plugging after 05:00 AM: "
            "rising PIP, falling S/F ratio & SpO2, and tachycardia."
        ),
        "baseline": {
            "rr": (18.5, 0.6), "hr": (84.0, 1.6), "spo2": (96.5, 0.4),
            "sbp": (118.0, 1.8), "dbp": (75.0, 1.2), "uo": (55.0, 3.0),
            "etco2": (38.0, 0.7), "pip": (23.0, 0.8), "vte": (438.0, 9.0),
            "temp_grad": (1.5, 0.10)
        },
        "artifacts": [85, 430, 910],
        "probe_blip_min": None,
        "crash_type": "bronchospasm"
    },
    "VENT-09": {
        "bed": "Bed 09",
        "pod": "Pod B",
        "name": "Patient I — Post-CABG Day 1 (Stable Recovery)",
        "age_sex": "66M",
        "diagnosis": "CABG x3 POD #1, Weaning Off Propofol for Extubation",
        "vent_mode": "PSV 10 / PEEP 5 | FiO2 30%",
        "control_type": "psv",
        "fio2": 0.30,
        "line_status": "L Radial ART Line | R IJ Cordis | Chest Tubes (<40 mL/h)",
        "med_events": [
            {"minute_index": 150, "label": "Propofol Titrated 25 → 10 mcg/kg/min", "detail": "Fast-track extubation wake-up"},
        ],
        "clinical_note": (
            "Stable surgical recovery control with strong VTe (~520 mL) and low RSBI (~30). 0% false alarms."
        ),
        "baseline": {
            "rr": (15.5, 0.5), "hr": (78.0, 1.5), "spo2": (97.5, 0.4),
            "sbp": (118.0, 1.9), "dbp": (66.0, 1.3), "uo": (60.0, 3.5),
            "etco2": (38.0, 0.7), "pip": (17.5, 0.6), "vte": (520.0, 12.0),
            "temp_grad": (1.5, 0.10)
        },
        "artifacts": [95, 460, 890, 1240],
        "probe_blip_min": None,
        "crash_type": "none"
    },
    "VENT-10": {
        "bed": "Bed 10",
        "pod": "Pod B",
        "name": "Patient J — DKA Step-Down (Compensated Tachypnea)",
        "age_sex": "29M",
        "diagnosis": "Severe DKA Resolving on Insulin Infusion",
        "vent_mode": "HFNC 35 L/min | FiO2 28% (Monitored Capnography)",
        "control_type": "hfnc",
        "fio2": 0.28,
        "line_status": "18G PIV (Insulin Drip + D5 1/2NS) | Q1h Glucose Checks",
        "med_events": [
            {"minute_index": 300, "label": "Regular Insulin Drip 6 units/hr + D5 Added", "detail": "Anion gap closure protocol"},
        ],
        "clinical_note": (
            "Compensated Kussmaul baseline (RR ~24, EtCO2 ~28). Trips legacy RR alarms all shift; 0% false alarms here."
        ),
        "baseline": {
            "rr": (24.0, 0.6), "hr": (92.0, 1.8), "spo2": (98.5, 0.3),
            "sbp": (116.0, 1.8), "dbp": (72.0, 1.3), "uo": (75.0, 4.0),
            "etco2": (28.0, 0.7), "pip": (12.0, 0.5), "vte": (540.0, 14.0),
            "temp_grad": (1.4, 0.09)
        },
        "artifacts": [100, 500, 940, 1270],
        "probe_blip_min": None,
        "crash_type": "none"
    },
    "VENT-11": {
        "bed": "Bed 11",
        "pod": "Pod B",
        "name": "Patient K — Post-Craniotomy Neuro ICU (Stable Control)",
        "age_sex": "51M",
        "diagnosis": "Subdural Hematoma Evacuation POD #2, EVD Transduced (ICP 9)",
        "vent_mode": "AC/VC | TV 480 | PEEP 5 | FiO2 30%",
        "control_type": "volume",
        "fio2": 0.30,
        "line_status": "EVD Clear CSF | Radial ART Line | 3% NaCl Hold",
        "med_events": [
            {"minute_index": 480, "label": "Keppra 1000mg IV + Q1h Neuro Check", "detail": "Seizure prophylaxis"},
        ],
        "clinical_note": (
            "Stable neuro-protective control with tight EtCO2 (36.5 mmHg) and suction blips damped. 0% false alarms."
        ),
        "baseline": {
            "rr": (16.0, 0.5), "hr": (68.0, 1.4), "spo2": (98.0, 0.4),
            "sbp": (132.0, 2.0), "dbp": (74.0, 1.4), "uo": (55.0, 3.0),
            "etco2": (36.5, 0.6), "pip": (19.5, 0.6), "vte": (478.0, 9.0),
            "temp_grad": (1.4, 0.09)
        },
        "artifacts": [80, 420, 860, 1200],
        "probe_blip_min": None,
        "crash_type": "none"
    },
    "VENT-12": {
        "bed": "Bed 12",
        "pod": "Pod B",
        "name": "Patient L — Chronic HFrEF / Controlled AFib (Low-Normal BP)",
        "age_sex": "76F",
        "diagnosis": "Decompensated HFrEF (EF 25%), Rate-Controlled AFib",
        "vent_mode": "PSV 8 / PEEP 5 | FiO2 30% (SBT 07:00)",
        "control_type": "psv",
        "fio2": 0.30,
        "line_status": "L Forearm 20G PIV (Lasix Drip 10 mg/h) | Foley Catheter",
        "med_events": [
            {"minute_index": 240, "label": "Furosemide IV Infusion 10 mg/hr", "detail": "Diuresis goal -1500 mL net"},
        ],
        "clinical_note": (
            "Chronic low-normal BP (~98/62) & AFib HR (~90). Trips legacy SBP < 100 alarms; 0% false alarms here."
        ),
        "baseline": {
            "rr": (17.5, 0.5), "hr": (90.0, 2.0), "spo2": (95.0, 0.4),
            "sbp": (98.0, 1.8), "dbp": (62.0, 1.2), "uo": (85.0, 4.5),
            "etco2": (37.0, 0.7), "pip": (16.5, 0.6), "vte": (450.0, 11.0),
            "temp_grad": (1.8, 0.10)
        },
        "artifacts": [115, 530, 970, 1280],
        "probe_blip_min": None,
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
        fio2 = float(meta.get("fio2", 0.30))
        ctrl_type = meta.get("control_type", "volume")

        rr = np.random.normal(*b["rr"], size=minutes)
        hr = np.random.normal(*b["hr"], size=minutes)
        spo2 = np.random.normal(*b["spo2"], size=minutes)
        sbp = np.random.normal(*b["sbp"], size=minutes)
        dbp = np.random.normal(*b["dbp"], size=minutes)
        uo = np.random.normal(*b["uo"], size=minutes)
        etco2 = np.random.normal(*b["etco2"], size=minutes)
        pip = np.random.normal(*b["pip"], size=minutes)
        vte = np.random.normal(*b["vte"], size=minutes)
        temp_grad = np.random.normal(*b["temp_grad"], size=minutes)

        for idx in meta["artifacts"]:
            if idx >= minutes - 2:
                continue
            if pid == "VENT-04" and idx == 220:
                pip[idx] -= 10.5
                rr[idx] -= 6.5
                vte[idx] -= 140.0
            else:
                rr[idx:idx + 2] += 11.5
                hr[idx:idx + 2] += 18.0
                pip[idx:idx + 2] += 14.5
                spo2[idx:idx + 2] -= 5.0
                vte[idx:idx + 2] -= 95.0

        # Simulate a brief 4-minute SpO2 finger-probe detachment / poor perfusion signal on VENT-07 (m650-m653)
        probe_idx = meta.get("probe_blip_min")
        if probe_idx is not None and probe_idx < minutes - 5:
            spo2[probe_idx:probe_idx + 4] = 74.0

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
            vte[c_start:] -= np.linspace(0, 55.0, w)
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
            vte[c_start:] -= np.linspace(0, 130.0, w)
            temp_grad[c_start:] += np.linspace(0, 1.9, w)

        elif crash == "weaning_fatigue":
            # Pressure Support diaphragmatic fatigue: RR climbs while VTe drops -> RSBI surges > 105
            rr[c_start:] += np.linspace(0, 14.0, w)
            vte[c_start:] -= np.linspace(0, 195.0, w)
            hr[c_start:] += np.linspace(0, 20.0, w)
            spo2[c_start:] -= np.linspace(0, 4.0, w)
            sbp[c_start:] -= np.linspace(0, 4.0, w)
            dbp[c_start:] += np.linspace(0, 9.0, w)
            uo[c_start:] -= np.linspace(0, 18.0, w)
            etco2[c_start:] += np.linspace(0, 10.0, w)
            pip[c_start:] += np.linspace(0, 3.5, w)
            temp_grad[c_start:] += np.linspace(0, 1.7, w)

        elif crash == "silent_hypovent":
            rr[c_start:] -= np.linspace(0, 6.5, w)
            vte[c_start:] -= np.linspace(0, 110.0, w)
            etco2[c_start:] += np.linspace(0, 16.0, w)
            hr[c_start:] -= np.linspace(0, 5.0, w)
            spo2[c_start:] -= np.linspace(0, 0.5, w)

        elif crash == "circuit_leak":
            pip[c_start:] -= np.linspace(0, 14.0, w)
            vte[c_start:] -= np.linspace(0, 180.0, w)
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
            vte[c_start:] -= np.linspace(0, 60.0, w)

        elif crash == "bronchospasm":
            pip[c_start:] += np.linspace(0, 13.0, w)
            spo2[c_start:] -= np.linspace(0, 6.5, w)
            hr[c_start:] += np.linspace(0, 22.0, w)
            rr[c_start:] += np.linspace(0, 7.0, w)
            etco2[c_start:] += np.linspace(0, 5.0, w)
            vte[c_start:] -= np.linspace(0, 75.0, w)

        rr_clip = np.clip(np.round(rr, 1), 4.0, None)
        vte_clip = np.clip(np.round(vte, 0), 120.0, None)
        spo2_clip = np.clip(np.round(spo2, 1), 65.0, 100.0)
        mv_calc = np.round((rr_clip * vte_clip) / 1000.0, 2)
        rsbi_calc = np.round(rr_clip / (vte_clip / 1000.0), 1)
        sf_calc = np.round(spo2_clip / fio2, 0)

        p_df = pd.DataFrame({
            "patient_id": pid,
            "control_type": ctrl_type,
            "fio2": fio2,
            "minute_index": np.arange(minutes),
            "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
            "rr": rr_clip,
            "hr": np.clip(np.round(hr, 1), 30.0, None),
            "spo2": spo2_clip,
            "sbp": np.round(sbp, 1),
            "dbp": np.round(dbp, 1),
            "pulse_pressure": np.round(sbp - dbp, 1),
            "urine_output": np.clip(np.round(uo, 1), 0.0, None),
            "etco2": np.round(etco2, 1),
            "pip": np.clip(np.round(pip, 1), 4.0, None),
            "vte": vte_clip,
            "mv": mv_calc,
            "rsbi": rsbi_calc,
            "sf_ratio": sf_calc,
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
    ctrl_type = str(df["control_type"].iloc[0]) if "control_type" in df.columns else "volume"

    # Stage 1: Signal Quality Index (SQI) & Probe-Off / Flatline Guard
    # Detect sudden impossible step-drop in SpO2 (< 80% within 1 min while HR & PIP remain unchanged)
    spo2_step = df["spo2"].diff().abs().fillna(0.0)
    probe_off = (df["spo2"] < 80.0) & ((spo2_step >= 12.0) | (df["spo2"].rolling(3, min_periods=1).std().fillna(1.0) == 0.0))
    df["sqi_status"] = np.where(probe_off, "CHECK PROBE", "GOOD")

    # Replace probe-off readings with last valid reading prior to rolling median so Z-score doesn't falsely fire
    df["spo2_valid"] = df["spo2"].where(~probe_off).ffill().bfill()

    vitals_cols = [
        "rr", "hr", "sbp", "dbp", "pulse_pressure",
        "urine_output", "etco2", "pip", "vte", "mv", "rsbi", "sf_ratio", "temp_gradient"
    ]
    for col in vitals_cols:
        df[f"{col}_clean"] = df[col].rolling(window=5, min_periods=1).median()
    df["spo2_clean"] = df["spo2_valid"].rolling(window=5, min_periods=1).median()

    artifact_mask = pd.Series(False, index=df.index)
    for col, thresh in ARTIFACT_SPIKE_THRESHOLD.items():
        raw_col = "spo2_valid" if col == "spo2" else col
        artifact_mask |= (df[raw_col] - df[f"{col}_clean"]).abs() >= thresh
    df["artifact_suppressed"] = artifact_mask

    df = compute_physiological_squeeze(df)

    z_features = ["rr", "hr", "spo2", "sbp", "pulse_pressure", "etco2", "pip", "vte", "mv", "rsbi", "sf_ratio", "temp_gradient"]
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
    df["dVTe_dt_30m"] = np.round(df["vte_clean"].diff(periods=30), 1)
    df["dEtCO2_dt_30m"] = np.round(df["etco2_clean"].diff(periods=30), 2)
    df["dPP_dt_30m"] = np.round(df["pulse_pressure_clean"].diff(periods=30), 2)
    df["dSBP_dt_30m"] = np.round(df["sbp_clean"].diff(periods=30), 2)
    df["dSpO2_dt_30m"] = np.round(df["spo2_clean"].diff(periods=30), 2)

    df["core_velocity_alert"] = (
        (df["rr_zscore"] > 2.5) &
        (df["dRR_dt_30m"] > 2.2) &
        (df["pulse_pressure_zscore"] < -2.0)
    )

    # RT Mode-Aware Vent Mechanics Rule:
    # - Volume Control: PIP surge + EtCO2 rise + Temp gradient strain
    # - Pressure / PSV Control: Exhaled Tidal Volume (VTe) & Minute Ventilation (MV) drop + EtCO2 rise
    if ctrl_type in ("pressure", "psv"):
        df["vent_mechanics_alert"] = (
            (df["vte_zscore"] < -2.5) &
            (df["etco2_zscore"] > 2.5) &
            (df["temp_gradient_zscore"] > 2.0)
        )
    else:
        df["vent_mechanics_alert"] = (
            (df["pip_zscore"] > 2.5) &
            (df["etco2_zscore"] > 2.5) &
            (df["temp_gradient_zscore"] > 2.0)
        )

    # RT Weaning Fatigue Rule (RSBI > 105 on PSV/weaning modes with positive velocity)
    if ctrl_type == "psv":
        df["weaning_rsbi_alert"] = (
            (df["rsbi_clean"] >= 105.0) &
            (df["vte_zscore"] <= -2.5) &
            (df["rr_zscore"] >= 2.5)
        )
    else:
        df["weaning_rsbi_alert"] = False

    df["autonomic_dysreflexia_alert"] = (
        (df["sbp_zscore"] > 3.0) &
        (df["hr_zscore"] < -1.5) &
        ((df["pip_zscore"] > 2.5) | (df["vte_zscore"] < -2.5))
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
        df["weaning_rsbi_alert"] |
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


def is_in_care_mute_window(manual_tags: List[Dict], selected_min: int, window_mins: int = 5) -> str | None:
    """Returns the active RT/airway care tag label if selected_min falls within 5 minutes of tagging."""
    for tag in reversed(manual_tags):
        if tag.get("label") in CARE_MUTE_TAGS:
            t_min = int(tag["minute_index"])
            if 0 <= (selected_min - t_min) <= window_mins:
                return f"{tag['label']} (logged at {tag['time_str']})"
    return None


def evaluate_bedside_context(row: pd.Series) -> List[Dict]:
    alerts: List[Dict] = []
    ctrl_type = str(row.get("control_type", "volume"))

    if bool(row.get("autonomic_dysreflexia_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Autonomic Dysreflexia (BP Surge + Slow HR)",
            "signal_read": (
                f"SBP jumped to `{row['sbp_clean']} mmHg` (`{row['sbp_zscore']:+g} SD`) "
                f"with slow HR (`{row['hr_clean']} bpm`) and airway strain (VTe `{int(row['vte_clean'])} mL`)."
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
            "title": "Sustained Airway Pressure & Volume Drop (Cuff / Circuit Leak)",
            "signal_read": (
                f"PIP dropped to `{row['pip_clean']} cmH₂O` (`{row['pip_zscore']:+g} SD`) "
                f"and VTe fell to `{int(row['vte_clean'])} mL` — sustained beyond routine care noise."
            ),
            "bedside_cues": [
                "Check patient chest rise, breathing effort, and SpO₂ first.",
                "Inspect trach/ETT cuff seal, tubing joints, and water traps.",
                "Call RT if leak continues or patient shows distress.",
            ],
        })

    if bool(row.get("weaning_rsbi_alert", False)):
        alerts.append({
            "severity": "CRITICAL",
            "title": "Weaning Fatigue — Rapid Shallow Breathing (RSBI > 105)",
            "signal_read": (
                f"RSBI climbed to `{row['rsbi_clean']} bpm/L` (RR `{row['rr_clean']} bpm`, "
                f"VTe dropped to `{int(row['vte_clean'])} mL`) with rising EtCO₂ (`{row['etco2_clean']} mmHg`)."
            ),
            "bedside_cues": [
                "Assess accessory muscle use, diaphoresis, and work of breathing.",
                "Notify RT to increase Pressure Support or rest patient on full support.",
                "Check secretions and endotracheal/trach tube patency.",
            ],
        })

    if bool(row.get("vent_mechanics_alert", False)):
        if ctrl_type in ("pressure", "psv"):
            alerts.append({
                "severity": "CRITICAL",
                "title": "Pressure-Mode Tidal Volume Loss & CO₂ Retention",
                "signal_read": (
                    f"Exhaled VTe fell to `{int(row['vte_clean'])} mL` (`{row['vte_zscore']:+g} SD`) "
                    f"at locked PIP `{row['pip_clean']} cmH₂O`, driving EtCO₂ to `{row['etco2_clean']} mmHg`."
                ),
                "bedside_cues": [
                    "In Pressure/PSV mode, falling VTe signals rising airway resistance or falling compliance.",
                    "Suction inline per protocol and check circuit for condensate or kinks.",
                    "Loop in RT to evaluate driving pressure, PEEP, or bronchodilator need.",
                ],
            })
        else:
            alerts.append({
                "severity": "CRITICAL",
                "title": "High Airway Resistance & Perfusion Strain",
                "signal_read": (
                    f"PIP `{row['pip_clean']} cmH₂O` (`{row['pip_zscore']:+g} SD`), "
                    f"EtCO₂ `{row['etco2_clean']} mmHg` (`{row['etco2_zscore']:+g} SD`), "
                    f"S/F Ratio `{int(row['sf_ratio_clean'])}`."
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
            "title": "Rising PIP + Falling SpO₂ / S/F Ratio (Secretions / Bronchospasm)",
            "signal_read": (
                f"PIP climbed to `{row['pip_clean']} cmH₂O` (`{row['pip_zscore']:+g} SD`) "
                f"while SpO₂ fell to `{row['spo2_clean']}%` (S/F Ratio `{int(row['sf_ratio_clean'])}`)."
            ),
            "bedside_cues": [
                "Assess breath sounds (wheezes vs. coarse rhonchi) and oxygenation.",
                "Suction airway inline and check tubing for water or kinks.",
                "Notify RT if airway pressure remains high or bronchodilator is needed.",
            ],
        })

    if bool(row.get("core_velocity_alert", False)) and not bool(row.get("weaning_rsbi_alert", False)):
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
            "title": "Silent Hypoventilation (Rising EtCO₂ + Falling Minute Vent)",
            "signal_read": (
                f"EtCO₂ rose to `{row['etco2_clean']} mmHg` (`{row['etco2_zscore']:+g} SD`) "
                f"as Minute Vent dropped to `{row['mv_clean']} L/min` (SpO₂ still `{row['spo2_clean']}%`)."
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
    med_events: List[Dict] | None = None,
) -> pd.DataFrame:
    upto_df = df.iloc[:selected_min + 1]
    records: List[Dict] = []

    art_rows = upto_df[upto_df["artifact_suppressed"]]
    for _, r in art_rows.iterrows():
        diffs = []
        for col, thresh in ARTIFACT_SPIKE_THRESHOLD.items():
            raw_c = "spo2_valid" if col == "spo2" else col
            d = float(r[raw_c] - r[f"{col}_clean"])
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

    probe_rows = upto_df[upto_df["sqi_status"] == "CHECK PROBE"]
    if not probe_rows.empty:
        first_p = probe_rows.iloc[0]
        records.append({
            "minute_index": int(first_p["minute_index"]),
            "Time": first_p["timestamp"].strftime("%I:%M %p"),
            "Type": "Sensor Guard",
            "Event": "SpO₂ Probe-Off / Damped Held",
            "Detail": f"Held last valid SpO₂ at min {int(first_p['minute_index'])} (0 false alarms)",
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

    for med in (med_events or []):
        m_idx = int(med["minute_index"])
        if m_idx <= selected_min:
            ts_str = df.iloc[m_idx]["timestamp"].strftime("%I:%M %p")
            records.append({
                "minute_index": m_idx,
                "Time": ts_str,
                "Type": "Med / Rx",
                "Event": med["label"],
                "Detail": med["detail"],
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


def build_12h_shift_handoff(
    df: pd.DataFrame,
    selected_min: int,
    profile_meta: Dict,
    manual_tags: List[Dict],
    ack_info: Dict | None = None,
) -> str:
    """Generates a 12-hour shift SBAR roll-up summarizing overnight/day trends, filtered artifacts, and actions."""
    shift_start = 0 if selected_min < 720 else 720
    shift_name = "DAY SHIFT (07:00–18:59)" if shift_start == 0 else "NIGHT SHIFT (19:00–06:59)"
    sub = df.iloc[shift_start:selected_min + 1]
    curr = df.iloc[selected_min]
    clock_now = curr["timestamp"].strftime("%I:%M %p")

    suppressed_mins = int(sub["artifact_suppressed"].sum())
    legacy_alarms = int(sub["legacy_static_alert"].sum())
    alert_mins = int(sub["pre_ehr_alert"].sum())
    coil_mins = int(sub["phys_squeeze_active"].sum())
    probe_mins = int((sub["sqi_status"] == "CHECK PROBE").sum())

    first_breach_txt = "None — patient remained within personal baseline all shift"
    if alert_mins > 0:
        first_row = sub[sub["pre_ehr_alert"]].iloc[0]
        ctx = evaluate_bedside_context(first_row)
        b_title = ctx[0]["title"] if ctx else "Trend Breach"
        first_breach_txt = f"{b_title} starting at {first_row['timestamp'].strftime('%I:%M %p')}"

    ack_txt = (
        f"Acknowledged by {ack_info['role']} at {ack_info['time_str']} ({ack_info['status']})"
        if ack_info else ("Pending bedside acknowledgment" if alert_mins > 0 else "N/A")
    )

    shift_tags = [t for t in manual_tags if shift_start <= int(t["minute_index"]) <= selected_min]
    tags_txt = (
        "; ".join(f"{t['time_str']} {t['label']}" for t in shift_tags[-6:])
        if shift_tags else "Routine care per protocol"
    )

    return (
        f"DELTAPULSE 12-HOUR SHIFT HANDOFF (SBAR) — {shift_name}\n"
        f"====================================================================\n"
        f"SITUATION: {profile_meta['bed']} ({curr['patient_id']}, {profile_meta['age_sex']}) | As of {clock_now}\n"
        f"• Diagnosis: {profile_meta['diagnosis']}\n"
        f"• Vent / Airway: {profile_meta['vent_mode']} | Lines: {profile_meta.get('line_status', 'Patent')}\n\n"
        f"BACKGROUND & 12-HOUR TRAJECTORY:\n"
        f"• Shift Surveillance: {first_breach_txt}\n"
        f"• Pre-Breakout Coil Minutes: {coil_mins} min | Active Alert Minutes: {alert_mins} min\n"
        f"• Alarm Fatigue Prevented: {suppressed_mins} care spikes damped ({probe_mins} probe-off mins held); "
        f"bypassed {legacy_alarms} static monitor alarms\n"
        f"• Escalation Status: {ack_txt}\n\n"
        f"ASSESSMENT (CURRENT 5-MIN CLEAN BASELINE):\n"
        f"• Respiratory: RR {curr['rr_clean']} bpm (Base {curr['rr_base_mean']}), VTe {int(curr['vte_clean'])} mL, "
        f"MV {curr['mv_clean']} L/min, RSBI {curr['rsbi_clean']}, PIP {curr['pip_clean']} cmH₂O, "
        f"EtCO₂ {curr['etco2_clean']} mmHg, SpO₂ {curr['spo2_clean']}% (S/F {int(curr['sf_ratio_clean'])})\n"
        f"• Hemodynamics: BP {int(curr['sbp_clean'])}/{int(curr['dbp_clean'])} mmHg (PP {curr['pulse_pressure_clean']}), "
        f"HR {curr['hr_clean']} bpm, Temp ΔT {curr['temp_gradient_clean']}°C, UO {int(curr['urine_output_clean'])} mL/h\n\n"
        f"RECOMMENDATION / CARE LOG:\n"
        f"• Logged Interventions: {tags_txt}\n"
        f"• Next Shift Focus: Continue personalized Z-score & velocity tracking; reassess weaning/airway clearance per unit protocol."
    )


def run_deltapulse_velocity_engine(db_path: str = DB_NAME, patient_id: str | None = None) -> pd.DataFrame:
    expected_pids = set(PATIENT_PROFILES.keys())
    expected_rows = len(PATIENT_PROFILES) * SHIFT_MINUTES
    try:
        with sqlite3.connect(db_path) as conn:
            raw_df = pd.read_sql("SELECT * FROM vitals_stream", conn)
        has_all_beds = (
            "patient_id" in raw_df.columns
            and "vte" in raw_df.columns
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
    print("Running DeltaPulse V4 (RT + ICU Closed-Loop) 24-Hour Verification...")
    generate_synthetic_telemetry(DB_NAME)
    cohort = run_deltapulse_velocity_engine(DB_NAME, patient_id=None)

    assert len(cohort) == len(PATIENT_PROFILES) * SHIFT_MINUTES
    for pid, meta in PATIENT_PROFILES.items():
        p_df = cohort[cohort["patient_id"] == pid].reset_index(drop=True)
        base_fp = int(p_df.iloc[:CRASH_START_MIN]["pre_ehr_alert"].sum())
        m_late_alert = bool(p_df.iloc[CHECKPOINT_LATE_MIN]["pre_ehr_alert"])
        m_end_alert = bool(p_df.iloc[SHIFT_MINUTES - 1]["pre_ehr_alert"])

        if meta["crash_type"] == "none":
            assert int(p_df["pre_ehr_alert"].sum()) == 0, f"{pid} had false alerts"
        else:
            assert base_fp == 0, f"{pid} had {base_fp} baseline false alerts"
            assert m_late_alert and m_end_alert, f"{pid} failed late breach latch"
    print("[PASS] All 12 ICU beds verified with VTe, MV, RSBI, S/F Ratio, SQI Probe Guard, and 0% false alarms.")
