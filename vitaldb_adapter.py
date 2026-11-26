"""
DeltaPulse VitalDB Real-Telemetry Adapter (Google Colab / Local Python)
Fetches real de-identified multi-parameter patient cases from the open VitalDB
dataset (`pip install vitaldb`), resamples to 1-minute resolution, maps tracks
to DeltaPulse's schema, and runs the 3-stage velocity & squeeze engine.
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import vitaldb
except ImportError as exc:
    raise ImportError("Install VitalDB first: pip install vitaldb") from exc

from database_and_engine import (
    DB_NAME,
    PATIENT_PROFILES,
    _process_single_patient,
    evaluate_bedside_context,
)

# VitalDB track names -> DeltaPulse internal vital columns
VITALDB_TRACK_MAP = {
    "rr": ["Solar8000/RR", "Solar8000/RR_CO2", "Primus/RR_CO2"],
    "hr": ["Solar8000/HR", "Vigileo/HR"],
    "spo2": ["Solar8000/PLETH_SPO2"],
    "sbp": ["Solar8000/ART_SBP", "Solar8000/NIBP_SBP"],
    "dbp": ["Solar8000/ART_DBP", "Solar8000/NIBP_DBP"],
    "etco2": ["Solar8000/ETCO2", "Primus/ETCO2"],
    "pip": ["Primus/PIP", "Primus/PEAK_PRESS"],
    "temp_core": ["Solar8000/BT", "Solar8000/TEMP1"],
}


def find_vitaldb_cases(max_cases: int = 12, min_duration_mins: int = 180) -> list[int]:
    """Queries VitalDB case metadata for cases containing core hemodynamic & airway tracks."""
    required_tracks = ["Solar8000/HR", "Solar8000/PLETH_SPO2", "Primus/PIP", "Primus/ETCO2"]
    case_ids = vitaldb.find_cases(required_tracks)
    selected = []
    for cid in case_ids:
        vf = vitaldb.VitalFile(cid, track_names=["Solar8000/HR"])
        arr = vf.to_numpy(["Solar8000/HR"], interval=60)
        if arr is not None and len(arr) >= min_duration_mins:
            selected.append(int(cid))
        if len(selected) >= max_cases:
            break
    return selected


def load_vitaldb_case_to_deltapulse(
    case_id: int,
    bed_slot: str = "VENT-01",
    start_time: datetime = datetime(2026, 11, 10, 7, 0, 0),
) -> pd.DataFrame:
    """Downloads a single VitalDB case at 1-min resolution and formats it for DeltaPulse."""
    candidate_tracks = [t for group in VITALDB_TRACK_MAP.values() for t in group]
    vf = vitaldb.VitalFile(case_id, track_names=candidate_tracks)
    available = set(vf.get_track_names())

    chosen = {}
    for target_col, aliases in VITALDB_TRACK_MAP.items():
        match = next((a for a in aliases if a in available), None)
        if match:
            chosen[target_col] = match

    if not chosen:
        raise ValueError(f"Case {case_id} has none of the mapped VitalDB tracks.")

    track_list = list(chosen.values())
    data = vf.to_numpy(track_list, interval=60)
    raw = pd.DataFrame(data, columns=list(chosen.keys()))

    raw = raw.ffill(limit=3).bfill(limit=3)
    n = len(raw)

    rr = raw["rr"] if "rr" in raw else pd.Series(16.0, index=range(n))
    hr = raw["hr"] if "hr" in raw else pd.Series(78.0, index=range(n))
    spo2 = raw["spo2"] if "spo2" in raw else pd.Series(97.0, index=range(n))
    sbp = raw["sbp"] if "sbp" in raw else pd.Series(118.0, index=range(n))
    dbp = raw["dbp"] if "dbp" in raw else pd.Series(72.0, index=range(n))
    etco2 = raw["etco2"] if "etco2" in raw else pd.Series(38.0, index=range(n))
    pip = raw["pip"] if "pip" in raw else pd.Series(19.0, index=range(n))

    if "temp_core" in raw and raw["temp_core"].notna().any():
        core_t = raw["temp_core"].fillna(36.8)
        temp_grad = np.clip(1.6 + (36.8 - core_t) * 0.8, 0.8, 4.5)
    else:
        temp_grad = pd.Series(1.6, index=range(n))

    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]
    p_df = pd.DataFrame({
        "patient_id": bed_slot,
        "vitaldb_case_id": case_id,
        "minute_index": np.arange(n),
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "rr": np.clip(np.round(rr.fillna(16.0), 1), 4.0, 60.0),
        "hr": np.clip(np.round(hr.fillna(78.0), 1), 30.0, 220.0),
        "spo2": np.clip(np.round(spo2.fillna(97.0), 1), 65.0, 100.0),
        "sbp": np.clip(np.round(sbp.fillna(118.0), 1), 45.0, 260.0),
        "dbp": np.clip(np.round(dbp.fillna(72.0), 1), 25.0, 160.0),
        "pulse_pressure": np.round(sbp.fillna(118.0) - dbp.fillna(72.0), 1),
        "urine_output": 50.0,
        "etco2": np.clip(np.round(etco2.fillna(38.0), 1), 10.0, 95.0),
        "pip": np.clip(np.round(pip.fillna(19.0), 1), 4.0, 65.0),
        "temp_gradient": np.round(temp_grad, 2),
    })
    return p_df


def run_vitaldb_validation(max_cases: int = 6, write_to_sqlite: bool = False) -> pd.DataFrame:
    """
    Pulls real VitalDB cases, runs DeltaPulse's 3-stage velocity & squeeze engine,
    and prints a clinical audit summary of suppressed artifacts, coils, and alerts.
    """
    print(f"Searching VitalDB for up to {max_cases} multi-parameter ventilated cases (>=3h)...")
    case_ids = find_vitaldb_cases(max_cases=max_cases, min_duration_mins=180)
    print(f"Found VitalDB Case IDs: {case_ids}")

    bed_keys = list(PATIENT_PROFILES.keys())
    raw_parts = []
    processed_parts = []

    for idx, cid in enumerate(case_ids):
        bed_id = bed_keys[idx % len(bed_keys)]
        raw_case = load_vitaldb_case_to_deltapulse(cid, bed_slot=bed_id)
        raw_case["timestamp"] = pd.to_datetime(raw_case["timestamp"])
        proc_case = _process_single_patient(raw_case.copy())

        suppressed = int(proc_case["artifact_suppressed"].sum())
        coil_mins = int(proc_case["phys_squeeze_active"].sum())
        alert_mins = int(proc_case["pre_ehr_alert"].sum())
        legacy_mins = int(proc_case["legacy_static_alert"].sum())

        first_alert_note = "None"
        if alert_mins > 0:
            first_row = proc_case[proc_case["pre_ehr_alert"]].iloc[0]
            ctx = evaluate_bedside_context(first_row)
            first_alert_note = f"min {int(first_row['minute_index'])}: {ctx[0]['title']}" if ctx else f"min {int(first_row['minute_index'])}"

        print(
            f"  - VitalDB Case #{cid} ({len(proc_case)} mins) -> "
            f"Artifacts Suppressed: {suppressed}m | Coil Mins: {coil_mins}m | "
            f"DeltaPulse Alerts: {alert_mins}m vs Legacy Static: {legacy_mins}m | First Breach: {first_alert_note}"
        )
        raw_parts.append(raw_case)
        processed_parts.append(proc_case)

    if write_to_sqlite and raw_parts:
        db_df = pd.concat(raw_parts, ignore_index=True)
        db_df["timestamp"] = db_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_NAME) as conn:
            db_df.drop(columns=["vitaldb_case_id"], errors="ignore").to_sql("vitals_stream", conn, if_exists="replace", index=False)
        print(f"[SAVED] Wrote {len(db_df)} real VitalDB patient-minutes to {DB_NAME}")

    return pd.concat(processed_parts, ignore_index=True) if processed_parts else pd.DataFrame()


if __name__ == "__main__":
    run_vitaldb_validation(max_cases=6, write_to_sqlite=False)
