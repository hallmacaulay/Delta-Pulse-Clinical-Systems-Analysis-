⚡ DeltaPulse: 12-Bed ICU Central Station & Pre-EHR Velocity Radar
DeltaPulse is a clinician-designed, deterministic physiological surveillance engine and 12-bed ICU central station cockpit built to reduce static-threshold alarm fatigue, bedside working-memory burden, and batch-charting EHR documentation lag (\approx 2.5 hours) in high-acuity ventilator and complex care units.
🩺 The Clinical Problem
Bedside ICU nurses and respiratory therapists act as "human middleware"—mentally stitching together disconnected ventilator screens, bedside monitors, and EHR flowsheets across 12-hour shifts. Legacy monitoring fails clinicians in three ways:
Static-Threshold Alarm Fatigue: Population cutoffs (e.g., RR > 20.5, SBP < 100, SpO_2 < 92\%) fire nonstop on chronic COPD retainers, resolving DKA patients, and high-cervical spinal cord injury baselines—while missing dangerous relative shifts (like Autonomic Dysreflexia surging from 94 to 130\text{ mmHg}).
Transient Care Noise: Routine inline suctioning, turning/repositioning, coughing fits, and brief circuit-open dips throw sharp spikes into raw telemetry, conditioning staff to tune out alarms.
Working-Memory & Charting Burden: Reconstructing what moved together—and whether a blip happened 10 minutes or 2 hours ago—turns acute troubleshooting and end-of-shift charting into an exhausting memory test.
🏗️ System Architecture
DeltaPulse processes 1-minute resolution telemetry across a 12-bed ICU pod (4,320 patient-minutes per 6-hour shift) through a deterministic 3-stage pipeline paired with an explainable ML benchmark layer:
Raw 1-Min Telemetry (SQLite: aeropulse_vitals.db)
        │
        ▼
[Stage 1: Bidirectional Artifact Suppression]
  • 5-min rolling median filter across 10 physiological channels
  • Suppresses <120s suctioning, turning, coughing spikes & circuit-open dips
        │
        ▼
[Stage 2: Self-Calibrating 240-Min Personal Baselines]
  • 60-min lagged reference window ("Boiling Frog" guard)
  • Personal rolling Z-scores with per-vital variance floors
        │
        ▼
[Stage 3: 30-Min Velocity & 6 Co-Moving Clinical Rules]
  • Evaluates multi-vital physiological trajectories + bedside cues
        │
        ├──────────────────────────────┐
        ▼                              ▼
[Explainable ML Benchmark]    [ICU Cockpit & Multi-EHR Scribe]
  • RF vs. L2 LogReg vs. GBDT   • 3x4 Pod Grid & Philips Neon Mode
  • Minute-level driver scores  • Quick-Tag Logger & Shift Event Strip
                                • Epic / Cerner / HL7 FHIR R4 Export


Stage 1: Bidirectional Artifact Suppression
A 5-minute rolling median filter acts as a general-purpose outlier damper, flattening both upward spikes (inline suctioning, coughing/asynchrony, turning) and downward dips (brief circuit openings, probe blips) while flagging suppressed minutes cleanly in the shift audit trail.
Stage 2: Lagged 240-Minute Personal Z-Scores ("Boiling Frog" Guard)
Rather than comparing patients to population norms, each vital is scored against the patient's own 4-hour rolling baseline, lagged by 60 minutes so slow clinical deterioration cannot contaminate its own reference mean and standard deviation:
Z_t = \frac{x_{\text{clean}, t} - \mu_{\text{lag } 60, 240\text{m}}}{\max(\sigma_{\text{lag } 60, 240\text{m}}, \, \sigma_{\text{floor}})}
Minimum standard-deviation floors (\sigma_{\text{floor}}) prevent flat-line windows from dividing by zero or exploding Z-scores on clinically negligible fluctuations.
Stage 3: 30-Minute Velocity & Co-Moving Clinical Rules
First-order 30-minute differentials (dRR/dt, dPIP/dt, dEtCO_2/dt, dPP/dt, dSBP/dt, dSpO_2/dt) are paired with Z-score thresholds to detect six multi-parameter failure patterns:
Autonomic Dysreflexia Crisis: Acute SBP surge (Z > +3.0) + reflex bradycardia (HR Z < -1.5) + rising PIP.
Sustained Peak Pressure Drop (Leak/Disconnect): Sustained PIP drop (Z \le -2.2 and 30m velocity \le -3.0\text{ cmH}_2\text{O}).
Ventilator Airway Resistance & Perfusion Strain: Rising PIP + rising EtCO_2 + widening core-peripheral temperature gradient (\Delta T).
Secretion Burden / Bronchospasm: Sustained PIP climb (Z \ge +2.2) paired with SpO_2 erosion (Z \le -2.0).
Hemodynamic & Tachypneic Velocity Creep: Rising RR + positive 30m velocity + narrowing pulse pressure (Z < -2.0).
Silent Hypoventilation (CO_2 Retention): Climbing EtCO_2 (Z \ge +2.2) + falling RR (Z \le -1.8) even when supplemental O_2 keeps SpO_2 deceptively steady.
🏥 The 12-Bed ICU Pod Cohort (VENT-01 – VENT-12)
Designed to mirror a standard 8–15 bed ICU pod, the cohort pairs 7 active deterioration archetypes with 5 challenging-baseline stable controls (0\% false-positive alarms):
Pod
Bed / ID
Clinical Profile
Modeled Trajectory (crash_type)
Pod A
Bed 01 (VENT-01)
68M — Chronic COPD / Trach-Vent
High baseline (RR \approx 22, EtCO_2 \approx 52, SpO_2 \approx 90.5\%); suction artifacts + mucus plugging & sepsis (mucus_sepsis)
Pod A
Bed 02 (VENT-02)
42M — C4 Tetraplegia / Neurogenic
Low baseline (BP \approx 94/60, HR \approx 56); turning noise + Autonomic Dysreflexia surge (autonomic_dysreflexia)
Pod A
Bed 03 (VENT-03)
59F — Post-ICU Trach Weaning (PSV)
Suction spike + progressive diaphragmatic fatigue & tachypnea (weaning_fatigue)
Pod A
Bed 04 (VENT-04)
71F — Stable ALS Vent Control
Upward care spikes + downward circuit-open dip (m220); 0\% false alerts (none)
Pod A
Bed 05 (VENT-05)
74M — Post-Op AAA / PCA Opioid
Falling RR + rising EtCO_2 masked by flat SpO_2 \approx 97.5\% (silent_hypovent)
Pod A
Bed 06 (VENT-06)
63F — Severe CAP / Trach-Vent Day 9
Gradual cuff deflation / circuit leak with sustained PIP drop (circuit_leak)
Pod B
Bed 07 (VENT-07)
79F — Urosepsis / Early Septic Shock
Slow-boil tachypnea, tachycardia, narrowing PP, widening \Delta T, oliguria (septic_shock_drift)
Pod B
Bed 08 (VENT-08)
34F — Status Asthmaticus (Intubated)
Rising PIP + SpO_2 erosion + compensatory tachycardia (bronchospasm)
Pod B
Bed 09 (VENT-09)
66M — Post-CABG Day 1 Recovery
Stable surgical control with coughing/turning artifacts suppressed; 0\% false alerts (none)
Pod B
Bed 10 (VENT-10)
29M — DKA Step-Down (HFNC)
Compensated Kussmaul RR \approx 24 & EtCO_2 \approx 28; 0\% false alerts (none)
Pod B
Bed 11 (VENT-11)
51M — Post-Craniotomy Neuro ICU
Stable neuro-protective control with Q1h care/suction blips damped; 0\% false alerts (none)
Pod B
Bed 12 (VENT-12)
76F — Chronic HFrEF / Controlled AFib
Low-normal SBP \approx 98 & HR \approx 90 control; 0\% false alerts (none)

🖥️ Bedside Cognitive Offloading & Interoperability Features
12-Bed Central Station Grid: Filterable 3 \times 4 view (All 12 Beds, Pod A, Pod B, Active Alerts Only) with live unit-wide status counts.
Dual Clinical Themes: Instant toggle between Night-Shift Slate Dark (#0B0F19) and Philips IntelliVue Hardware Neon (#000000).
Assessment-First Bedside Cues: Every active alert pairs the co-moving signal story with 3 scannable physical checks (patient first, airway/circuit second, escalation third).
Quick-Tag Care Logger & "What Just Happened" Event Strip: One-click bedside tags (Suctioning, Circuit / Condensate Clear, Cuff Check, Inline Nebulizer, Turning / Repositioning, Coughing / Asynchrony, Oral Care, Sensor / Probe Blip) merged chronologically with auto-suppressed artifacts and alert onset.
Vendor-Agnostic Multi-EHR Scribe: Generates copy-ready handoff blocks strictly scoped to monitored vitals and logged care context:
Epic Hyperspace (.DELTAPULSE SmartPhrase)
Oracle Health / Cerner PowerChart (,,deltapulse AutoText / iView sub-bands)
HL7 FHIR R4 (Bundle JSON payload)
📂 Repository Structure
DeltaPulse/
├── requirements.txt        # Python dependencies for local & Streamlit Cloud deployment
├── database_and_engine.py  # SQLite store, 12-bed generator, 3-stage pipeline, co-moving rules & self-test
├── ml_classifier.py        # Cohort-wide Random Forest classifier & local driver attribution
├── ml_benchmark.py         # RF vs. L2 Logistic Regression vs. HistGBDT/LightGBM benchmark & UI tab
├── app.py                  # Streamlit 12-bed ICU Central Station cockpit & Multi-EHR Scribe
└── README.md               # Architecture documentation & deployment guide


🚀 Quick Start & Verification
Run Locally
Clone the repository and install dependencies:
git clone https://github.com/<your-username>/DeltaPulse.git
cd DeltaPulse
pip install -r requirements.txt


Run the built-in 12-bed deterministic verification harness:
python3 database_and_engine.py
Verifies all 4,320 patient-minutes, confirms 0\% false alarms across baseline windows and all 5 stable controls, and asserts alert latch from m305 through m359 across all 7 deterioration beds.
Launch the ICU Central Station cockpit:
streamlit run app.py


Deploy on Streamlit Community Cloud
Push all 5 files to a public GitHub repository.
Connect the repository at share.streamlit.io, set Main file path to app.py, and click Deploy.
The app automatically generates aeropulse_vitals.db on startup and caches the scored 12-bed cohort.
🗺️ Architectural Roadmap
Phase 0 (Complete): Unified 5-file stack, 12-bed cohort, 60-min lagged Z-scores, and automated m305–m359 verification harness.
Phase 1 (V1.0 Launch): Public GitHub release and live Streamlit Community Cloud demo.
Phase 2 (V1.5 Rigor & Persistence): Upgrade ml_benchmark.py from in-sample synthetic benchmarking to patient-grouped cross-validation (GroupKFold by patient_id), persist bedside care tags to SQLite, and conduct 5-second bedside usability audits with RTs and ICU nurses.
Phase 3 (V2.0 Live FHIR Sandbox): Read-only HL7 FHIR R4 Observation ingestion from a public test server with automatic local SQLite fallback.
⚠️ Scope Boundaries & Clinical Disclaimer
Strict Parameter Scope: DeltaPulse processes only the continuous physiological and ventilator parameters ingested into the pipeline plus manual bedside care/artifact tags. It intentionally does not attempt to replace EHR charting, MAR/PRN medication documentation, or full nursing narratives.
ML Benchmark Transparency (V1.0): In V1.0, the ML Benchmark tab trains and evaluates models on the same synthetic 12-bed cohort to demonstrate comparative behavior and local driver explainability in real time. Reported AUROC/PPV metrics reflect synthetic in-sample performance; real-world clinical validation requires patient-grouped cross-validation on de-identified ICU datasets (e.g., MIMIC-IV).
Not a Medical Device: This repository is an open-source clinical informatics prototype using 100% synthetic telemetry (zero PHI). It is not FDA-cleared or Health Canada-licensed and must not be used for direct patient care or clinical decision-making.

