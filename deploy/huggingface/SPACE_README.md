---
title: ICU-PREDICT
emoji: 🫀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
short_description: 6-hour ICU deterioration early warning on MIMIC-IV
---

# ICU-PREDICT

A 6-hour early-warning system for ICU patients. Every hour, for every bed, it estimates the chance a patient
deteriorates in the next 6 hours, forecasts each vital sign hour by hour with an 80% range, and explains why.

- **Ward view:** every bed as a mini monitor, most urgent first. Press **Play** to advance the ward hour by hour.
- **Patient view:** vitals, risk and its history, the reasons behind it, the next 6 hours hour by hour, trend charts.
- **Reports:** download a PDF or CSV for any patient.
- **Teaching mode:** "Show what happened next" compares each forecast with what really happened.

Built with MIMIC-IV-format data, LightGBM / XGBoost / logistic regression (plus a GRU benchmark), conformal
forecast ranges, FastAPI and a custom web UI. The hosted demo trains on the open MIMIC-IV Clinical Database Demo
v2.2 (100 patients; Johnson et al., PhysioNet, ODbL) during the build, or on synthetic data if PhysioNet is
unreachable; the footer says which.

**Research prototype. Not a medical device and not for patient care.**
