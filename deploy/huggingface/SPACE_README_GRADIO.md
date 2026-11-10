---
title: ICU-PREDICT
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: space_app.py
pinned: false
short_description: 6-hour ICU deterioration early warning on MIMIC-IV
---

# ICU-PREDICT

A 6-hour early-warning system for ICU patients. Every hour, for every bed, it estimates the chance a patient
deteriorates in the next 6 hours, forecasts each vital sign hour by hour with an 80% range, and explains why.

- **Ward view:** every bed as a mini monitor, most urgent first. Press **Play** to advance the ward hour by hour.
- **Patient view:** vitals, risk and its history, the reasons behind it, the next 6 hours hour by hour, trend charts.
- **Reports:** download a PDF or CSV for any patient.

The first start downloads the open MIMIC-IV Clinical Database Demo v2.2 (100 patients; Johnson et al., PhysioNet,
ODbL) and trains the models, which takes a few minutes; the footer says whether real or synthetic data is shown.

**Research prototype. Not a medical device and not for patient care.**
