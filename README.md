# Sou-IDS — Machine Learning-Based Network Intrusion Detection System for Small Companies

A complete Flask + scikit-learn mini-project. Upload CICIDS2017-style CSV files, train and compare
**Decision Tree, Random Forest and Gaussian Naive Bayes**, analyse new CSV files in the browser, review alerts,
and download CSV / PDF reports.

> **Honest scope:** Sou-IDS analyses uploaded CSV files **in batches**. It does **not** capture live packets and
> never blocks IPs or changes a network. Login accounts are **demonstration only**, not production authentication.

---
## 1. Folder structure
```
Sou-IDS/
├── app.py              Flask routes, login, CSRF, APIs
├── config.py           All settings (paths, limits, demo users, dataset file names)
├── database.py         SQLite: users, analyses (prediction history), alerts, model versions
├── preprocessing.py    Cleaning, label mapping, pipelines, schema validation
├── train_model.py      Trains 3 models, validation comparison, one-time test evaluation, saves model
├── evaluation.py       Accuracy / precision / recall / F1 / confusion matrix / FP & FN rates
├── prediction.py       Model loading, predictions, summaries, alerts, recommendations
├── reporting.py        PDF report (ReportLab)
├── requirements.txt
├── models/  uploads/  reports/  datasets/  instance/   (created/used at run time)
├── templates/          base, login, dashboard, detect, analytics, alerts, model_evaluation,
│                       recommendations, datasets, error
├── static/css/style.css   static/js/dashboard.js
├── tests/              test_core.py (unit tests) + make_sample_data.py (synthetic demo data)
└── sample_data/        (created by make_sample_data.py)
```

## 2. Windows setup (exact steps)
1. Install **Python 3.10 – 3.12** from https://www.python.org/downloads/ — tick **"Add python.exe to PATH"**.
2. Unzip the project, e.g. to `C:\Sou-IDS`. Open **Command Prompt** and run:
```bat
cd C:\Sou-IDS
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(Your prompt will show `(venv)`. Re-run `venv\Scripts\activate` every time you open a new Command Prompt.)

3. The browser pages load Bootstrap, icons and Plotly from CDNs, so the laptop needs **internet** while presenting.

## 3. Try it first with synthetic data (2 minutes, no download)
```bat
python tests\make_sample_data.py
python train_model.py --files sample_monday_benign.csv sample_wednesday_dos.csv sample_friday_ddos.csv sample_friday_portscan.csv
python app.py
```
Open **http://127.0.0.1:5000**. Upload `sample_data\upload_unlabelled.csv` on *Traffic Detection*.
(The synthetic data is artificial — never quote its accuracy as a real result. Delete the `sample_*.csv` files from
`datasets\` before training on the real dataset.)

## 4. Real CICIDS2017 dataset
1. Download from Kaggle (search “CICIDS2017”; you need a free Kaggle account).
2. Copy these four files into the **`datasets`** folder (names may differ slightly on Kaggle):
   - `Monday-WorkingHours.pcap_ISCX.csv`
   - `Wednesday-workingHours.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
3. If your names differ: sign in as Admin → **Datasets & Training** → choose the right file for each slot
   (or upload it there). The ZIP does **not** contain the dataset.

These four files contain only **Normal, DoS, DDoS and PortScan**. The app shows the classes it actually finds and
never claims to detect Brute Force / Bot / Web Attack / Heartbleed / Infiltration unless a labelled dataset containing
them is used for (re)training.

## 5. Train the models
**Option A – browser:** log in as `admin` → *Datasets & Training* → tick the files → **Train models** (progress log shown).
**Option B – terminal:** `python train_model.py` (uses the mapping / default file names) or
`python train_model.py --files file1.csv file2.csv`.

What happens: load (chunked, sampled) → clean (duplicates, inf, NaN) → map labels → drop identifier columns →
leakage check → **80/20 stratified train/test split** → train part split again into train/validation →
3 models fitted on train, compared on **validation** → best chosen by
`0.4·attack recall + 0.4·attack F1 + 0.2·(1 − false-positive rate)` → best model refitted on train+validation →
evaluated **once** on the untouched **test** set → saved as `models/sou_ids_vN.joblib` and registered in SQLite.

Laptop with little RAM? Lower `MAX_ROWS_PER_FILE` and `MAX_TRAIN_ROWS` in `config.py`.

## 6. Start the app & demo credentials
```bat
venv\Scripts\activate
python app.py
```
Open http://127.0.0.1:5000

| Role | Username | Password |
|---|---|---|
| Admin (can train/retrain) | `admin` | `Admin@12345` |
| Security Analyst | `analyst` | `Analyst@12345` |

**Demonstration-only credentials.** Passwords are stored hashed; sessions + CSRF protection are enabled; the secret key
is generated into `instance/secret_key` (or set env var `SOU_IDS_SECRET_KEY`).

## 7. Example workflow — analyse a new CSV
1. *Traffic Detection* → drop a `.csv` → the app checks size/type, shows a preview and **feature validation**.
2. Pick rows (all / first N / random N / ticked preview rows) → **Analyze Traffic**.
3. See Normal vs Attack, predicted categories, confidence, record table; download CSV / PDF.
4. Check *Attack Analytics*, *Alerts & Logs* (change status New → Reviewed → Resolved), *Recommendations & Reports*.

If columns are missing you get a clear error listing them. **Retraining** (Admin → Datasets & Training) needs a
**labelled** file (a `Label` column). Predictions are never used as training labels, and unlabelled files can never
be used for retraining. A file with a different schema must be retrained on, which builds a matching pipeline.

## 8. Run the unit tests
```bat
python -m pytest -q tests
```

## 9. Troubleshooting
| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with “Add to PATH”, reopen Command Prompt. |
| `No module named flask` | Run `venv\Scripts\activate` then `pip install -r requirements.txt`. |
| Page unstyled / charts empty | No internet (CDN files not loading). Connect to the internet. |
| “No trained model available” | Admin must train (section 5). |
| `MemoryError` / training very slow | Lower `MAX_ROWS_PER_FILE`, `MAX_TRAIN_ROWS`, `RF_TREES` in `config.py`. |
| “File not found in datasets folder” | Copy CSVs into `datasets\` or upload on the Datasets page; fix slot names there. |
| “missing required feature columns” | The CSV isn’t CICIDS-style; use a compatible file or retrain with a labelled file of that schema. |
| Port 5000 busy | Change `port=5000` at the bottom of `app.py`. |
| Forgot demo password / reset everything | Stop the app, delete `instance\sou_ids.db` (and `models\*.joblib`), start again. |

## 10. Architecture (for the viva)
```
Browser (HTML/Bootstrap/Plotly)  ⇄  Flask (app.py: routes, auth, CSRF)
                                         │
      ┌──────────────┬───────────────────┼──────────────┬─────────────┐
 preprocessing.py  train_model.py    prediction.py   database.py   reporting.py
 (clean, pipeline) (3 models, select) (predict, alerts) (SQLite)     (PDF)
                          │                 │
                   models/*.joblib ←────────┘     evaluation.py (metrics)
```
**Data flow:** CSV → clean → (Imputer → Scaler → Classifier) pipeline → Normal/Attack + category → summary,
alerts, charts, reports. The pipeline is saved with the model so new files get **exactly** the same preprocessing.

## 11. Problems with existing systems (stated carefully)
Some traditional IDS products: rely heavily on predefined rules; signature-based detection may miss previously unseen
attacks; can produce high false-positive rates that overwhelm analysts; may face scalability challenges with large
traffic volumes; may adapt slowly to evolving threats; and enterprise-grade solutions can be expensive for small
companies. (Not every traditional IDS has all these limitations.)

## 12. Sample abstract
Small companies often cannot afford enterprise intrusion-detection solutions, and rule-based systems may miss new
attack patterns while generating many false alarms. This project presents **Sou-IDS**, a low-cost, browser-based
network intrusion detection system that uses machine learning. Using the CICIDS2017 dataset, network-flow records are
cleaned and processed with a leakage-safe scikit-learn pipeline. Decision Tree, Random Forest and Gaussian Naive Bayes
classifiers are trained and compared on validation data using security-relevant metrics (attack recall, F1-score and
false-positive rate); the selected model is evaluated once on a held-out test set. A Flask web application lets users
upload new CSV files for batch analysis, view interactive charts, manage alerts stored in SQLite, receive advisory
security recommendations, and download CSV/PDF reports. Models can be retrained on new labelled datasets and versioned.

## 13. Objectives
1. Build a ML-based classifier that labels network flows as Normal or Attack (and category where labelled data exists).
2. Compare Decision Tree, Random Forest and Gaussian Naive Bayes with honest validation/test evaluation.
3. Provide leakage-safe preprocessing and reliable handling of bad or incompatible files.
4. Offer a web dashboard with upload, analytics, alerts, model evaluation and reports.
5. Support retraining on new labelled data with model versioning.
6. Keep the solution affordable and runnable on an ordinary laptop.

## 14. Viva questions and simple answers
1. **What is an IDS?** Software that watches network data and raises alerts about suspicious activity.
2. **Why machine learning?** It learns patterns from examples, so it can recognise attacks that fixed rules don't describe — if they resemble training data.
3. **Why CICIDS2017?** A well-known public dataset of labelled flows (benign + modern attacks).
4. **Which models and why?** Decision Tree (simple, explainable), Random Forest (many trees, usually most accurate), Gaussian Naive Bayes (fast baseline).
5. **What is a data leak?** When information from the test data or the label influences training, making results look too good. We fit scalers/imputers on training data only, remove label-like and identifier columns, and test once.
6. **Why train/validation/test?** Validation chooses the model; the test set is a final, untouched exam.
7. **Why not just accuracy?** Attacks are a minority; a model could be 90% accurate and miss attacks. We use attack recall, F1 and false-positive rate.
8. **False positive vs false negative?** FP = normal flagged as attack (wasted analyst time). FN = missed attack (security risk).
9. **Macro vs weighted average?** Macro treats every class equally; weighted follows class sizes.
10. **What is a pipeline?** Imputer → Scaler → Model chained together and saved, so new data gets identical preprocessing.
11. **Is it real-time?** No. It analyses uploaded CSV batches; live packet capture is future work.
12. **What does confidence mean?** The model's probability estimate; it is not a guarantee of real risk.
13. **How is severity decided?** A simple rule (attack type + confidence), not a validated risk model.
14. **How do you retrain?** Admin selects labelled CSVs; the system trains all three models again, picks the best and saves a new version. Predictions are never used as labels.
15. **How is login secured?** Hashed passwords, sessions, CSRF tokens, role checks (demo only, not production grade).
16. **Limitations?** Only the attack types present in the training files; dataset age; CSV-based batch use; accuracy on real networks may differ.
17. **Future work?** Live capture (e.g. CICFlowMeter), more models, drift monitoring, email alerts.
