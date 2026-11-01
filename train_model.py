"""
train_model.py - trains Decision Tree, Random Forest and Gaussian Naive Bayes, compares them
on VALIDATION data, picks the best, and evaluates ONLY that model ONCE on the held-out TEST set.

Run from a terminal:
    python train_model.py                      (uses datasets/mapping.json or the default file names)
    python train_model.py --files a.csv b.csv  (any labelled CSV files inside the datasets folder)

The same function `train_all` is used by the web page (Datasets -> Train).
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import config
import database
from evaluation import evaluate_predictions, json_safe
from preprocessing import (NORMAL_LABEL, DataError, build_pipeline, clean_dataframe, detect_leaky_features,
                           find_label_column, infer_feature_columns, load_csv_efficient, map_labels,
                           split_meta_features, stratified_cap, to_binary)

TARGET = "__target__"


def model_factories():
    rs = config.RANDOM_STATE
    return {
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=25, min_samples_leaf=2,
                                                        class_weight="balanced", random_state=rs),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=config.RF_TREES, max_depth=25,
                                                        min_samples_leaf=2, class_weight="balanced_subsample",
                                                        n_jobs=-1, random_state=rs),
        "Gaussian Naive Bayes": lambda: GaussianNB(),
    }


def _safe_split(X, y, test_size, warnings, what):
    """Stratified split; falls back to a plain split (with a warning) if a class is too small."""
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE)
    except ValueError:
        warnings.append(f"Stratified {what} split was not possible (a class is too small); used a random split.")
        return train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE)


def resolve_dataset_files(names):
    """Turn file names (inside datasets/) into existing paths. Case-insensitive."""
    existing = {p.name.lower(): p for p in config.DATASETS_DIR.glob("*.csv")}
    paths, missing = [], []
    for n in names:
        p = existing.get(Path(n).name.lower())
        (paths if p else missing).append(p or n)
    return paths, missing


def default_file_names():
    if config.DATASET_MAPPING_FILE.exists():
        try:
            mapping = json.loads(config.DATASET_MAPPING_FILE.read_text())
            return [v for v in mapping.values() if v]
        except Exception:
            pass
    return list(config.DEFAULT_DATASET_FILES.values())


def load_labelled_file(path, progress, warnings):
    df = load_csv_efficient(path, max_rows=config.MAX_ROWS_PER_FILE, mode="sample", seed=config.RANDOM_STATE)
    label_col = find_label_column(df)
    if label_col is None:
        raise DataError(f"'{Path(path).name}' has no label column.",
                        "Training needs a labelled file. Add a column named 'Label' (e.g. BENIGN / DoS ...).")
    df[TARGET] = map_labels(df[label_col])
    classes = df[TARGET].value_counts().to_dict()
    progress(None, f"Loaded {Path(path).name}: {len(df):,} rows sampled, labels found: {classes}")
    return df


def train_all(file_paths, progress=None):
    """Full training workflow. Returns a dict with the summary; raises DataError on bad input."""
    def log(pct, msg):
        if progress:
            progress(pct, msg)

    warnings = []
    file_paths = [Path(p) for p in file_paths]
    if not file_paths:
        raise DataError("No dataset files were selected.", "Select at least one labelled CSV file.")

    # ---- 1. load -----------------------------------------------------------------
    log(3, "Loading CSV files (memory-efficient, chunked)...")
    frames = []
    for i, p in enumerate(file_paths):
        frames.append(load_labelled_file(p, lambda pc, m: log(None, m), warnings))
        log(3 + int(15 * (i + 1) / len(file_paths)), f"File {i + 1}/{len(file_paths)} done")
    df = pd.concat(frames, ignore_index=True)
    del frames

    # ---- 2. clean ----------------------------------------------------------------
    log(20, "Cleaning data: removing duplicates, infinite values, empty rows...")
    rows_before = len(df)
    df = clean_dataframe(df, label_col=TARGET, drop_duplicates=True)
    log(22, f"Rows: {rows_before:,} -> {len(df):,} after cleaning")

    counts = df[TARGET].value_counts()
    small = counts[counts < config.MIN_CLASS_SAMPLES].index.tolist()
    if small:
        warnings.append(f"Classes removed because they have fewer than {config.MIN_CLASS_SAMPLES} rows: {small}")
        df = df[~df[TARGET].isin(small)]
    if NORMAL_LABEL not in set(df[TARGET]):
        raise DataError("No Normal/BENIGN records found.", "Include a benign traffic file (e.g. Monday-WorkingHours).")
    if df[TARGET].nunique() < 2:
        raise DataError("Only one traffic class was found.", "Training needs Normal AND at least one attack class.")
    df = stratified_cap(df, TARGET, config.MAX_TRAIN_ROWS, seed=config.RANDOM_STATE)

    # ---- 3. features / target kept separate ---------------------------------------
    log(25, "Separating target label from input features; dropping identifier columns...")
    y = df[TARGET].astype(str).values
    X, _meta = split_meta_features(df, label_col=TARGET)
    del df
    numeric_cols, categorical_cols = infer_feature_columns(X)
    leaky = detect_leaky_features(X, to_binary(y), numeric_cols)
    if leaky:
        warnings.append(f"Possible label-leakage features removed: {leaky}")
        numeric_cols = [c for c in numeric_cols if c not in leaky]
    if not numeric_cols and not categorical_cols:
        raise DataError("No usable feature columns found.")
    X = X[numeric_cols + categorical_cols]
    class_counts = pd.Series(y).value_counts().to_dict()
    log(28, f"{len(X):,} rows, {len(numeric_cols)} numeric + {len(categorical_cols)} categorical features. "
            f"Classes: {class_counts}")

    # ---- 4. split: train / validation / test ----------------------------------------
    X_trainfull, X_test, y_trainfull, y_test = _safe_split(X, y, config.TEST_SIZE, warnings, "train/test")
    X_train, X_val, y_train, y_val = _safe_split(X_trainfull, y_trainfull, config.VALIDATION_SIZE, warnings,
                                                 "train/validation")
    log(30, f"Split -> train {len(X_train):,} | validation {len(X_val):,} | test (held out) {len(X_test):,}")

    # ---- 5. train the three models; compare on VALIDATION ---------------------------
    factories = model_factories()
    val_results, train_seconds = {}, {}
    for i, (name, make) in enumerate(factories.items()):
        log(32 + i * 16, f"Training {name} ...")
        t0 = time.time()
        try:
            pipe = build_pipeline(make(), numeric_cols, categorical_cols)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            res = evaluate_predictions(y_val, pred, pipe.classes_)
            train_seconds[name] = round(time.time() - t0, 1)
            val_results[name] = res
            log(32 + i * 16 + 12, f"{name}: validation accuracy {res['accuracy']:.4f}, attack recall "
                                  f"{res['binary_recall']:.4f}, attack F1 {res['binary_f1']:.4f}, "
                                  f"false-positive rate {res['false_positive_rate']:.4f} ({train_seconds[name]}s)")
        except Exception as e:  # keep going with the other models
            warnings.append(f"{name} failed to train: {e}")
            log(None, f"{name} FAILED: {e}")
    if not val_results:
        raise DataError("All models failed to train.", "; ".join(warnings[-3:]))

    # ---- 6. choose best model using validation only -----------------------------------
    best_name = max(val_results, key=lambda n: val_results[n]["selection_score"])
    log(82, f"Best model on validation data: {best_name} "
            f"(selection score {val_results[best_name]['selection_score']:.4f})")

    # ---- 7. refit best on train+validation and evaluate ONCE on the test set ----------
    log(84, f"Refitting {best_name} on training + validation data...")
    best_pipe = build_pipeline(factories[best_name](), numeric_cols, categorical_cols)
    best_pipe.fit(X_trainfull, y_trainfull)
    log(92, "Evaluating the chosen model once on the held-out test set...")
    test_result = evaluate_predictions(y_test, best_pipe.predict(X_test), best_pipe.classes_)

    importance = None
    model = best_pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = [n.split("__", 1)[-1] for n in best_pipe.named_steps["preprocess"].get_feature_names_out()]
        order = np.argsort(model.feature_importances_)[::-1][:15]
        importance = [{"feature": names[i], "importance": float(model.feature_importances_[i])} for i in order]

    # ---- 8. save model + register version ------------------------------------------
    version = f"v{database.next_version_number()}"
    path = config.MODELS_DIR / f"sou_ids_{version}.joblib"
    trained_at = database.now()
    bundle = {"pipeline": best_pipe, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
              "classes": [str(c) for c in best_pipe.classes_], "model_name": best_name, "version": version,
              "trained_at": trained_at}
    joblib.dump(bundle, path, compress=3)

    metrics = json_safe({
        "validation": val_results, "test": test_result, "best_model": best_name,
        "selection_rule": "0.40*attack recall + 0.40*attack F1 + 0.20*(1 - false positive rate), on VALIDATION data",
        "rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test),
                 "train_plus_validation": len(X_trainfull)},
        "n_features": len(numeric_cols) + len(categorical_cols),
        "class_counts": class_counts, "train_seconds": train_seconds,
        "feature_importance": importance, "warnings": warnings,
    })
    version = database.register_model(best_name, path, [p.name for p in file_paths], int(len(X)),
                                      bundle["classes"], metrics)
    log(100, f"Training finished. Saved model {version} ({best_name}).")
    return {"version": version, "best_model": best_name, "metrics": metrics, "warnings": warnings}


def main():
    ap = argparse.ArgumentParser(description="Train Sou-IDS models")
    ap.add_argument("--files", nargs="*", help="CSV file names located in the datasets folder")
    args = ap.parse_args()
    database.init_db()
    names = args.files or default_file_names()
    paths, missing = resolve_dataset_files(names)
    for m in missing:
        print(f"[warning] file not found in datasets/: {m}")
    try:
        result = train_all(paths, progress=lambda pc, msg: print(f"[{'' if pc is None else str(pc) + '%'}] {msg}"))
    except DataError as e:
        print(f"\nERROR: {e.message}\n{e.hint or ''}")
        raise SystemExit(1)
    t = result["metrics"]["test"]
    print(f"\nDone. Model {result['version']} = {result['best_model']}")
    print(f"TEST accuracy {t['accuracy']:.4f} | attack recall {t['binary_recall']:.4f} | "
          f"attack F1 {t['binary_f1']:.4f} | false-positive rate {t['false_positive_rate']:.4f}")


if __name__ == "__main__":
    main()
