"""
preprocessing.py - reusable data cleaning / preprocessing functions for Sou-IDS.

Key ideas
---------
* Column names are normalised (e.g. " Flow Bytes/s" -> "flow_bytes_s").
* Labels are mapped to friendly categories (BENIGN -> Normal, "DoS Hulk" -> DoS ...).
* Identifier columns (IPs, ports of the source, flow id, timestamp) are NOT used as features.
* Imputers / encoders / scalers live inside a scikit-learn Pipeline, so they are
  fitted ONLY on training data (no test-set leakage).
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NORMAL_LABEL = "Normal"

LABEL_CANDIDATES = ["label", "class", "attack_cat", "attack", "target", "traffic_label"]
LEAKY_NAMES = set(LABEL_CANDIDATES) | {"label_binary", "is_attack", "attack_type", "category"}

# Columns used only as metadata for alerts (never as model inputs)
META_ALIASES = {
    "source_ip": "src_ip", "src_ip": "src_ip",
    "destination_ip": "dst_ip", "dst_ip": "dst_ip",
    "timestamp": "timestamp", "flow_id": "flow_id",
}
# Columns dropped because they only identify a flow (not describe behaviour)
IDENTIFIER_COLS = {
    "flow_id", "src_ip", "dst_ip", "source_ip", "destination_ip", "timestamp",
    "source_port", "src_port", "unnamed_0", "unnamed_0_1", "id", "index", "row",
}

EXTRA_NA_VALUES = ["Infinity", "-Infinity", "inf", "-inf", "+inf", "Inf", "NaN", "nan", "NULL", "null", ""]


class DataError(Exception):
    """A user-facing data problem (bad CSV, missing features, no label...)."""

    def __init__(self, message, hint=None):
        super().__init__(message)
        self.message = message
        self.hint = hint


# --------------------------------------------------------------------------- #
# 1. Column names
# --------------------------------------------------------------------------- #
def normalize_column_name(name):
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unnamed"


def clean_columns(df):
    """Strip whitespace / normalise names and make duplicate names unique."""
    new, seen = [], {}
    for c in df.columns:
        n = normalize_column_name(c)
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        new.append(n)
    out = df.copy(deep=False)
    out.columns = new
    return out


def find_label_column(df):
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


# --------------------------------------------------------------------------- #
# 2. Labels
# --------------------------------------------------------------------------- #
def map_label(value):
    """Map a raw CICIDS-style label to a friendly category."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    low = s.lower()
    if low in ("", "nan", "none"):
        return None
    if low in ("benign", "normal", "0", "legitimate"):
        return NORMAL_LABEL
    if "heartbleed" in low:
        return "Heartbleed"
    if "infiltration" in low or "infilteration" in low:
        return "Infiltration"
    if "web attack" in low or "sql injection" in low or low == "xss":
        return "Web Attack"
    if "ddos" in low:
        return "DDoS"
    if "portscan" in low or "port scan" in low:
        return "PortScan"
    if "patator" in low or "brute force" in low or "bruteforce" in low:
        return "Brute Force"
    if "botnet" in low or low == "bot" or low.startswith("bot "):
        return "Bot"
    if "dos" in low:
        return "DoS"
    return s  # unknown category - keep the original name (still treated as an attack)


def map_labels(series):
    return series.map(map_label)


def to_binary(labels):
    """1 = Attack, 0 = Normal."""
    return (np.asarray(labels) != NORMAL_LABEL).astype(int)


# --------------------------------------------------------------------------- #
# 3. Loading big CSV files with low memory
# --------------------------------------------------------------------------- #
def count_data_rows(path):
    """Fast line count (minus header)."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


def _downcast(chunk):
    for c in chunk.columns:
        if pd.api.types.is_float_dtype(chunk[c]) or pd.api.types.is_integer_dtype(chunk[c]):
            chunk[c] = chunk[c].astype("float32")
    return chunk


def read_csv_safe(path, **kwargs):
    """pd.read_csv with friendly errors."""
    try:
        return pd.read_csv(path, na_values=EXTRA_NA_VALUES, keep_default_na=True,
                           encoding_errors="replace", **kwargs)
    except pd.errors.EmptyDataError:
        raise DataError("The CSV file is empty.", "Upload a CSV file that contains a header row and data rows.")
    except (pd.errors.ParserError, ValueError) as e:
        raise DataError(f"The file could not be parsed as a CSV ({str(e)[:150]}).",
                        "Open the file in a text editor, check that it is comma-separated and not damaged.")


def load_csv_efficient(path, max_rows=None, mode="head", chunksize=100_000, seed=42):
    """
    Load a CSV in chunks, normalise column names and down-cast numbers to float32.

    mode="head"   -> read the first `max_rows` rows (used for analysis).
    mode="sample" -> read the whole file in chunks but keep a random fraction so about
                     `max_rows` rows remain (used for training big files).
    """
    if mode == "head":
        df = read_csv_safe(path, nrows=max_rows, low_memory=False)
        return _downcast(clean_columns(df))

    total = count_data_rows(path)
    frac = 1.0 if (not max_rows or total <= max_rows) else max_rows / total
    parts = []
    reader = read_csv_safe(path, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader):
        chunk = _downcast(clean_columns(chunk))
        if frac < 1.0:
            chunk = chunk.sample(frac=frac, random_state=seed + i)
        parts.append(chunk)
    if not parts:
        raise DataError("The CSV file contains no data rows.")
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# 4. Cleaning
# --------------------------------------------------------------------------- #
def clean_dataframe(df, label_col=None, drop_duplicates=True):
    """
    * replace +/-inf by NaN
    * drop completely empty columns / rows
    * drop rows without a label (when a label column is given)
    * remove duplicate rows
    Missing feature values are NOT filled here - the pipeline imputer does it
    (fitted on training data only).
    """
    out = df.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all")
    out = out.dropna(axis=0, how="all")
    if label_col:
        out = out[out[label_col].notna()]
    if drop_duplicates:
        out = out.drop_duplicates()
    return out.reset_index(drop=True)


def coerce_numeric_columns(df, exclude=()):
    """Turn text columns that really hold numbers (e.g. '12.5') into float32."""
    out = df.copy()
    for c in out.columns:
        if c in exclude or out[c].dtype != object and not pd.api.types.is_string_dtype(out[c]):
            continue
        conv = pd.to_numeric(out[c], errors="coerce")
        nonnull = out[c].notna().sum()
        if nonnull and conv.notna().sum() / nonnull >= 0.9:
            out[c] = conv.replace([np.inf, -np.inf], np.nan).astype("float32")
    return out


def split_meta_features(df, label_col=None):
    """
    Separate (features, metadata).  Metadata = IPs / timestamp used only for alerts.
    The label column and any label-like column are removed from the features.
    """
    meta = pd.DataFrame(index=df.index)
    for col, std in META_ALIASES.items():
        if col in df.columns and std not in meta.columns:
            meta[std] = df[col]
    drop = {c for c in df.columns if c in IDENTIFIER_COLS or c in LEAKY_NAMES or c == label_col}
    features = df.drop(columns=list(drop & set(df.columns)))
    features = coerce_numeric_columns(features)
    return features, meta


def infer_feature_columns(X, max_categories=30):
    """Return (numeric_cols, categorical_cols).  High-cardinality text columns are dropped."""
    numeric, categorical = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            if X[c].notna().any():
                numeric.append(c)
        elif X[c].nunique(dropna=True) <= max_categories:
            categorical.append(c)
    return numeric, categorical


def detect_leaky_features(X, y_binary, numeric_cols, threshold=0.999, sample=50_000):
    """A feature almost perfectly correlated with the label is probably a leaked label."""
    leaky = []
    if len(numeric_cols) == 0:
        return leaky
    idx = np.arange(len(X))
    if len(idx) > sample:
        idx = np.random.RandomState(0).choice(idx, sample, replace=False)
    ys = pd.Series(np.asarray(y_binary)[idx])
    if ys.nunique() < 2:
        return leaky
    for c in numeric_cols:
        col = X[c].iloc[idx].reset_index(drop=True).replace([np.inf, -np.inf], np.nan).fillna(0)
        if col.nunique() < 2:
            continue
        r = np.corrcoef(col.values.astype(float), ys.values)[0, 1]
        if np.isfinite(r) and abs(r) >= threshold:
            leaky.append(c)
    return leaky


def stratified_cap(df, label_col, cap, min_keep=500, seed=42):
    """Reduce df to about `cap` rows while keeping every class represented."""
    if len(df) <= cap:
        return df
    frac = cap / len(df)
    parts = []
    for _, g in df.groupby(label_col, sort=False):
        n = len(g)
        keep = min(n, max(min_keep, int(round(n * frac))))
        parts.append(g.sample(n=keep, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Pipelines (fit on training data only!)
# --------------------------------------------------------------------------- #
def build_pipeline(model, numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(numeric_cols)))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical_cols)))
    if not transformers:
        raise DataError("No usable feature columns were found in the dataset.")
    pre = ColumnTransformer(transformers, sparse_threshold=0.0)
    return Pipeline([("preprocess", pre), ("model", model)])


# --------------------------------------------------------------------------- #
# 6. Schema validation for new data
# --------------------------------------------------------------------------- #
def missing_features(columns, required):
    cols = set(columns)
    return [c for c in required if c not in cols]


def prepare_features_for_prediction(X, numeric_cols, categorical_cols):
    """Check that all required columns exist, convert types, keep column order."""
    required = list(numeric_cols) + list(categorical_cols)
    missing = missing_features(X.columns, required)
    if missing:
        shown = ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")
        raise DataError(
            f"The file is missing {len(missing)} required feature column(s): {shown}",
            "Use a CSV with the same columns as the training data (CICIDS2017 style flow features), "
            "or retrain the models with a labelled dataset that has your schema (Datasets page).",
        )
    out = X[required].copy()
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out
