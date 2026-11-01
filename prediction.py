"""
prediction.py - load the saved model, validate new CSVs, predict, summarise, create alerts
and produce advisory security recommendations.
"""
import warnings as _warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
import database
from evaluation import evaluate_predictions, json_safe
from preprocessing import (NORMAL_LABEL, DataError, clean_columns, coerce_numeric_columns, find_label_column,
                           map_labels, missing_features, prepare_features_for_prediction, split_meta_features)

_cache = {}
LEVELS = ["Low", "Medium", "High", "Critical"]
BASE_SEVERITY = {"Heartbleed": 3, "Infiltration": 3, "DDoS": 2, "DoS": 2, "Brute Force": 2, "Bot": 2,
                 "Web Attack": 2, "PortScan": 1}
PROTOCOL_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}
PREFERRED_HIST = ["flow_duration", "total_fwd_packets", "total_length_of_fwd_packets", "flow_bytes_s",
                  "flow_packets_s", "packet_length_mean", "average_packet_size", "destination_port",
                  "fwd_packet_length_mean", "flow_iat_mean"]


# --------------------------------------------------------------------------- #
def load_bundle():
    """Return (bundle, model_info) of the ACTIVE model, or (None, None) if nothing is trained."""
    info = database.get_active_model()
    if not info:
        return None, None
    path = Path(info["model_path"])
    if not path.exists():
        raise DataError("The active model file is missing from the models folder.",
                        "Retrain the models from the Datasets page.")
    key = (str(path), path.stat().st_mtime)
    if key not in _cache:
        _cache.clear()
        _cache[key] = joblib.load(path)
    return _cache[key], info


def validate_columns(columns, bundle):
    """Check an uploaded file's (cleaned) columns against the model's required features."""
    required = list(bundle["numeric_cols"]) + list(bundle["categorical_cols"])
    missing = missing_features(columns, required)
    return {"compatible": not missing, "required": len(required), "missing": missing,
            "matched": len(required) - len(missing)}


def severity_for(category, confidence):
    """Rule-based prioritisation (NOT a validated risk model)."""
    level = BASE_SEVERITY.get(category, 1)
    if confidence is not None and confidence < 0.6:
        level = max(0, level - 1)
    return LEVELS[level]


def predict_frame(bundle, X, batch=50_000):
    pipe = bundle["pipeline"]
    preds, confs = [], []
    has_proba = hasattr(pipe, "predict_proba")
    for i in range(0, len(X), batch):
        part = X.iloc[i:i + batch]
        preds.append(pipe.predict(part))
        if has_proba:
            confs.append(pipe.predict_proba(part).max(axis=1))
    return np.concatenate(preds), (np.concatenate(confs) if has_proba else None)


# --------------------------------------------------------------------------- #
def analyze_dataframe(df, bundle):
    """
    df: raw uploaded data (column names already cleaned).
    Returns (results_df, summary_dict, alerts_list).
    """
    label_col = find_label_column(df)
    truth = map_labels(df[label_col]).values if label_col else None
    X_all, meta = split_meta_features(df, label_col)
    X = prepare_features_for_prediction(X_all, bundle["numeric_cols"], bundle["categorical_cols"])
    if len(X) == 0:
        raise DataError("The file has no data rows to analyse.")

    pred, conf = predict_frame(bundle, X)
    pred = np.asarray(pred, dtype=object)
    is_attack = pred != NORMAL_LABEL

    res = pd.DataFrame({"row": df.index.values + 1})
    for c in ("timestamp", "src_ip", "dst_ip"):
        if c in meta.columns:
            res[c] = meta[c].values
    if "protocol" in X_all.columns:
        res["protocol"] = X_all["protocol"].values
    if truth is not None:
        res["ground_truth_label"] = truth
    res["prediction"] = np.where(is_attack, "Attack", "Normal")
    res["predicted_category"] = pred
    res["confidence"] = np.round(conf, 4) if conf is not None else np.nan

    summary = build_summary(res, X, is_attack, truth, bundle)
    alerts = build_alerts(res)
    return res, summary, alerts


def build_alerts(res):
    atk = res[res["prediction"] == "Attack"]
    if atk.empty:
        return []
    if atk["confidence"].notna().any():
        atk = atk.sort_values("confidence", ascending=False)
    atk = atk.head(config.MAX_ALERTS_PER_ANALYSIS)
    alerts = []
    for _, r in atk.iterrows():
        c = None if pd.isna(r["confidence"]) else float(r["confidence"])
        alerts.append({
            "row_number": int(r["row"]),
            "event_time": None if "timestamp" not in r or pd.isna(r["timestamp"]) else str(r["timestamp"]),
            "src_ip": None if "src_ip" not in r or pd.isna(r["src_ip"]) else str(r["src_ip"]),
            "dst_ip": None if "dst_ip" not in r or pd.isna(r["dst_ip"]) else str(r["dst_ip"]),
            "prediction": "Attack", "category": str(r["predicted_category"]),
            "confidence": c, "severity": severity_for(r["predicted_category"], c)})
    return alerts


def build_summary(res, X, is_attack, truth, bundle):
    total = int(len(res))
    attack = int(is_attack.sum())
    s = {
        "total": total, "normal": total - attack, "attack": attack,
        "attack_pct": round(100.0 * attack / total, 2) if total else 0.0,
        "categories": res["predicted_category"].value_counts().to_dict(),
        "supported_classes": bundle["classes"],
        "mean_attack_confidence": (float(res.loc[res["prediction"] == "Attack", "confidence"].mean())
                                   if attack and res["confidence"].notna().any() else None),
        "low_confidence_attacks": int(((res["prediction"] == "Attack") & (res["confidence"] < 0.6)).sum())
        if res["confidence"].notna().any() else 0,
        "has_ground_truth": truth is not None,
    }

    # ---- ground truth (only if the uploaded file has a label column) ----
    if truth is not None:
        t = pd.Series(truth)
        s["truth_categories"] = t.value_counts().to_dict()
        tb = int((t != NORMAL_LABEL).sum())
        s["truth_binary"] = {"Normal": int(len(t) - tb), "Attack": tb}
        unsupported = sorted(set(t.dropna()) - set(bundle["classes"]) - {NORMAL_LABEL})
        s["unsupported_truth_categories"] = unsupported
        mask = t.notna().values
        ev = evaluate_predictions(t[mask].values, res["predicted_category"].values[mask],
                                  sorted(set(t[mask]) | set(res["predicted_category"][mask])))
        s["labelled_eval"] = {k: ev[k] for k in ("accuracy", "binary_precision", "binary_recall", "binary_f1",
                                                 "tn", "fp", "fn", "tp", "false_positive_rate",
                                                 "false_negative_rate")}

    # ---- time trends ----
    if "timestamp" in res.columns:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            ts = pd.to_datetime(res["timestamp"], errors="coerce")
        if ts.notna().mean() >= 0.5:
            d = pd.DataFrame({"ts": ts, "attack": is_attack}).dropna(subset=["ts"])

            def series(key):
                g = d.groupby(key)["attack"].agg(["count", "sum"])
                return {"labels": [str(i) for i in g.index], "attack": g["sum"].astype(int).tolist(),
                        "normal": (g["count"] - g["sum"]).astype(int).tolist()}
            s["daily"] = series(d["ts"].dt.date)
            hourly = series(d["ts"].dt.floor("h"))
            if len(hourly["labels"]) <= 400:
                s["hourly"] = hourly

    # ---- protocol-wise ----
    if "protocol" in res.columns:
        p = res["protocol"].map(lambda v: PROTOCOL_NAMES.get(int(v), f"Proto {int(v)}") if pd.notna(v) else "Unknown")
        g = pd.DataFrame({"p": p, "attack": is_attack}).groupby("p")["attack"].agg(["count", "sum"])
        s["protocols"] = {"labels": g.index.tolist(), "attack": g["sum"].astype(int).tolist(),
                          "normal": (g["count"] - g["sum"]).astype(int).tolist()}

    # ---- top suspicious sources ----
    if "src_ip" in res.columns and attack:
        top = res[res["prediction"] == "Attack"]["src_ip"].value_counts().head(10)
        s["top_sources"] = [{"ip": str(k), "attacks": int(v)} for k, v in top.items()]

    # ---- numeric feature analytics (on a sample) ----
    num_cols = [c for c in bundle["numeric_cols"] if c in X.columns]
    if num_cols:
        n = min(len(X), 50_000)
        idx = np.random.RandomState(0).choice(len(X), n, replace=False) if len(X) > n else np.arange(len(X))
        Xs = X.iloc[idx][num_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
        atk_s = np.asarray(is_attack)[idx]
        std = Xs.std().replace(0, np.nan)
        usable = [c for c in num_cols if pd.notna(std.get(c))]

        # distributions
        chosen = [c for c in PREFERRED_HIST if c in usable][:6]
        for c in sorted(usable, key=lambda c: -float(std[c]))[:6]:
            if len(chosen) >= 6:
                break
            if c not in chosen:
                chosen.append(c)
        hist = {}
        for c in chosen:
            v = Xs[c]
            lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
            if not hi > lo:
                hi = lo + 1
            edges = np.linspace(lo, hi, 21)
            clipped = v.clip(lo, hi)
            hn, _ = np.histogram(clipped[~atk_s].dropna(), bins=edges)
            ha, _ = np.histogram(clipped[atk_s].dropna(), bins=edges)
            hist[c] = {"edges": [float(e) for e in edges], "normal": hn.tolist(), "attack": ha.tolist()}
        s["feature_hist"] = hist

        # normal vs attack comparison (standardised means so features are comparable)
        z = (Xs[usable] - Xs[usable].mean()) / std[usable]
        ranking = list(usable)
        if atk_s.any() and (~atk_s).any():
            gm = z.groupby(atk_s).mean()
            diff = (gm.loc[True] - gm.loc[False]).abs().fillna(0).sort_values(ascending=False)
            ranking = diff.index.tolist()
            top = ranking[:8]
            s["feature_compare"] = {"features": top, "normal": gm.loc[False, top].round(4).tolist(),
                                    "attack": gm.loc[True, top].round(4).tolist()}
        # correlation heatmap
        cols = ranking[:12]
        if len(cols) >= 2:
            corr = Xs[cols].corr().fillna(0)
            s["correlation"] = {"features": cols, "matrix": corr.round(3).values.tolist()}

    return json_safe(s)


# --------------------------------------------------------------------------- #
def recommendations_for(summary):
    """Advisory, rule-based recommendations. The system never blocks or changes anything itself."""
    recs = []
    cats = {k: v for k, v in (summary.get("categories") or {}).items() if k != NORMAL_LABEL}
    if not summary.get("attack"):
        recs.append({"priority": "Info", "title": "No attacks predicted in this file",
                     "detail": "The model predicted every record as Normal. This does not prove the traffic is safe - "
                               "keep monitoring and keep the model updated with new labelled data."})
        return recs
    recs.append({"priority": "High", "title": "Review firewall and gateway logs",
                 "detail": "Check firewall/router logs for the time window of the flagged records and confirm whether "
                           "the connections are expected."})
    top = summary.get("top_sources") or []
    if top:
        ips = ", ".join(t["ip"] for t in top[:3])
        recs.append({"priority": "High", "title": "Verify suspicious source IP addresses",
                     "detail": f"Most flagged records came from: {ips}. Check whether these belong to your own "
                               f"devices or partners. Blocking is a manual decision - Sou-IDS never changes your network."})
        if top[0]["attacks"] >= 10:
            recs.append({"priority": "High", "title": "Investigate repeated alerts",
                         "detail": f"{top[0]['ip']} triggered {top[0]['attacks']} attack predictions. Repeated alerts "
                                   f"from one source deserve a closer look."})
    advice = {
        "DoS": ("Possible Denial-of-Service activity", "Enable rate limiting, check server CPU/bandwidth, review web "
                "server limits and ask your ISP about mitigation if the service is affected."),
        "DDoS": ("Possible Distributed Denial-of-Service activity", "Use rate limiting / a CDN or DDoS-protection "
                 "service, and coordinate with your ISP. Check which services are being targeted."),
        "PortScan": ("Possible port scanning (reconnaissance)", "Close unused ports, restrict exposed services with "
                     "firewall rules and watch for follow-up exploit attempts from the same source."),
        "Brute Force": ("Possible brute-force login attempts", "Enforce strong passwords and multi-factor "
                        "authentication, enable account lockout, and review SSH/FTP authentication logs."),
        "Bot": ("Possible botnet activity", "Isolate suspected hosts, run an endpoint malware scan and review outbound "
                "connections / DNS requests for unknown servers."),
        "Web Attack": ("Possible web application attack", "Review web server logs, patch web applications and consider "
                       "a web application firewall (WAF)."),
        "Heartbleed": ("Possible Heartbleed exploitation", "Patch OpenSSL, then re-issue certificates and rotate keys "
                       "and passwords that may have been exposed."),
        "Infiltration": ("Possible infiltration", "Isolate the affected machine, preserve logs for investigation and "
                         "check for unusual internal connections."),
    }
    for cat, n in sorted(cats.items(), key=lambda kv: -kv[1]):
        title, detail = advice.get(cat, (f"Predicted category: {cat}", "Investigate the flagged records manually "
                                         "and compare them with your normal traffic patterns."))
        recs.append({"priority": "High" if cat in advice else "Medium", "title": f"{title} ({n:,} records)",
                     "detail": detail})
    if summary.get("low_confidence_attacks"):
        recs.append({"priority": "Medium", "title": "Manually review low-confidence predictions",
                     "detail": f"{summary['low_confidence_attacks']:,} attack predictions have model confidence "
                               f"below 60%. Model confidence is not a guarantee of real-world risk."})
    if summary.get("has_ground_truth") and summary.get("unsupported_truth_categories"):
        recs.append({"priority": "Medium", "title": "Some labelled categories are not supported by the model",
                     "detail": "The file contains categories the model was not trained on: "
                               f"{', '.join(summary['unsupported_truth_categories'])}. Retrain with labelled data "
                               "that includes them."})
    recs.append({"priority": "Info", "title": "Keep the model up to date",
                 "detail": "Retrain periodically with new, verified labelled traffic so the model adapts to evolving threats."})
    return recs
