"""
app.py - Flask web application for Sou-IDS.
Run with:   python app.py      then open http://127.0.0.1:5000
"""
import functools
import hmac
import json
import logging
import secrets
import threading
import time
import webbrowser
import uuid
from pathlib import Path

import pandas as pd
from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request, send_file, session,
                   url_for)
from werkzeug.utils import secure_filename

import config
import database
import prediction
import train_model
from evaluation import json_safe
from preprocessing import (DataError, clean_columns, count_data_rows, find_label_column, load_csv_efficient,
                           map_labels, read_csv_safe)
from reporting import build_pdf

app = Flask(__name__)
app.config.update(SECRET_KEY=config.SECRET_KEY, MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sou-ids")
database.init_db()

TRAIN = {"running": False, "pct": 0, "messages": [], "result": None, "error": None, "hint": None}
TRAIN_LOCK = threading.Lock()
LOGIN_FAILS = {}  # ip -> [timestamps]  (very small brute-force limiter for the demo)


# ----------------------------------------------------------------------------- #
# Security helpers: CSRF, login, roles
# ----------------------------------------------------------------------------- #
def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)
    return session["csrf_token"]


@app.before_request
def csrf_protect():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
        if not hmac.compare_digest(str(sent), str(session.get("csrf_token", ""))):
            if request.path.startswith("/api/"):
                return jsonify(error="Security token missing or expired. Refresh the page and try again."), 400
            abort(400, "Invalid CSRF token. Refresh the page and try again.")


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    return resp


@app.before_request
def load_user():
    g.user = database.get_user(session["user_id"]) if "user_id" in session else None


def login_required(f):
    @functools.wraps(f)
    def wrapper(*a, **kw):
        if not g.user:
            if request.path.startswith("/api/"):
                return jsonify(error="Please sign in first."), 401
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


def admin_required(f):
    @functools.wraps(f)
    @login_required
    def wrapper(*a, **kw):
        if g.user["role"] != "Admin":
            if request.path.startswith("/api/"):
                return jsonify(error="This action needs the Admin role."), 403
            flash("That page needs the Admin role.", "danger")
            return redirect(url_for("dashboard"))
        return f(*a, **kw)
    return wrapper


def api(f):
    """Convert exceptions into clean JSON errors."""
    @functools.wraps(f)
    def wrapper(*a, **kw):
        try:
            return f(*a, **kw)
        except DataError as e:
            return jsonify(error=e.message, hint=e.hint), 400
        except Exception as e:  # noqa
            log.exception("API error")
            return jsonify(error=f"Unexpected error: {e}"), 500
    return wrapper


@app.context_processor
def inject_globals():
    info = None
    try:
        info = database.get_active_model()
    except Exception:
        pass
    return dict(csrf_token=csrf_token, app_name=config.APP_NAME, company=config.COMPANY_NAME,
                user=g.get("user"), active_model=info)


# ----------------------------------------------------------------------------- #
# Auth pages
# ----------------------------------------------------------------------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        ip = request.remote_addr
        recent = [t for t in LOGIN_FAILS.get(ip, []) if time.time() - t < 300]
        LOGIN_FAILS[ip] = recent
        if len(recent) >= 5:
            flash("Too many failed attempts. Please wait 5 minutes.", "danger")
        else:
            u = database.authenticate(request.form.get("login", ""), request.form.get("password", ""))
            if u:
                session.clear()
                session["user_id"] = u["id"]
                csrf_token()
                return redirect(url_for("dashboard"))
            LOGIN_FAILS[ip].append(time.time())
            flash("Invalid username/email or password.", "danger")
    return render_template("login.html", demo=config.DEMO_USERS)


@app.post("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


# ----------------------------------------------------------------------------- #
# Pages
# ----------------------------------------------------------------------------- #
@app.route("/")
def index():
    return redirect(url_for("dashboard"))


def page(name):
    return render_template(name, analyses=database.list_analyses(30))


@app.route("/dashboard")
@login_required
def dashboard():
    return page("dashboard.html")


@app.route("/detect")
@login_required
def detect():
    return page("detect.html")


@app.route("/analytics")
@login_required
def analytics():
    return page("analytics.html")


@app.route("/alerts")
@login_required
def alerts():
    return page("alerts.html")


@app.route("/model-evaluation")
@login_required
def model_evaluation():
    return page("model_evaluation.html")


@app.route("/recommendations")
@login_required
def recommendations():
    return page("recommendations.html")


@app.route("/datasets")
@admin_required
def datasets():
    return page("datasets.html")


# ----------------------------------------------------------------------------- #
# API - dashboard / model / analysis
# ----------------------------------------------------------------------------- #
@app.get("/api/dashboard")
@login_required
@api
def api_dashboard():
    info = database.get_active_model()
    model = None
    if info:
        t = info["metrics"].get("test", {})
        model = {"name": info["model_name"], "version": info["version"], "trained_at": info["trained_at"],
                 "classes": info["classes"], "test_accuracy": t.get("accuracy"),
                 "test_attack_recall": t.get("binary_recall"), "dataset_files": info["dataset_files"],
                 "n_rows": info["n_rows"]}
    trend = list(reversed(database.list_analyses(12)))
    cat_totals = {}
    for a in database.list_analyses(50):
        full = database.get_analysis(a["id"])
        for k, v in (full["summary"].get("categories") or {}).items():
            cat_totals[k] = cat_totals.get(k, 0) + v
    files = [{"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
             for p in sorted(config.DATASETS_DIR.glob("*.csv"))]
    return jsonify(json_safe({"totals": database.dashboard_totals(), "model": model, "trend": trend,
                              "categories": cat_totals, "recent_alerts": database.recent_alerts(8),
                              "alert_status": database.alert_status_counts(), "dataset_files": files}))


@app.get("/api/model")
@login_required
@api
def api_model():
    info = database.get_active_model()
    return jsonify(json_safe({"model": info, "history": database.list_models()}))


@app.get("/api/analysis/")
@app.get("/api/analysis/<int:analysis_id>")
@login_required
@api
def api_analysis(analysis_id=None):
    a = database.get_analysis(analysis_id)
    if not a:
        return jsonify(error="No analysis found yet. Analyse a CSV file first."), 404
    a["recommendations"] = prediction.recommendations_for(a["summary"])
    a.pop("results_path", None)
    return jsonify(json_safe(a))


# ----------------------------------------------------------------------------- #
# API - traffic detection
# ----------------------------------------------------------------------------- #
def _check_csv_upload(file, max_mb):
    if not file or not file.filename:
        raise DataError("No file selected.", "Choose a .csv file first.")
    name = secure_filename(file.filename)
    if Path(name).suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise DataError("Only .csv files are allowed.", "Export your data as CSV (comma separated).")
    if request.content_length and request.content_length > max_mb * 1024 * 1024:
        raise DataError(f"File is larger than the {max_mb} MB limit.",
                        "Split the file into smaller parts or analyse a sample of it.")
    return name


@app.post("/api/detect/preview")
@login_required
@api
def api_preview():
    bundle, info = prediction.load_bundle()
    name = _check_csv_upload(request.files.get("file"), config.MAX_DETECT_UPLOAD_MB)
    upload_id = f"{uuid.uuid4().hex[:12]}_{name}"
    path = config.UPLOADS_DIR / upload_id
    request.files["file"].save(path)
    try:
        head = clean_columns(read_csv_safe(path, nrows=2000, low_memory=False))
        if head.shape[1] < 2:
            raise DataError("The file has fewer than 2 columns - it does not look like a comma-separated CSV.",
                            "Check the delimiter (commas) and that the first row contains column names.")
        rows = count_data_rows(path)
        validation = (prediction.validate_columns(head.columns, bundle) if bundle else None)
        label_col = find_label_column(head)
        raw = read_csv_safe(path, nrows=10, low_memory=False)
        raw.columns = [str(c).strip() for c in raw.columns]
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return jsonify(json_safe({
        "upload_id": upload_id, "filename": name, "rows": rows, "columns": int(head.shape[1]),
        "has_label": label_col is not None, "label_column": label_col, "validation": validation,
        "model_available": bundle is not None,
        "preview_columns": raw.columns.tolist()[:14],
        "preview_rows": raw.iloc[:, :14].astype(object).where(raw.iloc[:, :14].notna(), "").values.tolist(),
        "max_rows": config.MAX_ANALYSIS_ROWS}))


@app.post("/api/detect/analyze")
@login_required
@api
def api_analyze():
    data = request.get_json(silent=True) or {}
    upload_id = secure_filename(str(data.get("upload_id", "")))
    path = config.UPLOADS_DIR / upload_id
    if not upload_id or not path.exists():
        raise DataError("The uploaded file was not found.", "Upload the CSV again, then click Analyze Traffic.")
    bundle, info = prediction.load_bundle()
    if bundle is None:
        raise DataError("No trained model is available.", "Ask an Admin to train the models (Datasets page).")
    try:
        df = load_csv_efficient(path, max_rows=config.MAX_ANALYSIS_ROWS, mode="head")
        truncated = count_data_rows(path) > len(df)
        mode = data.get("mode", "all")
        n = max(1, min(int(data.get("n") or 1000), len(df)))
        if mode == "first":
            df = df.head(n)
        elif mode == "random":
            df = df.sample(n=n, random_state=1).sort_index()
        elif mode == "selected":
            idx = [int(i) for i in data.get("rows", []) if 0 <= int(i) < len(df)]
            if not idx:
                raise DataError("No preview rows were selected.", "Tick at least one row in the preview table.")
            df = df.iloc[idx]
        res, summary, alert_list = prediction.analyze_dataframe(df, bundle)
        original_name = upload_id.split("_", 1)[-1]
        summary.update({"filename": original_name, "mode": mode, "truncated": bool(truncated),
                        "max_rows": config.MAX_ANALYSIS_ROWS})
        aid = database.create_analysis(g.user["id"], original_name, summary, info["version"], info["model_name"])
        for col in ("src_ip", "dst_ip", "timestamp"):  # protect against CSV/Excel formula injection
            if col in res.columns:
                res[col] = res[col].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in "=+-@" else v)
        out = config.REPORTS_DIR / f"analysis_{aid}.csv"
        res.to_csv(out, index=False)
        database.set_analysis_results_path(aid, out)
        n_alerts = database.add_alerts(aid, alert_list)
    finally:
        if config.DELETE_UPLOADS_AFTER_ANALYSIS:
            path.unlink(missing_ok=True)
    preview = res.head(200).astype(object).where(res.head(200).notna(), None)
    return jsonify(json_safe({"analysis_id": aid, "summary": summary, "alerts_created": n_alerts,
                              "columns": preview.columns.tolist(), "rows": preview.values.tolist()}))


# ----------------------------------------------------------------------------- #
# API - alerts
# ----------------------------------------------------------------------------- #
@app.get("/api/alerts")
@login_required
@api
def api_alerts():
    a = request.args
    return jsonify(json_safe(database.query_alerts(
        search=a.get("search", "").strip(), status=a.get("status", ""), severity=a.get("severity", ""),
        category=a.get("category", ""), sort=a.get("sort", "id"), order=a.get("order", "desc"),
        page=int(a.get("page", 1) or 1), per_page=15,
        analysis_id=int(a["analysis"]) if a.get("analysis", "").isdigit() else None)))


@app.get("/api/alerts/<int:alert_id>")
@login_required
@api
def api_alert(alert_id):
    al = database.get_alert(alert_id)
    if not al:
        return jsonify(error="Alert not found."), 404
    return jsonify(json_safe(al))


@app.post("/api/alerts/<int:alert_id>/status")
@login_required
@api
def api_alert_status(alert_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")
    if status not in ("New", "Reviewed", "Resolved"):
        raise DataError("Invalid status.", "Use New, Reviewed or Resolved.")
    if not database.get_alert(alert_id):
        return jsonify(error="Alert not found."), 404
    database.update_alert(alert_id, status=status, note=data.get("note"))
    return jsonify(ok=True)


# ----------------------------------------------------------------------------- #
# API - datasets and training (Admin)
# ----------------------------------------------------------------------------- #
def _mapping():
    if config.DATASET_MAPPING_FILE.exists():
        try:
            return json.loads(config.DATASET_MAPPING_FILE.read_text())
        except Exception:
            pass
    return dict(config.DEFAULT_DATASET_FILES)


@app.get("/api/datasets")
@admin_required
@api
def api_datasets():
    files = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(config.DATASETS_DIR.glob("*.csv"))}
    lower = {n.lower(): n for n in files}
    mapping = _mapping()
    slots = []
    for key, title in config.DATASET_SLOT_TITLES.items():
        wanted = mapping.get(key) or config.DEFAULT_DATASET_FILES[key]
        slots.append({"key": key, "title": title, "selected": lower.get(wanted.lower(), wanted),
                      "exists": wanted.lower() in lower})
    return jsonify({"slots": slots, "files": [{"name": n, "size_mb": s} for n, s in files.items()],
                    "mapped": [s["selected"] for s in slots if s["exists"]]})


@app.post("/api/datasets/upload")
@admin_required
@api
def api_dataset_upload():
    f = request.files.get("file")
    name = _check_csv_upload(f, config.MAX_CONTENT_LENGTH // (1024 * 1024))
    f.save(config.DATASETS_DIR / name)
    slot = request.form.get("slot")
    if slot in config.DATASET_SLOT_TITLES:
        m = _mapping()
        m[slot] = name
        config.DATASET_MAPPING_FILE.write_text(json.dumps(m, indent=2))
    return jsonify(ok=True, name=name)


@app.post("/api/datasets/mapping")
@admin_required
@api
def api_dataset_mapping():
    data = request.get_json(silent=True) or {}
    m = {k: secure_filename(str(v)) for k, v in data.items() if k in config.DATASET_SLOT_TITLES}
    config.DATASET_MAPPING_FILE.write_text(json.dumps({**_mapping(), **m}, indent=2))
    return jsonify(ok=True)


@app.post("/api/datasets/validate")
@admin_required
@api
def api_dataset_validate():
    name = secure_filename((request.get_json(silent=True) or {}).get("filename", ""))
    path = config.DATASETS_DIR / name
    if not name or not path.exists():
        raise DataError("File not found in the datasets folder.")
    df = clean_columns(read_csv_safe(path, nrows=20000, low_memory=False))
    label = find_label_column(df)
    if not label:
        return jsonify(ok=False, columns=int(df.shape[1]),
                       message="No label column found (expected a column named 'Label').")
    counts = map_labels(df[label]).value_counts().to_dict()
    return jsonify(ok=True, columns=int(df.shape[1]), label_column=label, classes_in_first_20000_rows=counts)


def _train_worker(paths):
    def progress(pct, msg):
        with TRAIN_LOCK:
            if pct is not None:
                TRAIN["pct"] = pct
            TRAIN["messages"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
    try:
        result = train_model.train_all(paths, progress)
        with TRAIN_LOCK:
            TRAIN["result"] = {"version": result["version"], "best_model": result["best_model"],
                               "warnings": result["warnings"]}
    except DataError as e:
        with TRAIN_LOCK:
            TRAIN["error"], TRAIN["hint"] = e.message, e.hint
    except MemoryError:
        with TRAIN_LOCK:
            TRAIN["error"] = "The computer ran out of memory."
            TRAIN["hint"] = "Lower MAX_ROWS_PER_FILE / MAX_TRAIN_ROWS in config.py and train again."
    except Exception as e:  # noqa
        log.exception("Training failed")
        with TRAIN_LOCK:
            TRAIN["error"], TRAIN["hint"] = f"Training failed: {e}", "See the terminal window for details."
    finally:
        with TRAIN_LOCK:
            TRAIN["running"] = False


@app.post("/api/train/start")
@admin_required
@api
def api_train_start():
    names = (request.get_json(silent=True) or {}).get("files", [])
    paths, missing = train_model.resolve_dataset_files([secure_filename(n) for n in names])
    if missing:
        raise DataError(f"Files not found in the datasets folder: {', '.join(missing)}",
                        "Upload them on this page or copy them into the 'datasets' folder.")
    if not paths:
        raise DataError("No dataset files selected.", "Tick at least one labelled CSV file.")
    with TRAIN_LOCK:
        if TRAIN["running"]:
            raise DataError("Training is already running.")
        TRAIN.update(running=True, pct=0, messages=[], result=None, error=None, hint=None)
    threading.Thread(target=_train_worker, args=(paths,), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/train/status")
@admin_required
def api_train_status():
    with TRAIN_LOCK:
        return jsonify(dict(TRAIN))


# ----------------------------------------------------------------------------- #
# Reports
# ----------------------------------------------------------------------------- #
@app.get("/reports/csv/<int:analysis_id>")
@login_required
def report_csv(analysis_id):
    a = database.get_analysis(analysis_id)
    if not a or not a.get("results_path") or not Path(a["results_path"]).exists():
        abort(404, "Results file not found.")
    return send_file(a["results_path"], as_attachment=True, download_name=f"sou_ids_analysis_{analysis_id}.csv")


@app.get("/reports/pdf/<int:analysis_id>")
@login_required
def report_pdf(analysis_id):
    a = database.get_analysis(analysis_id)
    if not a:
        abort(404, "Analysis not found.")
    out = config.REPORTS_DIR / f"report_{analysis_id}.pdf"
    top = database.query_alerts(analysis_id=analysis_id, sort="confidence", order="desc", per_page=25)["alerts"]
    build_pdf(out, a, prediction.recommendations_for(a["summary"]), top, database.get_active_model())
    return send_file(out, as_attachment=True, download_name=f"sou_ids_report_{analysis_id}.pdf")


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="The file is too large for the server limit."), 413


@app.errorhandler(400)
@app.errorhandler(404)
def http_error(e):
    if request.path.startswith("/api/"):
        return jsonify(error=getattr(e, "description", "Error")), e.code
    return render_template("error.html", code=e.code, message=getattr(e, "description", "")), e.code


if __name__ == "__main__":
    import threading
    import webbrowser

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000")

    threading.Timer(1.5, open_browser).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
