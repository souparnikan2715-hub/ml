"""
database.py - SQLite storage (users, analyses/prediction history, alerts, model versions).
Only summary data and prediction results are stored - never raw feature rows or passwords
in plain text.
"""
import json
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

import config


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_db():
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    model_path TEXT NOT NULL,
    dataset_files TEXT,
    n_rows INTEGER,
    classes TEXT,
    metrics_json TEXT,
    is_active INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS analyses(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    user_id INTEGER,
    filename TEXT,
    total INTEGER, normal INTEGER, attack INTEGER, attack_pct REAL,
    model_version TEXT, model_name TEXT,
    summary_json TEXT,
    results_path TEXT
);
CREATE TABLE IF NOT EXISTS alerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    row_number INTEGER,
    created_at TEXT NOT NULL,
    event_time TEXT,
    src_ip TEXT, dst_ip TEXT,
    prediction TEXT NOT NULL,
    category TEXT,
    confidence REAL,
    severity TEXT,
    status TEXT NOT NULL DEFAULT 'New',
    note TEXT,
    FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_analysis ON alerts(analysis_id);
"""


def init_db():
    with get_db() as db:
        db.executescript(SCHEMA)
        for u in config.DEMO_USERS:
            if not db.execute("SELECT 1 FROM users WHERE username=?", (u["username"],)).fetchone():
                db.execute(
                    "INSERT INTO users(username,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (u["username"], u["email"], generate_password_hash(u["password"]), u["role"], now()))


# ----------------------------- users ---------------------------------------- #
def authenticate(login, password):
    with get_db() as db:
        row = db.execute("SELECT * FROM users WHERE lower(username)=lower(?) OR lower(email)=lower(?)",
                         (login.strip(), login.strip())).fetchone()
    if row and check_password_hash(row["password_hash"], password):
        return dict(row)
    return None


def get_user(user_id):
    with get_db() as db:
        row = db.execute("SELECT id,username,email,role FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


# ----------------------------- models --------------------------------------- #
def register_model(model_name, model_path, dataset_files, n_rows, classes, metrics):
    with get_db() as db:
        count = db.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
        version = f"v{count + 1}"
        db.execute("UPDATE model_versions SET is_active=0")
        db.execute(
            "INSERT INTO model_versions(version,model_name,trained_at,model_path,dataset_files,n_rows,classes,"
            "metrics_json,is_active) VALUES(?,?,?,?,?,?,?,?,1)",
            (version, model_name, now(), str(model_path), json.dumps(dataset_files), n_rows,
             json.dumps(classes), json.dumps(metrics)))
    return version


def next_version_number():
    with get_db() as db:
        return db.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0] + 1


def get_active_model():
    with get_db() as db:
        row = db.execute("SELECT * FROM model_versions WHERE is_active=1 ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    d = dict(row)
    d["dataset_files"] = json.loads(d["dataset_files"] or "[]")
    d["classes"] = json.loads(d["classes"] or "[]")
    d["metrics"] = json.loads(d["metrics_json"] or "{}")
    return d


def list_models():
    with get_db() as db:
        rows = db.execute("SELECT id,version,model_name,trained_at,n_rows,is_active FROM model_versions "
                          "ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


# ----------------------------- analyses ------------------------------------- #
def create_analysis(user_id, filename, summary, model_version, model_name):
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO analyses(created_at,user_id,filename,total,normal,attack,attack_pct,model_version,"
            "model_name,summary_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (now(), user_id, filename, summary["total"], summary["normal"], summary["attack"],
             summary["attack_pct"], model_version, model_name, json.dumps(summary)))
        return cur.lastrowid


def set_analysis_results_path(analysis_id, path):
    with get_db() as db:
        db.execute("UPDATE analyses SET results_path=? WHERE id=?", (str(path), analysis_id))


def get_analysis(analysis_id=None):
    with get_db() as db:
        if analysis_id is None:
            row = db.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 1").fetchone()
        else:
            row = db.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["summary"] = json.loads(d.pop("summary_json") or "{}")
    return d


def list_analyses(limit=50):
    with get_db() as db:
        rows = db.execute("SELECT id,created_at,filename,total,normal,attack,attack_pct,model_version "
                          "FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# ----------------------------- alerts --------------------------------------- #
def add_alerts(analysis_id, alerts):
    if not alerts:
        return 0
    t = now()
    with get_db() as db:
        db.executemany(
            "INSERT INTO alerts(analysis_id,row_number,created_at,event_time,src_ip,dst_ip,prediction,category,"
            "confidence,severity) VALUES(?,?,?,?,?,?,?,?,?,?)",
            [(analysis_id, a["row_number"], t, a.get("event_time"), a.get("src_ip"), a.get("dst_ip"),
              a["prediction"], a["category"], a.get("confidence"), a["severity"]) for a in alerts])
    return len(alerts)


SORTABLE = {"id", "created_at", "event_time", "src_ip", "dst_ip", "category", "severity_rank", "status",
            "confidence"}


def query_alerts(search="", status="", severity="", category="", sort="id", order="desc", page=1, per_page=15,
                 analysis_id=None):
    where, args = [], []
    if search:
        where.append("(src_ip LIKE ? OR dst_ip LIKE ? OR category LIKE ? OR CAST(id AS TEXT)=?)")
        like = f"%{search}%"
        args += [like, like, like, search]
    for col, val in (("status", status), ("severity", severity), ("category", category)):
        if val:
            where.append(f"{col}=?")
            args.append(val)
    if analysis_id:
        where.append("analysis_id=?")
        args.append(analysis_id)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    sort = sort if sort in SORTABLE else "id"
    order = "ASC" if str(order).lower() == "asc" else "DESC"
    rank = ("CASE severity WHEN 'Critical' THEN 4 WHEN 'High' THEN 3 WHEN 'Medium' THEN 2 ELSE 1 END")
    order_expr = rank if sort == "severity_rank" else sort
    with get_db() as db:
        total = db.execute(f"SELECT COUNT(*) FROM alerts {clause}", args).fetchone()[0]
        rows = db.execute(
            f"SELECT * FROM alerts {clause} ORDER BY {order_expr} {order}, id DESC LIMIT ? OFFSET ?",
            args + [per_page, (max(page, 1) - 1) * per_page]).fetchall()
        cats = [r[0] for r in db.execute("SELECT DISTINCT category FROM alerts ORDER BY category").fetchall()]
    return {"total": total, "page": page, "per_page": per_page, "alerts": [dict(r) for r in rows],
            "categories": cats}


def get_alert(alert_id):
    with get_db() as db:
        row = db.execute("SELECT a.*, n.filename FROM alerts a LEFT JOIN analyses n ON n.id=a.analysis_id "
                         "WHERE a.id=?", (alert_id,)).fetchone()
    return dict(row) if row else None


def update_alert(alert_id, status=None, note=None):
    with get_db() as db:
        if status:
            db.execute("UPDATE alerts SET status=? WHERE id=?", (status, alert_id))
        if note is not None:
            db.execute("UPDATE alerts SET note=? WHERE id=?", (note[:500], alert_id))


def recent_alerts(limit=8):
    with get_db() as db:
        rows = db.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def alert_status_counts():
    with get_db() as db:
        rows = db.execute("SELECT status, COUNT(*) c FROM alerts GROUP BY status").fetchall()
    return {r["status"]: r["c"] for r in rows}


def dashboard_totals():
    with get_db() as db:
        r = db.execute("SELECT COUNT(*) n, COALESCE(SUM(total),0) t, COALESCE(SUM(normal),0) nm, "
                       "COALESCE(SUM(attack),0) a FROM analyses").fetchone()
    return {"analyses": r["n"], "total": r["t"], "normal": r["nm"], "attack": r["a"],
            "attack_pct": round(100.0 * r["a"] / r["t"], 2) if r["t"] else 0.0}
