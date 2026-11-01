"""
Sou-IDS configuration.
Everything you might want to change lives here.
"""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
DATASETS_DIR = BASE_DIR / "datasets"
INSTANCE_DIR = BASE_DIR / "instance"
DATABASE_PATH = INSTANCE_DIR / "sou_ids.db"
DATASET_MAPPING_FILE = DATASETS_DIR / "mapping.json"

for _d in (MODELS_DIR, UPLOADS_DIR, REPORTS_DIR, DATASETS_DIR, INSTANCE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_secret_key():
    """Secret key comes from an environment variable, otherwise it is generated
    once and stored in instance/secret_key (never hard-coded in source)."""
    env_key = os.environ.get("SOU_IDS_SECRET_KEY")
    if env_key:
        return env_key
    key_file = INSTANCE_DIR / "secret_key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
    return key_file.read_text().strip()


SECRET_KEY = _load_secret_key()

# ---- Branding ---------------------------------------------------------------
APP_NAME = "Sou-IDS"
COMPANY_NAME = "Demo Company Pvt. Ltd."   # change to your company / college name

# ---- Demo accounts (DEMONSTRATION ONLY - not production authentication) ------
DEMO_USERS = [
    {"username": "admin", "email": "admin@sou-ids.local", "password": "Admin@12345", "role": "Admin"},
    {"username": "analyst", "email": "analyst@sou-ids.local", "password": "Analyst@12345", "role": "Security Analyst"},
]

# ---- Upload limits ----------------------------------------------------------
MAX_CONTENT_LENGTH = 1200 * 1024 * 1024      # hard limit for any request (dataset files)
MAX_DETECT_UPLOAD_MB = 200                    # limit for "Traffic Detection" CSV uploads
ALLOWED_EXTENSIONS = {".csv"}
MAX_ANALYSIS_ROWS = 500_000                   # rows analysed per uploaded file (memory safety)
DELETE_UPLOADS_AFTER_ANALYSIS = True          # do not keep raw network data on disk

# ---- Training settings (tune for your laptop's RAM) --------------------------
MAX_ROWS_PER_FILE = 150_000      # random sample read from each big CSV file
MAX_TRAIN_ROWS = 250_000         # total rows kept for training after cleaning
MIN_CLASS_SAMPLES = 10           # classes with fewer rows are removed (cannot be learned/evaluated)
TEST_SIZE = 0.20                 # held-out test share
VALIDATION_SIZE = 0.20           # share of the TRAINING part used as validation
RANDOM_STATE = 42
RF_TREES = 100

# ---- Alerts -----------------------------------------------------------------
MAX_ALERTS_PER_ANALYSIS = 500    # top-confidence attack rows become alert records

# ---- Default dataset file names (change in the Datasets page if yours differ) --
DEFAULT_DATASET_FILES = {
    "monday_benign": "Monday-WorkingHours.pcap_ISCX.csv",
    "wednesday_dos": "Wednesday-workingHours.pcap_ISCX.csv",
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "friday_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
}
DATASET_SLOT_TITLES = {
    "monday_benign": "Monday (benign traffic)",
    "wednesday_dos": "Wednesday (DoS)",
    "friday_ddos": "Friday afternoon (DDoS)",
    "friday_portscan": "Friday afternoon (PortScan)",
}
