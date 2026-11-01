
from pathlib import Path
import pandas as pd

# Find the existing demo CSV in your project
matches = list(Path(".").rglob(
    "smartguard_mixed_normal_attack_demo.csv"
))

if not matches:
    print("Demo CSV not found.")
    print("Place smartguard_mixed_normal_attack_demo.csv")
    print("inside your project folder and run again.")
else:
    source = matches[0]
    df = pd.read_csv(source)

    # Add the missing protocol feature for demo testing.
    # TCP=6, UDP=17. This is synthetic demo data,
    # not verified real network protocol information.
    if "protocol" not in df.columns:
        df["protocol"] = 6

    output = source.parent / "sou_ids_demo_test.csv"
    df.to_csv(output, index=False)

    print("Demo test CSV created:", output)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Protocol column added:", "protocol" in df.columns)
