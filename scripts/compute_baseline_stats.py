# ============================================================
# COMPUTE_BASELINE_STATS.PY
# ============================================================
# Computes baseline mean/std/min/max for the raw numeric input
# features (as defined in app/schemas.py) from the ORIGINAL
# training data, and writes them to the specified model version's
# artifacts folder as baseline_stats.json.
#
# This is a build-time / training-time script — it is NOT run by
# the API at request time, and data/Raw/train.csv is intentionally
# excluded from the Docker image (see .dockerignore). The API only
# ever reads the small baseline_stats.json this script produces,
# not the raw CSV.
#
# Run once per model version, right after training that version:
#   python scripts/compute_baseline_stats.py --version v1.0.0
#
# Why this lives with the model version, not as a single global
# file: drift is only meaningful relative to what THAT SPECIFIC
# model was actually trained on. If a future v1.1.0 is trained on
# updated data, it should get its own baseline reflecting that —
# comparing live traffic against a stale v1.0.0 baseline after
# deploying v1.1.0 would produce misleading drift signals.

import argparse
import json

import pandas as pd

# Maps API field names (app/schemas.py) to their corresponding
# column names in the raw training CSV, where they differ.
API_FIELD_TO_RAW_COLUMN = {
    "TotalBsmtSF": "TotalBsmtSF",
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF",
    "YearBuilt": "YearBuilt",
    "YrSold": "YrSold",
    "OverallQual": "OverallQual",
    "OverallCond": "OverallCond",
    "GarageCars": "GarageCars",
    "Fireplaces": "Fireplaces",
    "KitchenAbvGr": "KitchenAbvGr",
    "FullBath": "FullBath",
    "HalfBath": "HalfBath",
    "BsmtFullBath": "BsmtFullBath",
    "BsmtHalfBath": "BsmtHalfBath",
}


def compute_baseline(raw_csv_path: str) -> dict:
    df = pd.read_csv(raw_csv_path)

    stats = {}
    for api_field, raw_column in API_FIELD_TO_RAW_COLUMN.items():
        series = df[raw_column].dropna()
        stats[api_field] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute drift baseline stats for a model version.")
    parser.add_argument("--version", required=True, help="Model version, e.g. v1.0.0")
    parser.add_argument("--raw-csv", default="data/Raw/train.csv", help="Path to raw training CSV")
    args = parser.parse_args()

    baseline = compute_baseline(args.raw_csv)

    output_path = f"artifacts/{args.version}/baseline_stats.json"
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Wrote baseline stats for {len(baseline)} features to {output_path}")
