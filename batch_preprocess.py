#!/usr/bin/env python3
# batch_preprocess.py

import pandas as pd
import zipfile
import re
from pathlib import Path
from datetime import datetime

##############################
# INLINE HELPER FUNCTIONS
##############################

def parse_date_from_filename(filename):
    """
    Extract and parse the date from a filename that follows the pattern:
    "CyberNations_SE_Nation_Stats_<dateToken><zipid>.zip"
    """
    pattern = r'^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$'
    match = re.match(pattern, filename)
    if not match:
        return None
    date_token, zip_id = match.groups()
    hour = 0 if zip_id == "510001" else 12

    for m_digits in [1, 2]:
        for d_digits in [1, 2]:
            if m_digits + d_digits + 4 == len(date_token):
                try:
                    month = int(date_token[:m_digits])
                    day   = int(date_token[m_digits:m_digits + d_digits])
                    year  = int(date_token[m_digits + d_digits:m_digits + d_digits + 4])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month, day, hour=hour)
                except Exception:
                    continue
    return None


def aggregate_by_alliance(df):
    """
    Aggregates nation stats by snapshot_date and Alliance.
    """
    numeric_cols = [
        'Technology', 'Infrastructure', 'Base Land',
        'Strength', 'Attacking Casualties', 'Defensive Casualties'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '').str.strip(),
                errors='coerce'
            )

    agg_dict = {
        'Nation ID': 'count',
        'Technology': 'sum',
        'Infrastructure': 'sum',
        'Base Land': 'sum',
        'Strength': 'sum',
        'Attacking Casualties': 'sum',
        'Defensive Casualties': 'sum',
    }

    return (
        df
        .groupby(['snapshot_date', 'Alliance'])
        .agg(agg_dict)
        .reset_index()
        .rename(columns={'Nation ID': 'nation_count'})
    )


##############################
# MAIN PREPROCESSING LOGIC
##############################

def load_all_zips():
    """
    Read every matching ZIP in downloaded_zips/, parse date,
    read its single CSV, and concat into a DataFrame.
    """
    data_frames = []
    zip_folder = Path("downloaded_zips")
    if not zip_folder.exists():
        raise FileNotFoundError(f"{zip_folder} not found")

    for zip_path in zip_folder.glob("CyberNations_SE_Nation_Stats_*.zip"):
        snapshot_date = parse_date_from_filename(zip_path.name)
        if snapshot_date is None:
            print(f"⚠️  Skipping unrecognized file: {zip_path.name}")
            continue

        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            if not members:
                print(f"⚠️  No files inside {zip_path.name}")
                continue
            with z.open(members[0]) as f:
                df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
                df['snapshot_date'] = pd.to_datetime(snapshot_date)
                data_frames.append(df)

    if not data_frames:
        raise RuntimeError("No data loaded from any ZIP files")

    return pd.concat(data_frames, ignore_index=True)


def main():
    # 1) Load raw snapshots
    print("▶️  Loading all ZIPs…")
    raw = load_all_zips()

    # Prepare output directory inside your Streamlit UI folder
    out_dir = Path(__file__).parent / "streamlit_ui" / "precomputed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Save out raw parquet
    raw_out = out_dir / "raw.parquet"
    raw.to_parquet(raw_out, index=False)
    print(f"✅  Wrote raw data to {raw_out}")

    # 3) Compute and save alliance aggregates
    print("▶️  Computing alliance aggregates…")
    agg = aggregate_by_alliance(raw)
    agg = agg.rename(columns={"snapshot_date": "date"})
    agg_out = out_dir / "alliance_agg.parquet"
    agg.to_parquet(agg_out, index=False)
    print(f"✅  Wrote aggregated data to {agg_out}")


if __name__ == "__main__":
    main()
