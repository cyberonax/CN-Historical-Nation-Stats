# batch_preprocess.py
import pandas as pd
from pathlib import Path

# reuse your parse_date_from_filename, etc.
from your_helpers import parse_date_from_filename, aggregate_by_alliance

def load_all_zips():
    dfs = []
    for zip_file in Path("downloaded_zips").glob("CyberNations_SE_Nation_Stats_*.zip"):
        date = parse_date_from_filename(zip_file.name)
        with zipfile.ZipFile(zip_file) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
            df['snapshot_date'] = pd.to_datetime(date)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def main():
    raw = load_all_zips()
    raw.to_parquet("precomputed/raw.parquet", index=False)

    agg = aggregate_by_alliance(raw)
    # rename its snapshot_date → date for convenience
    agg.rename(columns={"snapshot_date":"date"}, inplace=True)
    agg.to_parquet("precomputed/alliance_agg.parquet", index=False)

if __name__ == "__main__":
    main()
