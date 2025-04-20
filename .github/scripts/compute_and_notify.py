#!/usr/bin/env python3
import os, re, zipfile, io
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt

# CONFIGURATION
WEBHOOK_URL      = os.environ['DISCORD_WEBHOOK_URL']
ZIP_FOLDER       = "downloaded_zips"
MSG_IDS_FILE     = ".github/scripts/msg_ids.txt"
TARGET_ALLIANCE  = "Freehold of The Wolves"

# Friendly labels
LABEL_MAP = {
    "nation_count":                "Nation Count by Alliance",
    "avg_inactivity":              "Average Alliance Inactivity (Days)",
    "Empty_Slots":                 "Total Empty Trade Slots by Alliance",
    "%_empty":                     "% of Nations with Empty Trade Slots",
    "Technology":                  "Total Technology by Alliance",
    "avg_Technology":              "Average Technology by Alliance",
    "Infrastructure":              "Total Infrastructure by Alliance",
    "avg_Infrastructure":          "Average Infrastructure by Alliance",
    "Base_Land":                   "Total Base Land by Alliance",
    "avg_Base_Land":               "Average Base Land by Alliance",
    "Strength":                    "Total Strength by Alliance",
    "avg_Strength":                "Average Strength by Alliance",
    "Attacking_Casualties":        "Total Attacking Casualties by Alliance",
    "avg_Attacking_Casualties":    "Average Attacking Casualties by Alliance",
    "Defensive_Casualties":        "Total Defensive Casualties by Alliance",
    "avg_Defensive_Casualties":    "Average Defensive Casualties by Alliance"
}

# Exact order of metrics
METRICS = [
    "nation_count",
    "avg_inactivity",
    "Empty_Slots",
    "%_empty",
    "Technology",
    "avg_Technology",
    "Infrastructure",
    "avg_Infrastructure",
    "Base_Land",
    "avg_Base_Land",
    "Strength",
    "avg_Strength",
    "Attacking_Casualties",
    "avg_Attacking_Casualties",
    "Defensive_Casualties",
    "avg_Defensive_Casualties"
]

RESOURCE_COLS = [f"Connected Resource {i}" for i in range(1, 11)]
ACTIVITY_MAP = {
    "Active In The Last 3 Days": 3,
    "Active This Week": 7,
    "Active Last Week": 14,
    "Active Three Weeks Ago": 21,
    "Active More Than Three Weeks Ago": 28
}

# HELPER FUNCTIONS

def parse_date(fn):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", fn)
    if not m:
        return None
    token, zipid = m.groups()
    hour = 0 if zipid == "510001" else 12
    for md in (1, 2):
        for dd in (1, 2):
            if md + dd + 4 == len(token):
                try:
                    month = int(token[:md]); day = int(token[md:md+dd]); year = int(token[md+dd:])
                    return datetime(year, month, day, hour)
                except:
                    pass
    return None


def load_data():
    dfs = []
    for zp in Path(ZIP_FOLDER).glob("CyberNations_SE_Nation_Stats_*.zip"):
        dt = parse_date(zp.name)
        if not dt:
            continue
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
            df["snapshot_date"] = pd.to_datetime(dt)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def preprocess(df):
    df = df[df.get("Alliance") == TARGET_ALLIANCE].copy()
    # Ensure numeric conversions
    for col in ["Technology", "Infrastructure", "Base Land", "Strength",
                "Attacking Casualties", "Defensive Casualties"]:
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    # Compute empty slots
    df["Empty_Slots"] = (
        df[RESOURCE_COLS].apply(lambda row: sum(1 for v in row if pd.isna(v) or str(v).strip()==""), axis=1) // 2
    )
    # Map activity to inactivity days
    if "Activity" in df:
        df["avg_inactivity"] = df["Activity"].map(ACTIVITY_MAP)
    return df


def aggregate(df):
    agg = df.groupby("snapshot_date").agg(
        nation_count=("Nation ID", "count"),
        Technology=("Technology", "sum"),
        Infrastructure=("Infrastructure", "sum"),
        Base_Land=("Base Land", "sum"),
        Strength=("Strength", "sum"),
        Attacking_Casualties=("Attacking Casualties", "sum"),
        Defensive_Casualties=("Defensive Casualties", "sum"),
        Empty_Slots=("Empty_Slots", "sum"),
        avg_inactivity=("avg_inactivity", "mean")
    ).sort_index()
    # Compute per-nation averages
    for total_col in ["Technology", "Infrastructure", "Base_Land", "Strength",
                      "Attacking_Casualties", "Defensive_Casualties"]:
        agg[f"avg_{total_col}"] = agg[total_col] / agg["nation_count"]
    # Compute percent empty
    total_n = df.groupby("snapshot_date")["Nation ID"].count()
    empty_n = df[df["Empty_Slots"] > 0].groupby("snapshot_date")["Nation ID"].count()
    agg["%_empty"] = (empty_n.reindex(agg.index, fill_value=0) / total_n.reindex(agg.index)) * 100
    return agg


def growth(s):
    first, last = s.iloc[0], s.iloc[-1]
    days = max((s.index[-1] - s.index[0]).days, 1)
    return (last - first) / days


def plot_series(series, title, filename):
    # Uniform figure size for all charts
    plt.figure(figsize=(8, 4))
    plt.plot(series.index, series.values, marker="o")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return filename, buf


def delete_old():
    if not Path(MSG_IDS_FILE).exists():
        return
    ids = Path(MSG_IDS_FILE).read_text().splitlines()
    for mid in ids:
        try:
            requests.delete(f"{WEBHOOK_URL}/messages/{mid}")
        except:
            pass
    Path(MSG_IDS_FILE).write_text("")


def record_ids(ids):
    Path(MSG_IDS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(MSG_IDS_FILE).write_text("\n".join(ids))


def post_chunks(content, attachments, chunk_size=10):
    delete_old()
    msg_ids = []
    for i in range(0, len(attachments), chunk_size):
        batch = attachments[i:i+chunk_size]
        embeds = [{"image": {"url": f"attachment://{fn}"}} for fn, _ in batch]
        payload = {"content": content, "embeds": embeds}
        files = [(f"file{j}", (fn, buf, "image/png")) for j, (fn, buf) in enumerate(batch)]
        resp = requests.post(
            WEBHOOK_URL + "?wait=true",
            data={"payload_json": json.dumps(payload)},
            files=files
        )
        resp.raise_for_status()
        msg_ids.append(resp.json()["id"])
        content = ""  # only include text on first chunk
    record_ids(msg_ids)
    return msg_ids


def main():
    df0 = load_data()
    df = preprocess(df0)
    if df.empty:
        print(f"No data for {TARGET_ALLIANCE}")
        return
    agg = aggregate(df)
    latest = agg.iloc[-1]
    ts = latest.name.strftime("%Y-%m-%d %H:%M")
    # Build summary text
    lines = [f"**Aggregated Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for metric in METRICS:
        if metric not in agg.columns:
            continue
        val = latest[metric]
        gr = growth(agg[metric])
        lbl = LABEL_MAP[metric]
        if pd.isna(val):
            continue
        if isinstance(val, float):
            lines.append(f"- Current {lbl}: {val:.2f}")
        else:
            lines.append(f"- Current {lbl}: {int(val):,}")
        lines.append(f"- {lbl} Growth/Day: {gr:.2f}")
    content = "\n".join(lines)

    # Generate attachments in the specified order
    attachments = []
    for metric in METRICS:
        if metric in agg.columns:
            fname = f"{metric}.png"
            title = f"{LABEL_MAP[metric]} Over Time"
            attachments.append(plot_series(agg[metric], title, fname))

    # Post in chunks to handle Discord limits
    post_chunks(content, attachments, chunk_size=10)

if __name__ == "__main__":
    main()
