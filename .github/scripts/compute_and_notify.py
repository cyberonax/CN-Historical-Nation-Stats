#!/usr/bin/env python3
import os, re, zipfile, io
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt

# CONFIGURATION
WEBHOOK_URL     = os.environ['DISCORD_WEBHOOK_URL']
ZIP_FOLDER      = "downloaded_zips"
LAST_ID_FILE    = ".github/scripts/last_msg_id.txt"
TARGET_ALLIANCE = "Freehold of The Wolves"

# Mapping column names to friendly labels
LABEL_MAP = {
    "nation_count":               "Nation Count",
    "avg_inactivity":             "Average Alliance Inactivity (Days)",
    "Empty_Slots":                "Total Empty Trade Slots",
    "%_empty":                    "% of Nations with Empty Trade Slots",
    "Technology":                 "Total Technology",
    "avg_Technology":             "Average Technology",
    "Infrastructure":             "Total Infrastructure",
    "avg_Infrastructure":         "Average Infrastructure",
    "Base_Land":                  "Total Base Land",
    "avg_Base_Land":              "Average Base Land",
    "Strength":                   "Total Strength",
    "avg_Strength":               "Average Strength",
    "Attacking_Casualties":       "Total Attacking Casualties",
    "avg_Attacking_Casualties":   "Average Attacking Casualties",
    "Defensive_Casualties":       "Total Defensive Casualties",
    "avg_Defensive_Casualties":   "Average Defensive Casualties"
}

# Desired order of metrics
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

# Activity mapping and resource columns
ACTIVITY_MAP = {
    "Active In The Last 3 Days":        3,
    "Active This Week":                  7,
    "Active Last Week":                  14,
    "Active Three Weeks Ago":            21,
    "Active More Than Three Weeks Ago":  28
}
RESOURCE_COLS = [f"Connected Resource {i}" for i in range(1, 11)]

# Helper functions

def parse_date(fn):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", fn)
    if not m: return None
    token, zipid = m.groups()
    hour = 0 if zipid == "510001" else 12
    for md in (1,2):
        for dd in (1,2):
            if md+dd+4 == len(token):
                try:
                    month = int(token[:md]); day = int(token[md:md+dd]); year = int(token[md+dd:])
                    return datetime(year, month, day, hour)
                except: pass
    return None


def load_data():
    dfs = []
    for zp in Path(ZIP_FOLDER).glob("*.zip"):
        dt = parse_date(zp.name)
        if not dt: continue
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
            df["snapshot_date"] = pd.to_datetime(dt)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def preprocess(df):
    df = df[df.get("Alliance") == TARGET_ALLIANCE].copy()
    # Numeric conversion
    for col in ["Technology","Infrastructure","Base Land","Strength",
                "Attacking Casualties","Defensive Casualties"]:
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce")
    # Empty trade slots
    df["Empty_Slots"] = df[RESOURCE_COLS].apply(
        lambda r: sum(1 for v in r if pd.isna(v) or str(v).strip()==""), axis=1
    ) // 2
    # Activity
    if "Activity" in df:
        df["avg_inactivity"] = df["Activity"].map(ACTIVITY_MAP)
    return df


def aggregate(df):
    agg = df.groupby("snapshot_date").agg(
        nation_count=("Nation ID","count"),
        Technology=("Technology","sum"),
        Infrastructure=("Infrastructure","sum"),
        Base_Land=("Base Land","sum"),
        Strength=("Strength","sum"),
        Attacking_Casualties=("Attacking Casualties","sum"),
        Defensive_Casualties=("Defensive Casualties","sum"),
        Empty_Slots=("Empty_Slots","sum"),
        avg_inactivity=("avg_inactivity","mean")
    ).sort_index()
    # Per-nation averages
    for base in ["Technology","Infrastructure","Base_Land","Strength",
                 "Attacking_Casualties","Defensive_Casualties"]:
        agg[f"avg_{base}"] = agg[base] / agg["nation_count"]
    # % empty
    total = df.groupby("snapshot_date")["Nation ID"].count()
    empties = df[df["Empty_Slots"]>0].groupby("snapshot_date")["Nation ID"].count()
    agg["%_empty"] = (empties.reindex(agg.index,0)/total.reindex(agg.index,0))*100
    return agg


def growth(s):
    first, last = s.iloc[0], s.iloc[-1]
    days = max((last.name - first.name).days, 1)
    return (last - first) / days


def plot_grid(metrics, agg, fname):
    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4*rows))
    axes = axes.flatten()
    for ax, m in zip(axes, metrics):
        ax.plot(agg.index, agg[m], marker='o')
        ax.set_title(LABEL_MAP[m])
        ax.tick_params(axis='x', rotation=45)
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
    return fname, buf


def delete_last():
    if Path(LAST_ID_FILE).exists():
        mid = Path(LAST_ID_FILE).read_text().strip()
        if mid:
            try: requests.delete(f"{WEBHOOK_URL}/messages/{mid}")
            except: pass
        Path(LAST_ID_FILE).write_text("")


def record_last(mid):
    Path(LAST_ID_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(LAST_ID_FILE).write_text(mid)


def post_new(content, attachments):
    delete_last()
    embeds=[{"image":{"url":f"attachment://{fn}"}} for fn,_ in attachments]
    payload={"content":content, "embeds":embeds}
    files=[(f"file{i}",(fn,buf,"image/png")) for i,(fn,buf) in enumerate(attachments)]
    resp = requests.post(f"{WEBHOOK_URL}?wait=true", data={"payload_json":json.dumps(payload)}, files=files)
    resp.raise_for_status()
    return resp.json()["id"]


def main():
    df0 = load_data()
    df = preprocess(df0)
    if df.empty:
        print("No data for alliance")
        return
    agg = aggregate(df)
    latest = agg.iloc[-1]
    ts = latest.name.strftime("%Y-%m-%d %H:%M")

    # Text summary
    lines=[f"**Aggregated Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for m in METRICS:
        if m not in agg.columns: continue
        curr = agg[m].iloc[-1]
        gr   = growth(agg[m])
        lbl  = LABEL_MAP[m]
        val_str = f"{curr:.2f}" if isinstance(curr, float) else f"{int(curr):,}"
        lines.append(f"- Current {lbl}: {val_str}")
        lines.append(f"- {lbl} Growth/Day: {gr:.2f}")
    content = "\n".join(lines)

    # Generate two grid attachments
    half = len(METRICS)//2
    att1 = plot_grid(METRICS[:half], agg, 'stats_part1.png')
    att2 = plot_grid(METRICS[half:], agg, 'stats_part2.png')
    attachments = [att1, att2]

    # Post and record
    mid = post_new(content, attachments)
    record_last(mid)

if __name__=="__main__":
    main()
