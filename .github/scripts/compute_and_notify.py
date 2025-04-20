#!/usr/bin/env python3
import os, re, zipfile, io, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt

# CONFIGURATION
WEBHOOK_URL     = os.environ['DISCORD_WEBHOOK_URL']
ZIP_FOLDER      = "downloaded_zips"
MSG_IDS_FILE    = ".github/scripts/msg_ids.txt"
TARGET_ALLIANCE = "Freehold of The Wolves"

# Labels for metrics, with exact phrasing
LABEL_MAP = {
    "nation_count":               "Nation Count Over Time",
    "avg_inactivity":             "Average Alliance Inactivity Over Time (Days)",
    "Empty_Slots":                "Total Empty Trade Slots Over Time",
    "%_empty":                    "% of Nations with Empty Trade Slots Over Time",
    "Technology":                 "Total Technology Over Time",
    "avg_Technology":             "Average Technology Over Time",
    "Infrastructure":             "Total Infrastructure Over Time",
    "avg_Infrastructure":         "Average Infrastructure Over Time",
    "Base_Land":                  "Total Base Land Over Time",
    "avg_Base_Land":              "Average Base Land Over Time",
    "Strength":                   "Total Strength Over Time",
    "avg_Strength":               "Average Strength Over Time",
    "Attacking_Casualties":       "Total Attacking Casualties Over Time",
    "avg_Attacking_Casualties":   "Average Attacking Casualties Over Time",
    "Defensive_Casualties":       "Total Defensive Casualties Over Time",
    "avg_Defensive_Casualties":   "Average Defensive Casualties Over Time"
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

# Resource and activity mapping
RESOURCE_COLS = [f"Connected Resource {i}" for i in range(1, 11)]
ACTIVITY_MAP = {
    "Active In The Last 3 Days": 3,
    "Active This Week": 7,
    "Active Last Week": 14,
    "Active Three Weeks Ago": 21,
    "Active More Than Three Weeks Ago": 28
}

# DATA FUNCTIONS

def parse_date(fn):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", fn)
    if not m:
        return None
    token, zipid = m.groups()
    hour = 0 if zipid == "510001" else 12
    for md in (1,2):
        for dd in (1,2):
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
    if df.empty:
        return df
    # numeric
    for col in ["Technology","Infrastructure","Base Land","Strength",
                "Attacking Casualties","Defensive Casualties"]:
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce")
    # empty slots
    df["Empty_Slots"] = df[RESOURCE_COLS].apply(
        lambda row: sum(1 for v in row if pd.isna(v) or str(v).strip()==""), axis=1
    ) // 2
    # inactivity
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
    # per-nation averages
    for tot in ["Technology","Infrastructure","Base_Land","Strength",
                "Attacking_Casualties","Defensive_Casualties"]:
        agg[f"avg_{tot}"] = agg[tot] / agg["nation_count"]
    # percent empty
    total_n = df.groupby("snapshot_date")["Nation ID"].count()
    empty_n = df[df["Empty_Slots"]>0].groupby("snapshot_date")["Nation ID"].count()
    agg["%_empty"] = (empty_n.reindex(agg.index, fill_value=0)/total_n.reindex(agg.index))*100
    return agg


def growth(series):
    first, last = series.iloc[0], series.iloc[-1]
    days = max((series.index[-1] - series.index[0]).days, 1)
    return (last - first) / days

# PLOTTING

def plot_series(series, title, filename):
    plt.figure(figsize=(10,6))
    plt.plot(series.index, series.values, marker="o", linewidth=2)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png"); buf.seek(0); plt.close()
    return filename, buf

# CLEAN UP / POST

def delete_old():
    if Path(MSG_IDS_FILE).exists():
        mids = Path(MSG_IDS_FILE).read_text().splitlines()
        for mid in mids:
            try: requests.delete(f"{WEBHOOK_URL}/messages/{mid}")
            except: pass
        Path(MSG_IDS_FILE).write_text("")


def record_ids(mids):
    Path(MSG_IDS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(MSG_IDS_FILE).write_text("\n".join(mids))


def post_in_chunks(content, attachments, chunk_size=10):
    delete_old()
    msg_ids = []
    for i in range(0, len(attachments), chunk_size):
        batch = attachments[i:i+chunk_size]
        embeds = [{"image": {"url": f"attachment://{fn}"}} for fn,_ in batch]
        payload = {"content": content, "embeds": embeds}
        files = [(f"file{j}", (fn, buf, "image/png")) for j,(fn, buf) in enumerate(batch)]
        resp = requests.post(
            WEBHOOK_URL + "?wait=true",
            data={"payload_json": json.dumps(payload)},
            files=files
        )
        resp.raise_for_status()
        msg_ids.append(resp.json()["id"])
        content = ""  # only include text in first message
    record_ids(msg_ids)

# MAIN

def main():
    df0 = load_data()
    df = preprocess(df0)
    if df.empty:
        print(f"No data for {TARGET_ALLIANCE}")
        return
    agg = aggregate(df)
    last = agg.iloc[-1]
    ts = last.name.strftime("%Y-%m-%d %H:%M")
    # summary
    lines = [f"**Aggregated Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for metric in METRICS:
        if metric not in agg.columns:
            continue
        val = last[metric]; g = growth(agg[metric]); label = LABEL_MAP[metric]
        display = f"{val:.2f}" if isinstance(val, float) else f"{int(val):,}"
        lines.append(f"- Current {label}: {display}")
        lines.append(f"- {label} Growth/Day: {g:.2f}")
    content = "\n".join(lines)
    # attachments
    attachments = []
    for metric in METRICS:
        if metric in agg.columns:
            fname = metric + ".png"
            title = LABEL_MAP[metric]
            attachments.append(plot_series(agg[metric], title, fname))
    # post
    post_in_chunks(content, attachments, chunk_size=10)

if __name__ == "__main__":
    main()
