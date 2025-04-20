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

# Friendly labels for metrics
LABEL_MAP = {
    "nation_count":                 "Nation Count by Alliance",
    "avg_inactivity":               "Average Alliance Inactivity (Days)",
    "Empty_Slots":                  "Total Empty Trade Slots by Alliance",
    "%_empty":                      "% of Nations with Empty Trade Slots",
    "Technology":                   "Total Technology by Alliance",
    "avg_Technology":               "Average Technology by Alliance",
    "Infrastructure":               "Total Infrastructure by Alliance",
    "avg_Infrastructure":           "Average Infrastructure by Alliance",
    "Base_Land":                    "Total Base Land by Alliance",
    "avg_Base_Land":                "Average Base Land by Alliance",
    "Strength":                     "Total Strength by Alliance",
    "avg_Strength":                 "Average Strength by Alliance",
    "Attacking_Casualties":         "Total Attacking Casualties by Alliance",
    "avg_Attacking_Casualties":     "Average Attacking Casualties by Alliance",
    "Defensive_Casualties":         "Total Defensive Casualties by Alliance",
    "avg_Defensive_Casualties":     "Average Defensive Casualties by Alliance"
}

# Exact order of charts
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

# Activity and resource mapping
ACTIVITY_MAP = {
    "Active In The Last 3 Days": 3,
    "Active This Week":           7,
    "Active Last Week":          14,
    "Active Three Weeks Ago":    21,
    "Active More Than Three Weeks Ago": 28
}
RESOURCE_COLS = [f"Connected Resource {i}" for i in range(1, 11)]

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
    # Filter to target alliance
    df = df[df.get("Alliance")==TARGET_ALLIANCE].copy()
    if df.empty:
        return df
    # Numeric conversion
    for col in ["Technology","Infrastructure","Base Land","Strength",
                "Attacking Casualties","Defensive Casualties"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce")
    # Empty trade slots count
    df["Empty_Slots"] = (
        df[RESOURCE_COLS]
          .apply(lambda row: sum(1 for v in row if pd.isna(v) or str(v).strip()==""), axis=1) // 2
    )
    # Activity score to inactivity days
    if "Activity" in df.columns:
        df["avg_inactivity"] = df["Activity"].map(ACTIVITY_MAP)
    return df


def aggregate(df):
    # Aggregate totals and compute per-nation averages and percent empty
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
    )
    # Compute averages per nation
    for total_col in ["Technology","Infrastructure","Base_Land","Strength",
                      "Attacking_Casualties","Defensive_Casualties"]:
        avg_col = f"avg_{total_col}"
        agg[avg_col] = agg[total_col] / agg["nation_count"]
    # Percent of nations with empty slots
    total_nations = df.groupby("snapshot_date")["Nation ID"].count()
    empty_nations = df[df["Empty_Slots"] > 0].groupby("snapshot_date")["Nation ID"].count()
    agg["%_empty"] = (empty_nations.reindex(agg.index, fill_value=0) / total_nations.reindex(agg.index) * 100)
    return agg.sort_index()


def growth(series):
    first, last = series.iloc[0], series.iloc[-1]
    days = max((series.index[-1] - series.index[0]).days, 1)
    return (last - first) / days


def plot_series(series, title, filename):
    plt.figure(figsize=(7,4))
    plt.plot(series.index, series.values, marker="o")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return filename, buf


def delete_last():
    if Path(LAST_ID_FILE).exists():
        last_id = Path(LAST_ID_FILE).read_text().strip()
        if last_id:
            try:
                requests.delete(f"{WEBHOOK_URL}/messages/{last_id}")
            except:
                pass
        Path(LAST_ID_FILE).write_text("")


def record_last(msg_id):
    Path(LAST_ID_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(LAST_ID_FILE).write_text(msg_id)


def post_new(content, attachments):
    delete_last()
    embeds = [{"image": {"url": f"attachment://{fn}"}} for fn,_ in attachments]
    payload = {"content": content, "embeds": embeds}
    files = [(f"file{i}", (fn, buf, "image/png")) for i,(fn, buf) in enumerate(attachments)]
    resp = requests.post(
        WEBHOOK_URL + "?wait=true",
        data={"payload_json": json.dumps(payload)},
        files=files
    )
    resp.raise_for_status()
    return resp.json()["id"]


def main():
    df_raw = load_data()
    df = preprocess(df_raw)
    if df.empty:
        print(f"No data for {TARGET_ALLIANCE}")
        return
    agg = aggregate(df)
    latest = agg.iloc[-1]
    ts = latest.name.strftime("%Y-%m-%d %H:%M")
    # Summary text
    lines = [f"**Aggregated Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for metric in METRICS:
        if metric not in agg.columns:
            continue
        val = latest[metric]
        gr = growth(agg[metric])
        label = LABEL_MAP[metric]
        if pd.isna(val):
            continue
        if isinstance(val, float):
            lines.append(f"- Current {label}: {val:.2f}")
        else:
            lines.append(f"- Current {label}: {int(val):,}")
        lines.append(f"- {label} Growth/Day: {gr:.2f}")
    content = "\n".join(lines)

    # Generate one chart per metric
    attachments = []
    for metric in METRICS:
        if metric in agg.columns:
            fname = metric.lower() + ".png"
            title = f"{LABEL_MAP[metric]} Over Time"
            attachments.append(plot_series(agg[metric], title, fname))

    # Post to Discord
    msg_id = post_new(content, attachments)
    record_last(msg_id)

if __name__ == "__main__":
    main()
