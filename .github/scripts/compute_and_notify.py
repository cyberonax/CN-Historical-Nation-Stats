#!/usr/bin/env python3
import os, re, zipfile, io
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt

# CONFIGURATION
WEBHOOK_URL     = os.environ['DISCORD_WEBHOOK_URL']
ZIP_FOLDER      = "downloaded_zips"
LAST_ID_FILE    = ".github/scripts/last_msg_id.txt"
TARGET_ALLIANCE = "Freehold of The Wolves"

# Friendly labels
LABEL_MAP = {
    "nation_count":         "Nation Count",
    "avg_inactivity":       "Avg Inactivity (Days)",
    "Empty Slots Count":    "Total Empty Trade Slots",
    "Technology":           "Total Technology",
    "Infrastructure":       "Total Infrastructure",
    "Base Land":            "Total Base Land",
    "Strength":             "Total Strength",
    "Attacking Casualties": "Total Attacking Casualties",
    "Defensive Casualties": "Total Defensive Casualties"
}

# Order for summary and charts
METRICS = [
    "nation_count",
    "avg_inactivity",
    "Empty Slots Count",
    "Technology",
    "Infrastructure",
    "Base Land",
    "Strength",
    "Attacking Casualties",
    "Defensive Casualties"
]

# Activity mapping and resource columns
ACTIVITY_MAP = {
    "Active In The Last 3 Days":        3,
    "Active This Week":                  7,
    "Active Last Week":                 14,
    "Active Three Weeks Ago":           21,
    "Active More Than Three Weeks Ago": 28
}
RESOURCE_COLS = [f"Connected Resource {i}" for i in range(1, 11)]

# FUNCTIONS

def parse_date(fn):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", fn)
    if not m: return None
    token, zipid = m.groups()
    hour = 0 if zipid == "510001" else 12
    for md in (1,2):
        for dd in (1,2):
            if md+dd+4 == len(token):
                try:
                    m_ = int(token[:md]); d_ = int(token[md:md+dd]); y_ = int(token[md+dd:])
                    return datetime(y_, m_, d_, hour)
                except:
                    pass
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
    if df.empty: return df
    # Numeric conversion
    for c in ["Technology","Infrastructure","Base Land","Strength",
              "Attacking Casualties","Defensive Casualties"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",""), errors="coerce")
    # Empty slots
    df["Empty Slots Count"] = (
        df[RESOURCE_COLS]
          .apply(lambda row: sum(1 for v in row if pd.isna(v) or str(v).strip()==""), axis=1) // 2
    )
    # Inactivity
    if "Activity" in df:
        df["avg_inactivity"] = df["Activity"].map(ACTIVITY_MAP)
    return df


def aggregate(df):
    agg = df.groupby("snapshot_date").agg({
        "Nation ID":            "count",
        "Technology":           "sum",
        "Infrastructure":       "sum",
        "Base Land":            "sum",
        "Strength":             "sum",
        "Attacking Casualties": "sum",
        "Defensive Casualties": "sum",
        "Empty Slots Count":    "sum",
        "avg_inactivity":       "mean"
    }).rename(columns={"Nation ID": "nation_count"}).sort_index()
    return agg


def growth(series):
    first, last = series.iloc[0], series.iloc[-1]
    days = max((series.index[-1] - series.index[0]).days, 1)
    return (last - first) / days


def plot_series(series, title):
    plt.figure(figsize=(6,3))
    plt.plot(series.index, series.values, marker="o")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


def delete_last_message():
    if not Path(LAST_ID_FILE).exists():
        return
    msg_id = Path(LAST_ID_FILE).read_text().strip()
    if msg_id:
        try:
            requests.delete(f"{WEBHOOK_URL}/messages/{msg_id}")
        except:
            pass
    # clear
    Path(LAST_ID_FILE).write_text("")


def record_last_id(msg_id):
    Path(LAST_ID_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(LAST_ID_FILE).write_text(msg_id)


def post_new(content, charts, nation_csv):
    # delete old
    delete_last_message()
    # build embeds
    embeds = [{"image": {"url": f"attachment://{fn}"}} for fn in charts.keys()]
    payload = {"content": content, "embeds": embeds}
    files = []
    for idx, (fn, buf) in enumerate(charts.items()):
        files.append((f"file{idx}", (fn, buf, "image/png")))
    files.append((f"file{len(charts)}", ("nation_stats.csv", nation_csv, "text/csv")))
    # post
    resp = requests.post(WEBHOOK_URL + "?wait=true", data={"payload_json": json.dumps(payload)}, files=files)
    resp.raise_for_status()
    return resp.json()["id"]


def main():
    df0 = load_data()
    df = preprocess(df0)
    if df.empty:
        print(f"No data for {TARGET_ALLIANCE}")
        return

    agg = aggregate(df)
    latest = agg.iloc[-1]
    ts = latest.name.strftime("%Y-%m-%d %H:%M")

    # build summary
    lines = [f"**Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for m in METRICS:
        if m not in agg.columns:
            continue
        curr = latest[m]
        gr = growth(agg[m])
        lbl = LABEL_MAP[m]
        if pd.isna(curr):
            continue
        if isinstance(curr, float):
            lines.append(f"- Current {lbl}: {curr:.2f}")
        else:
            lines.append(f"- Current {lbl}: {int(curr):,}")
        lines.append(f"- {lbl} Growth/Day: {gr:.2f}")
    content = "\n".join(lines)

    # generate charts
    charts = {m.lower().replace(' ', '_') + '.png': plot_series(agg[m], f"{TARGET_ALLIANCE} — {LABEL_MAP[m]}")
              for m in METRICS if m in agg.columns}

    # nation CSV
    nation_df = pd.DataFrame([])  # placeholder
    buf = io.BytesIO()
    nation_df.to_csv(buf, index=False)
    buf.seek(0)

    new_id = post_new(content, charts, buf)
    record_last_id(new_id)

if __name__ == "__main__":
    main()
