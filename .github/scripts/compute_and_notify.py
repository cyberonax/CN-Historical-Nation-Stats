#!/usr/bin/env python3
import os, re, zipfile, io, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────
WEBHOOK_URL     = os.environ['DISCORD_WEBHOOK_URL']
ZIP_FOLDER      = "downloaded_zips"
CACHE_FILE      = ".github/scripts/last_msg_id.txt"
TARGET_ALLIANCE = "Freehold of The Wolves"

# Map raw column→friendly label
LABEL_MAP = {
    "nation_count":        "Nation Count",
    "Empty Slots Count":   "Total Empty Trade Slots",
    "avg_inactivity":      "Avg Inactivity (Days)",
    "Technology":          "Total Technology",
    "Infrastructure":      "Total Infrastructure",
    "Base Land":           "Total Base Land",
    "Strength":            "Total Strength",
    "Attacking Casualties":"Total Attacking Casualties",
    "Defensive Casualties":"Total Defensive Casualties"
}
# The exact sequence of metrics we’ll chart & summarize
METRICS = [
    "nation_count",
    "Empty Slots Count",
    "avg_inactivity",
    "Technology",
    "Infrastructure",
    "Base Land",
    "Strength",
    "Attacking Casualties",
    "Defensive Casualties"
]
# Activity mapping
ACTIVITY_MAP = {
    "Active In The Last 3 Days":        3,
    "Active This Week":                  7,
    "Active Last Week":                 14,
    "Active Three Weeks Ago":           21,
    "Active More Than Three Weeks Ago": 28
}
RESOURCE_COLS = [f"Connected Resource {i}" for i in range(1, 11)]
# ──────────────────────────────────────────────────

def parse_date(fn):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", fn)
    if not m: 
        return None
    token, zipid = m.groups()
    hour = 0 if zipid=="510001" else 12
    # try M/D/YYYY splits
    for md in (1,2):
        for dd in (1,2):
            if md+dd+4 == len(token):
                try:
                    month=int(token[:md])
                    day  =int(token[md:md+dd])
                    year =int(token[md+dd:])
                    return datetime(year, month, day, hour)
                except ValueError:
                    pass
    return None

def load_data():
    dfs=[]
    for zp in Path(ZIP_FOLDER).glob("*.zip"):
        dt = parse_date(zp.name)
        if not dt: continue
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
            df["snapshot_date"] = pd.to_datetime(dt)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def preprocess(df):
    df = df[df["Alliance"]==TARGET_ALLIANCE].copy()
    if df.empty:
        return df
    # Numeric cols
    for c in ["Technology","Infrastructure","Base Land","Strength",
              "Attacking Casualties","Defensive Casualties"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",""), errors="coerce")
    # Nation count computed later
    # Empty slots
    df["Empty Slots Count"] = (
        df[RESOURCE_COLS]
          .apply(lambda row: sum(1 for v in row if pd.isna(v) or str(v).strip()==""), axis=1)
          // 2
    )
    # Activity → inactivity days
    if "Activity" in df:
        df["activity_score"] = df["Activity"].map(ACTIVITY_MAP)
    return df

def aggregate(df):
    agg = df.groupby("snapshot_date").agg({
        "Nation ID":             "count",
        "Empty Slots Count":     "sum",
        "activity_score":        "mean",
        "Technology":            "sum",
        "Infrastructure":        "sum",
        "Base Land":             "sum",
        "Strength":              "sum",
        "Attacking Casualties":  "sum",
        "Defensive Casualties":  "sum"
    }).rename(columns={
        "Nation ID":      "nation_count",
        "activity_score": "avg_inactivity"
    }).sort_index()
    return agg

def growth(series: pd.Series) -> float:
    first, last = series.iloc[0], series.iloc[-1]
    days = max((series.index[-1] - series.index[0]).days, 1)
    return (last - first) / days

def plot_series(series: pd.Series, title: str, fname: str) -> io.BytesIO:
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

def load_msg_id() -> str | None:
    try:
        return Path(CACHE_FILE).read_text().strip()
    except FileNotFoundError:
        return None

def save_msg_id(mid: str):
    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE_FILE).write_text(mid)

def send_or_edit(content: str, buffers: dict[str, io.BytesIO]) -> str:
    # Build embeds array for each image
    embeds = [{"image": {"url": f"attachment://{fn}"}} for fn in buffers.keys()]
    payload = {"content": content, "embeds": embeds}

    # Prepare files list
    files = [
        ("file", (fn, buf, "image/png"))
        for fn, buf in buffers.items()
    ]

    last_id = load_msg_id()
    if last_id:
        url = f"{WEBHOOK_URL}/messages/{last_id}"
        r = requests.patch(url, data={"payload_json": json.dumps(payload)}, files=files)
    else:
        url = WEBHOOK_URL + "?wait=true"
        r = requests.post(url, data={"payload_json": json.dumps(payload)}, files=files)

    r.raise_for_status()
    return r.json()["id"]

def main():
    df_raw = load_data()
    df     = preprocess(df_raw)
    if df.empty:
        print(f"No data for {TARGET_ALLIANCE}")
        return

    agg = aggregate(df)
    latest = agg.iloc[-1]
    # Build summary
    snapshot = latest.name.strftime("%Y-%m-%d %H:%M")
    lines = [f"**Stats for {TARGET_ALLIANCE} (snapshot {snapshot})**"]
    for m in METRICS:
        if m not in agg.columns:
            continue
        label = LABEL_MAP[m]
        curr  = latest[m]
        gr    = growth(agg[m])
        # Format
        if pd.isna(curr):
            continue
        if isinstance(curr, float):
            lines.append(f"- Current {label}: {curr:.2f}")
        else:
            lines.append(f"- Current {label}: {int(curr):,}")
        lines.append(f"- {label} Growth/Day: {gr:.2f}")
    content = "\n".join(lines)

    # Generate one chart per metric
    buffers = {}
    for m in METRICS:
        if m in agg.columns:
            fname = m.lower().replace(" ", "_") + ".png"
            title = f"{TARGET_ALLIANCE} — {LABEL_MAP[m]}"
            buf   = plot_series(agg[m], title, fname)
            buffers[fname] = buf

    # Send or edit
    msg_id = send_or_edit(content, buffers)
    save_msg_id(msg_id)

if __name__ == "__main__":
    main()
