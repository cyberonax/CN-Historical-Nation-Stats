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

# Order we want in the summary
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
    if not m: return None
    token, zipid = m.groups()
    hour = 0 if zipid=="510001" else 12
    for md in (1,2):
        for dd in (1,2):
            if md+dd+4 == len(token):
                try:
                    month=int(token[:md]); day=int(token[md:md+dd])
                    year=int(token[md+dd:]); return datetime(year,month,day,hour)
                except: pass
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
    if df.empty: return df
    # numeric
    for c in ["Technology","Infrastructure","Base Land","Strength",
              "Attacking Casualties","Defensive Casualties"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",""), errors="coerce")
    # empty slots
    df["Empty Slots Count"] = (
        df[RESOURCE_COLS]
          .apply(lambda row: sum(1 for v in row if pd.isna(v) or str(v).strip()==""), axis=1)//2
    )
    # inactivity
    if "Activity" in df:
        df["avg_inactivity"] = df["Activity"].map(ACTIVITY_MAP)
    return df

def aggregate(df):
    agg = df.groupby("snapshot_date").agg({
        "Nation ID":             "count",
        "Technology":            "sum",
        "Infrastructure":        "sum",
        "Base Land":             "sum",
        "Strength":              "sum",
        "Attacking Casualties":  "sum",
        "Defensive Casualties":  "sum",
        "Empty Slots Count":     "sum",
        "avg_inactivity":        "mean"
    }).rename(columns={"Nation ID":"nation_count"}).sort_index()
    return agg

def growth(s: pd.Series) -> float:
    first, last = s.iloc[0], s.iloc[-1]
    days = max((s.index[-1]-s.index[0]).days,1)
    return (last-first)/days

def plot_series(s: pd.Series, title: str) -> io.BytesIO:
    plt.figure(figsize=(6,3))
    plt.plot(s.index, s.values, marker="o")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf,format="png"); buf.seek(0); plt.close()
    return buf

def compute_nation_stats(df):
    # For each Nation ID & Ruler Name: current + growth/day for each metric
    metrics = ["Technology","Infrastructure","Base Land","Strength",
               "Attacking Casualties","Defensive Casualties",
               "Empty Slots Count","avg_inactivity"]
    rows=[]
    for (nid, ruler), grp in df.groupby(["Nation ID","Ruler Name"]):
        grp = grp.sort_values("snapshot_date")
        first = grp.iloc[0]; last = grp.iloc[-1]
        days  = max((last.snapshot_date-first.snapshot_date).days,1)
        row = {
            "Nation ID": nid,
            "Ruler Name": ruler
        }
        for m in metrics:
            if m in grp.columns:
                fv = first[m] or 0; lv = last[m] or 0
                row[f"{m} (current)"] = lv
                row[f"{m} Growth/Day"] = (lv-fv)/days
        rows.append(row)
    return pd.DataFrame(rows)

def load_msg_id():
    try: return Path(CACHE_FILE).read_text().strip()
    except: return None

def save_msg_id(mid):
    Path(CACHE_FILE).parent.mkdir(parents=True,exist_ok=True)
    Path(CACHE_FILE).write_text(mid)

def send_or_edit(content: str, buffers: dict[str, io.BytesIO]) -> str:
    # Build one embed per chart, referencing the attachment://filename
    embeds = [{"image": {"url": f"attachment://{fn}"}} for fn in buffers.keys()]
    payload = {"content": content, "embeds": embeds}

    # Now attach each chart under a unique field name: file0, file1, ...
    files = []
    for idx, (fn, buf) in enumerate(buffers.items()):
        field_name = f"file{idx}"
        files.append((field_name, (fn, buf, "image/png")))

    last_id = load_msg_id()
    if last_id:
        url = f"{WEBHOOK_URL}/messages/{last_id}"
        r = requests.patch(
            url,
            data={"payload_json": json.dumps(payload)},
            files=files
        )
    else:
        url = WEBHOOK_URL + "?wait=true"
        r = requests.post(
            url,
            data={"payload_json": json.dumps(payload)},
            files=files
        )

    r.raise_for_status()
    return r.json()["id"]

def main():
    df0 = load_data()
    df  = preprocess(df0)
    if df.empty:
        print("No data for",TARGET_ALLIANCE); return

    agg = aggregate(df)
    latest = agg.iloc[-1]

    # build summary with custom order
    ts = latest.name.strftime("%Y-%m-%d %H:%M")
    lines=[f"**Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for m in METRICS:
        if m not in agg.columns: continue
        curr = latest[m]; gr = growth(agg[m])
        label=LABEL_MAP[m]
        if pd.isna(curr): continue
        if isinstance(curr,float):
            lines.append(f"- Current {label}: {curr:.2f}")
        else:
            lines.append(f"- Current {label}: {int(curr):,}")
        lines.append(f"- {label} Growth/Day: {gr:.2f}")
    content="\n".join(lines)

    # generate charts
    charts={}
    for m in METRICS:
        if m in agg.columns:
            fname=m.lower().replace(" ","_")+".png"
            title=f"{TARGET_ALLIANCE} — {LABEL_MAP[m]}"
            charts[fname]=plot_series(agg[m],title)

    # compute and serialize nation-level stats
    nd = compute_nation_stats(df)
    buf=io.BytesIO()
    nd.to_csv(buf,index=False)
    buf.seek(0)

    # send/edit
    mid = send_or_edit(content, charts, buf)
    save_msg_id(mid)

if __name__=="__main__":
    main()
