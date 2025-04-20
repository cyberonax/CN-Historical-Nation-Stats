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

# Friendly labels
LABEL_MAP = {
    "nation_count":               "Nation Count",
    "avg_inactivity":             "Avg Inactivity (Days)",
    "Empty Slots Count":          "Total Empty Trade Slots",
    "%_empty":                    "% of Nations w/ Empty Slots",
    "Technology":                 "Total Technology",
    "Infrastructure":             "Total Infrastructure",
    "Base Land":                  "Total Base Land",
    "Strength":                   "Total Strength",
    "Attacking Casualties":       "Total Attacking Casualties",
    "Defensive Casualties":       "Total Defensive Casualties"
}

# FUNCTIONS

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
                    m_, d_, y_ = int(token[:md]), int(token[md:md+dd]), int(token[md+dd:])
                    return datetime(y_, m_, d_, hour)
                except:
                    pass
    return None


def load_data():
    dfs = []
    for zp in Path(ZIP_FOLDER).glob("*.zip"):
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
    # ensure numeric
    num_cols = ["Technology","Infrastructure","Base Land","Strength",
                "Attacking Casualties","Defensive Casualties"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",""), errors="coerce")
    # empty slots
    df["Empty Slots Count"] = df[[f"Connected Resource {i}" for i in range(1,11)]].apply(
        lambda r: sum(1 for v in r if pd.isna(v) or str(v).strip()==""), axis=1
    ) // 2
    # inactivity
    if "Activity" in df:
        mapping = {"Active In The Last 3 Days":3,"Active This Week":7,
                   "Active Last Week":14,"Active Three Weeks Ago":21,
                   "Active More Than Three Weeks Ago":28}
        df["avg_inactivity"] = df["Activity"].map(mapping)
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
        Empty_Slots=("Empty Slots Count","sum"),
        avg_inactivity=("avg_inactivity","mean")
    ).sort_index()
    # compute averages per nation
    for col in ["Technology","Infrastructure","Base_Land","Strength",
                "Attacking_Casualties","Defensive_Casualties"]:
        agg[f"avg_{col}"] = agg[col] / agg["nation_count"]
    # percent empty
    total = df.groupby("snapshot_date")["Nation ID"].count()
    empties = df[df["Empty Slots Count"]>0].groupby("snapshot_date")["Nation ID"].count()
    agg["%_empty"] = (empties.reindex(agg.index, fill_value=0) / total.reindex(agg.index))*100
    return agg


def growth(s):
    first, last = s.iloc[0], s.iloc[-1]
    days = max((s.index[-1]-s.index[0]).days,1)
    return (last-first)/days


def plot_series(series, title, fname):
    plt.figure(figsize=(6,3))
    plt.plot(series.index, series.values, marker="o")
    plt.title(title)
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf,format="png"); buf.seek(0); plt.close()
    return fname, buf


def plot_multiseries(d, title, fname):
    plt.figure(figsize=(8,4))
    for lbl, ser in d.items():
        plt.plot(ser.index, ser.values, marker="o", label=lbl)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf,format="png"); buf.seek(0); plt.close()
    return fname, buf


def delete_last():
    if Path(LAST_ID_FILE).exists():
        mid = Path(LAST_ID_FILE).read_text().strip()
        if mid:
            try: requests.delete(f"{WEBHOOK_URL}/messages/{mid}")
            except: pass
        Path(LAST_ID_FILE).write_text("")


def record_last(mid):
    Path(LAST_ID_FILE).parent.mkdir(parents=True,exist_ok=True)
    Path(LAST_ID_FILE).write_text(mid)


def post_new(content, attachments):
    delete_last()
    embeds=[{"image":{"url":f"attachment://{fn}"}} for fn,_ in attachments]
    payload={"content":content,"embeds":embeds}
    files=[(f"file{i}",(fn,buf,"image/png")) for i,(fn,buf) in enumerate(attachments)]
    resp = requests.post(WEBHOOK_URL+"?wait=true",data={"payload_json":json.dumps(payload)},files=files)
    resp.raise_for_status()
    return resp.json()["id"]


def main():
    df0=load_data(); df=preprocess(df0)
    if df.empty:
        print("No data"); return
    agg=aggregate(df)
    latest=agg.iloc[-1]
    ts=latest.name.strftime("%Y-%m-%d %H:%M")
    # build summary
    parts=[f"**Aggregated Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for k in ["nation_count","avg_inactivity","Empty_Slots","%_empty"]:
        if k in agg.columns:
            val=latest[k]; g=growth(agg[k])
            lbl=LABEL_MAP.get(k,k)
            parts.append(f"- Current {lbl}: {val:.2f}" if isinstance(val,float) else f"- Current {lbl}: {int(val):,}")
            parts.append(f"- {lbl} Growth/Day: {g:.2f}")
    content="\n".join(parts)
    # attachments: charts
    atts=[]
    # single metric charts
    for key in ["nation_count","%_empty","avg_inactivity"]:
        fname=f"{key}.png"
        atts.append(plot_series(agg[key],f"{LABEL_MAP[key]} over time",fname))
    # multi-series charts
    atts.append(plot_multiseries({
        "Tech":agg["Technology"],"Infra":agg["Infrastructure"],
        "Base Land":agg["Base_Land"],"Strength":agg["Strength"]
    },"Totals over time","totals.png"))
    atts.append(plot_multiseries({
        "Attacking":agg["Attacking_Casualties"],"Defensive":agg["Defensive_Casualties"]
    },"Casualties over time","casualties.png"))
    # post
    msg_id=post_new(content,atts)
    record_last(msg_id)

if __name__=="__main__": main()
