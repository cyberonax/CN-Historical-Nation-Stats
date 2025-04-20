#!/usr/bin/env python3
import os, re, zipfile, io
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt

# CONFIGURATION
SCRIPT_DIR = Path(__file__).parent.resolve()
LAST_ID_FILE    = SCRIPT_DIR / "last_msg_id.txt"
WEBHOOK_URL     = os.environ['DISCORD_WEBHOOK_URL']
ZIP_FOLDER      = "downloaded_zips"
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
        # % of nations with empty slots
    agg["%_empty"] = (
        empties.reindex(agg.index, fill_value=0) /
        total.reindex(agg.index, fill_value=0)
    ) * 100
