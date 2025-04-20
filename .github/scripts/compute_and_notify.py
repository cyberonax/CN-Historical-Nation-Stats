#!/usr/bin/env python3
import os, re, zipfile, io, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt

# CONFIGURATION
WEBHOOK_URL      = os.environ['DISCORD_WEBHOOK_URL']
ZIP_FOLDER       = "downloaded_zips"
MSG_IDS_FILE     = ".github/scripts/msg_ids.txt"
TARGET_ALLIANCE  = "Freehold of The Wolves"

# Metric labels (exact phrasing)
LABEL_MAP = {
    "nation_count":               "Nation Count Over Time",
    "avg_inactivity":             "Average Alliance Inactivity Over Time (Days)",
    "empty_slots_total":          "Total Empty Trade Slots Over Time",
    "percent_empty":              "% of Nations with Empty Trade Slots Over Time",
    "total_technology":           "Total Technology Over Time",
    "avg_technology":             "Average Technology Over Time",
    "total_infrastructure":       "Total Infrastructure Over Time",
    "avg_infrastructure":         "Average Infrastructure Over Time",
    "total_base_land":            "Total Base Land Over Time",
    "avg_base_land":              "Average Base Land Over Time",
    "total_strength":             "Total Strength Over Time",
    "avg_strength":               "Average Strength Over Time",
    "total_attacking_casualties": "Total Attacking Casualties Over Time",
    "avg_attacking_casualties":   "Average Attacking Casualties Over Time",
    "total_defensive_casualties": "Total Defensive Casualties Over Time",
    "avg_defensive_casualties":   "Average Defensive Casualties Over Time"
}

# Exact rendering order
METRICS = [
    "nation_count",
    "avg_inactivity",
    "empty_slots_total",
    "percent_empty",
    "total_technology",
    "avg_technology",
    "total_infrastructure",
    "avg_infrastructure",
    "total_base_land",
    "avg_base_land",
    "total_strength",
    "avg_strength",
    "total_attacking_casualties",
    "avg_attacking_casualties",
    "total_defensive_casualties",
    "avg_defensive_casualties"
]

# Resource & activity map
RESOURCE_COLS = [f"Connected Resource {i}" for i in range(1, 11)]
ACTIVITY_MAP  = {
    "Active In The Last 3 Days": 3,
    "Active This Week":           7,
    "Active Last Week":          14,
    "Active Three Weeks Ago":    21,
    "Active More Than Three Weeks Ago": 28
}

# 1) Load and parse ZIP data

def parse_date(fn):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", fn)
    if not m: return None
    token, zipid = m.groups()
    hour = 0 if zipid=="510001" else 12
    for md in (1,2):
        for dd in (1,2):
            if md+dd+4==len(token):
                try:
                    mm = int(token[:md]); dd_ = int(token[md:md+dd]); yyyy = int(token[md+dd:])
                    return datetime(yyyy, mm, dd_, hour)
                except: pass
    return None


def load_data():
    dfs=[]
    for zp in Path(ZIP_FOLDER).glob("CyberNations_SE_Nation_Stats_*.zip"):
        dt=parse_date(zp.name)
        if not dt: continue
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
            df=pd.read_csv(f,delimiter="|",encoding="ISO-8859-1",low_memory=False)
            df["snapshot_date"]=pd.to_datetime(dt)
            dfs.append(df)
    return pd.concat(dfs,ignore_index=True) if dfs else pd.DataFrame()

# 2) Preprocess

def preprocess(df):
    df=df[df.get("Alliance")==TARGET_ALLIANCE].copy()
    # numeric metrics
    for col in ["Technology","Infrastructure","Base Land","Strength",
                "Attacking Casualties","Defensive Casualties"]:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col].astype(str).str.replace(",",""),errors="coerce")
    # empty slots per nation
    df['empty_slots_total']=df[RESOURCE_COLS].apply(
        lambda r: sum(1 for v in r if pd.isna(v) or str(v).strip()==""), axis=1
    )//2
    # inactivity score
    if 'Activity' in df.columns:
        df['avg_inactivity']=df['Activity'].map(ACTIVITY_MAP)
    return df

# 3) Aggregate by snapshot

def aggregate(df):
    agg=df.groupby('snapshot_date').agg(
        nation_count=('Nation ID','count'),
        total_technology=('Technology','sum'),
        total_infrastructure=('Infrastructure','sum'),
        total_base_land=('Base Land','sum'),
        total_strength=('Strength','sum'),
        total_attacking_casualties=('Attacking Casualties','sum'),
        total_defensive_casualties=('Defensive Casualties','sum'),
        empty_slots_total=('empty_slots_total','sum'),
        avg_inactivity=('avg_inactivity','mean')
    ).sort_index()
    # per-nation averages
    for col in ['total_technology','total_infrastructure','total_base_land','total_strength',
                'total_attacking_casualties','total_defensive_casualties']:
        agg[f'avg_{col}']=agg[col]/agg['nation_count']
    # percent empty
    total_n= df.groupby('snapshot_date')['Nation ID'].count()
    empty_n= df[df['empty_slots_total']>0].groupby('snapshot_date')['Nation ID'].count()
    agg['percent_empty']=empty_n.reindex(agg.index,fill_value=0)/total_n.reindex(agg.index)*100
    return agg

# 4) Growth helper

def growth(s):
    first,last=s.iloc[0],s.iloc[-1]
    days=max((s.index[-1]-s.index[0]).days,1)
    return (last-first)/days

# 5) Plot one metric

def plot_series(s, title, filename):
    plt.figure(figsize=(10,6))
    plt.plot(s.index,s.values,marker='o',linewidth=2)
    plt.title(title,fontsize=16)
    plt.xticks(rotation=45,ha='right')
    plt.tight_layout()
    buf=io.BytesIO(); plt.savefig(buf,format='png'); buf.seek(0); plt.close()
    return filename,buf

# 6) Delete old Discord messages

def delete_old():
    if Path(MSG_IDS_FILE).exists():
        mids=Path(MSG_IDS_FILE).read_text().splitlines()
        for mid in mids:
            try: requests.delete(f"{WEBHOOK_URL}/messages/{mid}")
            except: pass
        Path(MSG_IDS_FILE).unlink()

# 7) Record new message IDs

def record_ids(mids):
    Path(MSG_IDS_FILE).parent.mkdir(parents=True,exist_ok=True)
    Path(MSG_IDS_FILE).write_text("\n".join(mids))

# 8) Post in chunks (<=10 per message)

def post_in_chunks(content,attachments,chunk_size=10):
    delete_old()
    msg_ids=[]
    for i in range(0,len(attachments),chunk_size):
        batch=attachments[i:i+chunk_size]
        embeds=[{'image':{'url':f'attachment://{fn}'}} for fn,_ in batch]
        payload={'content':content,'embeds':embeds}
        files=[(f'file{j}',(fn,buf,'image/png')) for j,(fn,buf) in enumerate(batch)]
        resp=requests.post(
            WEBHOOK_URL+'?wait=true',
            data={'payload_json':json.dumps(payload)},
            files=files
        )
        resp.raise_for_status()
        msg_ids.append(resp.json()['id'])
        content=''  # only include text once
    record_ids(msg_ids)

# MAIN

def main():
    df0=load_data(); df=preprocess(df0)
    if df.empty:
        print(f'No data for {TARGET_ALLIANCE}'); return
    agg=aggregate(df)
    last=agg.iloc[-1]; ts=last.name.strftime('%Y-%m-%d %H:%M')
    # build summary
    lines=[f"**Aggregated Stats for {TARGET_ALLIANCE} (snapshot {ts})**"]
    for m in METRICS:
        if m not in agg.columns: continue
        v=last[m]; g=growth(agg[m]); lbl=LABEL_MAP[m]
        disp=f"{v:.2f}" if isinstance(v,float) else f"{int(v):,}"
        lines.append(f"- Current {lbl}: {disp}")
        lines.append(f"- {lbl} Growth/Day: {g:.2f}")
    content='\n'.join(lines)
    # attachments
    attachments=[]
    for m in METRICS:
        if m in agg.columns:
            fname=f"{m}.png"; title=LABEL_MAP[m]
            attachments.append(plot_series(agg[m],title,fname))
    post_in_chunks(content,attachments)

if __name__=='__main__': main()
