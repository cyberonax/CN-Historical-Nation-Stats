#!/usr/bin/env python3
import os, re, zipfile, io, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
WEBHOOK_URL = os.environ['https://discord.com/api/webhooks/1363298225116811486/XyBzniB7TTwz2Tn1k0SHyMK2H3j5MI626aO3ohBd-KTseuhPlXq0EtwXfA40T-8T_Ckr']
ZIP_FOLDER  = "downloaded_zips"
CACHE_FILE  = ".github/scripts/last_msg_id.txt"
# ──────────────────────────────────────────────────

def parse_date_from_filename(fn: str) -> datetime | None:
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", fn)
    if not m: return None
    token, zipid = m.groups()
    hour = 0 if zipid == "510001" else 12
    # split token M/D/YYYY
    for md in (1,2):
        for dd in (1,2):
            if md+dd+4 == len(token):
                try:
                    month = int(token[:md])
                    day   = int(token[md:md+dd])
                    year  = int(token[md+dd:])
                    return datetime(year, month, day, hour)
                except ValueError:
                    continue
    return None

def load_all_data() -> pd.DataFrame:
    dfs = []
    for zp in Path(ZIP_FOLDER).glob("CyberNations_SE_Nation_Stats_*.zip"):
        dt = parse_date_from_filename(zp.name)
        if not dt: continue
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
            df["snapshot_date"] = dt
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def aggregate_strength(df: pd.DataFrame) -> pd.Series:
    df["Strength"] = pd.to_numeric(df["Strength"].astype(str).str.replace(",",""), errors="coerce")
    latest = df[df.snapshot_date == df.snapshot_date.max()]
    return latest.groupby("Alliance")["Strength"].sum().sort_values(ascending=False)

def make_strength_chart(series: pd.Series) -> io.BytesIO:
    plt.figure(figsize=(8,4))
    series.plot(marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Alliance Strength as of {series.name:%Y-%m-%d %H:%M}")
    plt.ylabel("Strength")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

def load_last_msg_id() -> str | None:
    try:
        return Path(CACHE_FILE).read_text().strip()
    except FileNotFoundError:
        return None

def save_last_msg_id(msg_id: str):
    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE_FILE).write_text(msg_id)

def send_or_edit(content: str, chart_buf: io.BytesIO) -> str:
    """
    If we have a previous message_id, PATCH to edit it;
    otherwise POST a new one with `?wait=true` to retrieve its id.
    """
    payload = {
        "content": content,
        "embeds": [
            {"image": {"url": "attachment://chart.png"}}
        ]
    }
    files = {
        "file": ("chart.png", chart_buf, "image/png")
    }

    last_id = load_last_msg_id()
    if last_id:
        url = f"{WEBHOOK_URL}/messages/{last_id}"
        r = requests.patch(url, data={"payload_json": json.dumps(payload)}, files=files)
    else:
        url = WEBHOOK_URL + "?wait=true"
        r = requests.post(url, data={"payload_json": json.dumps(payload)}, files=files)

    r.raise_for_status()
    msg = r.json()
    return msg["id"]

def main():
    df = load_all_data()
    if df.empty:
        return

    strength = aggregate_strength(df)
    # name the index so chart title can pick it up
    strength.name = df.snapshot_date.max()
    chart = make_strength_chart(strength)

    lines = [f"**Top 5 Alliances by Strength (snapshot {strength.name:%Y-%m-%d %H:%M})**"]
    for name, val in strength.head(5).items():
        lines.append(f"- {name}: {int(val):,}")
    text = "\n".join(lines)

    msg_id = send_or_edit(text, chart)
    save_last_msg_id(msg_id)

if __name__ == "__main__":
    main()
