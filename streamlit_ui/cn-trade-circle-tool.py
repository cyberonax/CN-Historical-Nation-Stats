import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
import re
from datetime import datetime
import altair as alt

st.set_page_config(layout="wide")

# -- Helpers ------------------------------------------------
def parse_date_from_filename(name):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", name)
    if not m:
        return None
    tok, z = m.groups()
    hour = 0 if z == '510001' else 12
    # split tok into m,d,y
    for md in [1,2]:
        for dd in [1,2]:
            if md+dd+4 == len(tok):
                try:
                    mon = int(tok[:md]); day = int(tok[md:md+dd]); year = int(tok[md+dd:])
                    if 1<=mon<=12 and 1<=day<=31:
                        return datetime(year, mon, day, hour)
                except:
                    pass
    return None

@st.cache_data(ttl=86400)
def load_data(folder='downloaded_zips'):
    zf_path = Path(folder)
    if not zf_path.exists():
        st.error(f"Folder '{folder}' not found.")
        return pd.DataFrame()
    dfs = []
    for zfile in zf_path.glob("CyberNations_SE_Nation_Stats_*.zip"):
        dt = parse_date_from_filename(zfile.name)
        if not dt:
            continue
        with zipfile.ZipFile(zfile) as z:
            files = z.namelist()
            if not files:
                continue
            with z.open(files[0]) as f:
                df = pd.read_csv(f, delimiter='|', encoding='ISO-8859-1', low_memory=False)
                df['date'] = dt
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# -- App -----------------------------------------------------
def main():
    st.title("Cyber Nations | Trade Circle Tool")

    df = load_data()
    if df.empty:
        return

    # Prepare
    df['Alliance'] = df['Alliance'].fillna("(No Alliance)")
    df['Ruler Name'] = df['Ruler Name'].fillna("Unknown")
    all_all = sorted(df['Alliance'].unique())

    sel = st.selectbox("Select Alliance", all_all)
    sub = df[df['Alliance'] == sel]

    # Show list of players
    players = sorted(sub['Ruler Name'].unique())
    st.markdown("**Players in Alliance:**")
    st.write(players)

    # Aggregate count over time
    agg = sub.groupby('date').agg(nations=('Nation ID','count')).reset_index()

    # Chart
    chart = alt.Chart(agg).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('nations:Q', title='Number of Nations')
    ).properties(width=800, height=400)

    st.markdown("**Nations Over Time**")
    st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()

