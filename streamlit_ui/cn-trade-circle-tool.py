import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
import re
from datetime import datetime
import altair as alt

st.set_page_config(layout="wide")

# -- Helper Functions ------------------------------------------------
def parse_date_from_filename(name):
    m = re.match(r"^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$", name)
    if not m:
        return None
    tok, z = m.groups()
    hour = 0 if z == '510001' else 12
    for md in [1, 2]:
        for dd in [1, 2]:
            if md + dd + 4 == len(tok):
                try:
                    mon = int(tok[:md]); day = int(tok[md:md+dd]); year = int(tok[md+dd:])
                    if 1 <= mon <= 12 and 1 <= day <= 31:
                        return datetime(year, mon, day, hour)
                except:
                    continue
    return None

@st.cache_data(ttl=24*60*60)
def load_data(folder: str = 'downloaded_zips') -> pd.DataFrame:
    """
    Load and parse all CyberNations zip snapshots into a single DataFrame.
    """
    zip_path = Path(folder)
    if not zip_path.exists():
        st.error(f"Folder '{folder}' not found. Ensure your zips are in place.")
        return pd.DataFrame()

    frames = []
    for file in zip_path.glob("CyberNations_SE_Nation_Stats_*.zip"):
        date = parse_date_from_filename(file.name)
        if not date:
            continue
        with zipfile.ZipFile(file, 'r') as z:
            names = z.namelist()
            if not names:
                continue
            with z.open(names[0]) as f:
                df = pd.read_csv(f, delimiter='|', encoding='ISO-8859-1', low_memory=False)
                df['date'] = date
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# -- Main App --------------------------------------------------------
def main():
    st.title("Cyber Nations | Trade Circle Tool")
    
    # Load data
    df = load_data()
    if df.empty:
        return

    # Clean and prepare
    df['Alliance'] = df['Alliance'].fillna("(No Alliance)")
    df['Ruler Name'] = df['Ruler Name'].fillna("Unknown")

    # Alliance selector
    alliances = sorted(df['Alliance'].unique())
    selected = st.selectbox("Select Alliance", alliances)
    subset = df[df['Alliance'] == selected].copy()
    subset = subset.sort_values('date')

    # Collapsible: Raw Alliance Data Table
    with st.expander("Raw Alliance Data", expanded=False):
        st.dataframe(subset)

    # Collapsible: Player List
    with st.expander("Players in Alliance", expanded=True):
        players = sorted(subset['Ruler Name'].unique())
        st.write(players)

    # Collapsible: Nations Over Time Chart
    with st.expander("Nations Over Time", expanded=True):
        # Aggregate count by snapshot date
        agg = (
            subset
            .groupby('date')
            .agg(nations=('Nation ID', 'count'))
            .reset_index()
        )

        # Interactive Altair line + points chart
        date_sel = alt.selection_single(on='mouseover', fields=['date'], nearest=True, empty='none')
        base = alt.Chart(agg).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('nations:Q', title='Number of Nations'),
            tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('nations:Q', title='Count')]
        )
        line = base.mark_line()
        points = base.mark_point().encode(
            opacity=alt.condition(date_sel, alt.value(1), alt.value(0))
        ).add_selection(date_sel)
        chart = (line + points).properties(width=800, height=400).interactive()
        st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()
