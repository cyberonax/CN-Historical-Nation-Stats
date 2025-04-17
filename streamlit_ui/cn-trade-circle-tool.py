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

    # Map activity to days of inactivity
    if 'Activity' in df.columns:
        activity_mapping = {
            "Active In The Last 3 Days": 3,
            "Active This Week": 7,
            "Active Last Week": 14,
            "Active Three Weeks Ago": 21,
            "Active More Than Three Weeks Ago": 28
        }
        df['activity_score'] = df['Activity'].map(activity_mapping)

    # Alliance selector with default
    alliances = sorted(df['Alliance'].unique())
    default_idx = alliances.index("Freehold of The Wolves") if "Freehold of The Wolves" in alliances else 0
    selected = st.selectbox("Select Alliance", alliances, index=default_idx)
    subset = df[df['Alliance'] == selected].copy()
    subset = subset.sort_values('date')

    # Collapsible: Raw Alliance Data Table
    with st.expander("Raw Alliance Data", expanded=False):
        st.dataframe(subset)

    # Collapsible: Players & Inactivity
    with st.expander("Players in Alliance", expanded=True):
        players = sorted(subset['Ruler Name'].unique())
        st.markdown("**Players:**")
        st.write(players)

        if 'activity_score' in subset.columns:
            avg_inact = (
                subset.dropna(subset=['activity_score'])
                .groupby('Ruler Name')['activity_score']
                .mean()
                .reset_index()
                .rename(columns={'activity_score': 'All Time Average Days of Inactivity'})
            )
            st.markdown("**All Time Average Days of Inactivity:**")
            st.dataframe(avg_inact)

    # Collapsible: Nation Inactivity Over Time Chart
    with st.expander("Nation Inactivity Over Time (Days)", expanded=True):
        if 'activity_score' in subset.columns:
            # Compute average by snapshot date
            agg = (
                subset.dropna(subset=['activity_score'])
                .groupby('date')
                .agg(avg_inactivity=('activity_score', 'mean'))
                .reset_index()
            )

            # Interactive Altair line + points chart
            date_sel = alt.selection_single(on='mouseover', fields=['date'], nearest=True, empty='none')
            base = alt.Chart(agg).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('avg_inactivity:Q', title='Days of Inactivity'),
                tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('avg_inactivity:Q', title='Avg Inactivity')]
            )
            line = base.mark_line()
            points = base.mark_point().encode(
                opacity=alt.condition(date_sel, alt.value(1), alt.value(0))
            ).add_selection(date_sel)
            chart = (line + points).properties(width=800, height=400).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No Activity data available for this alliance.")

if __name__ == '__main__':
    main()
