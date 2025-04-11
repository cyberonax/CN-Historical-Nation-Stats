import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
import re
from datetime import datetime

##############################
# HELPER FUNCTIONS
##############################

def parse_date_from_filename(filename):
    """
    Extract and parse the date from a filename that follows the pattern:
    "CyberNations_SE_Nation_Stats_<dateToken><zipid>.zip"
    
    The dateToken is a concatenation of month, day, and year.
    For example, from "CyberNations_SE_Nation_Stats_452025510002.zip":
      date_token = "452025" is interpreted as month=4, day=5, year=2025.
    Returns a datetime object on success, otherwise None.
    """
    pattern = r'^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$'
    match = re.match(pattern, filename)
    if not match:
        return None
    date_token = match.group(1)
    for m_digits in [1, 2]:
        for d_digits in [1, 2]:
            if m_digits + d_digits + 4 == len(date_token):
                try:
                    month = int(date_token[:m_digits])
                    day = int(date_token[m_digits:m_digits+d_digits])
                    year = int(date_token[m_digits+d_digits:m_digits+d_digits+4])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month, day)
                except Exception:
                    continue
    return None

@st.cache_data(show_spinner="Loading historical data...")
def load_data():
    """
    Loads data from all zip files in the "downloaded_zips" folder.
    Each zip file is expected to contain a CSV file with a '|' delimiter and
    encoding 'ISO-8859-1'. A new column 'snapshot_date' is added based on the filename.
    """
    data_frames = []
    zip_folder = Path("downloaded_zips")
    if not zip_folder.exists():
        st.warning("Folder 'downloaded_zips' not found. Please ensure zip files are available.")
        return pd.DataFrame()
    
    for zip_file in zip_folder.glob("CyberNations_SE_Nation_Stats_*.zip"):
        snapshot_date = parse_date_from_filename(zip_file.name)
        if snapshot_date is None:
            st.warning(f"Could not parse date from filename: {zip_file.name}")
            continue
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                file_list = z.namelist()
                if not file_list:
                    st.warning(f"No files found in zip: {zip_file.name}")
                    continue
                with z.open(file_list[0]) as f:
                    df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
                    df['snapshot_date'] = snapshot_date
                    data_frames.append(df)
        except Exception as e:
            st.error(f"Error processing {zip_file.name}: {e}")
    
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame()

def aggregate_by_alliance(df):
    """
    Aggregates nation stats by snapshot_date and Alliance.
    
    For each group (by snapshot_date and Alliance):
      - The number of nations is computed via count (as 'nation_count').
      - Other metrics (Empty Trade Slots, Attacking Casualties, Defensive Casualties,
        Infrastructure, Technology, Base Land, and Strength) are summed.
      
    In this aggregated DataFrame, the summed 'Strength' represents the Total Strength.
    """
    # Use the column names as they appear in the raw data.
    numeric_cols = ['Empty Trade Slots', 'Attacking Casualties', 'Defensive Casualties',
                    'Infrastructure', 'Technology', 'Base Land', 'Strength']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    agg_dict = {
        'Nation ID': 'count',  # This will be renamed to nation_count.
        'Empty Trade Slots': 'sum',
        'Attacking Casualties': 'sum',
        'Defensive Casualties': 'sum',
        'Infrastructure': 'sum',
        'Technology': 'sum',
        'Base Land': 'sum',
        'Strength': 'sum'
    }
    
    grouped = df.groupby(['snapshot_date', 'Alliance']).agg(agg_dict).reset_index()
    grouped.rename(columns={'Nation ID': 'nation_count'}, inplace=True)
    return grouped

def aggregate_totals(df, col):
    """
    Groups the raw data by snapshot_date and Alliance and returns a pivoted DataFrame 
    with the SUM for the specified column.
    
    For example, for 'Strength', this sums the strength of each nation to give Total Strength.
    """
    grouped = df.groupby(['snapshot_date', 'Alliance'])[col].sum().reset_index()
    grouped['date'] = grouped['snapshot_date'].dt.date
    return grouped.pivot(index='date', columns='Alliance', values=col)

##############################
# STREAMLIT APP
##############################

def main():
    st.title("Cyber Nations | Nation Stats Timeline Tracker")
    st.markdown("""
        This dashboard displays a time-based stream of nation statistics grouped by alliance.
    """)
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data loaded. Please check the 'downloaded_zips' folder.")
        return
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Sidebar Filters: Create a dropdown with available alliances.
    st.sidebar.header("Filters")
    alliances = sorted(df['Alliance'].dropna().unique())
    # Set default to "Freehold of The Wolves" if available.
    default_index = alliances.index("Freehold of The Wolves") if "Freehold of The Wolves" in alliances else 0
    selected_alliance = st.sidebar.selectbox("Select Alliance", options=alliances, index=default_index)
    df = df[df['Alliance'] == selected_alliance]
    
    # Date range filter
    df['date'] = df['snapshot_date'].dt.date
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    with st.expander("Show Raw Data"):
        st.dataframe(df.head(50))
    
    # Aggregate data by alliance (using sum for totals)
    agg_df = aggregate_by_alliance(df)
    agg_df['date'] = agg_df['snapshot_date'].dt.date
    
    # Compute averages by dividing totals by the number of nations.
    agg_df['avg_attacking_casualties'] = agg_df['Attacking Casualties'] / agg_df['nation_count']
    agg_df['avg_defensive_casualties'] = agg_df['Defensive Casualties'] / agg_df['nation_count']
    agg_df['avg_infrastructure'] = agg_df['Infrastructure'] / agg_df['nation_count']
    agg_df['avg_technology'] = agg_df['Technology'] / agg_df['nation_count']
    agg_df['avg_base_land'] = agg_df['Base Land'] / agg_df['nation_count']
    agg_df['avg_strength'] = agg_df['Strength'] / agg_df['nation_count']
    
    with st.expander("Show Aggregated Alliance Data Table"):
        st.dataframe(agg_df.sort_values('date'))
    
    # Collapsible charts for different metrics:
    
    # 1. Nation Count by Alliance Over Time
    with st.expander("Nation Count by Alliance Over Time"):
        pivot_count = agg_df.pivot(index='date', columns='Alliance', values='nation_count')
        st.line_chart(pivot_count)
    
    # 2. Total Attacking Casualties by Alliance Over Time
    if 'Attacking Casualties' in agg_df.columns:
        with st.expander("Total Attacking Casualties by Alliance Over Time"):
            pivot_attack = agg_df.pivot(index='date', columns='Alliance', values='Attacking Casualties')
            st.line_chart(pivot_attack)
    
    # 3. Average Attacking Casualties by Alliance Over Time
    with st.expander("Average Attacking Casualties by Alliance Over Time"):
        pivot_avg_attack = agg_df.pivot(index='date', columns='Alliance', values='avg_attacking_casualties')
        st.line_chart(pivot_avg_attack)
    
    # 4. Total Defensive Casualties by Alliance Over Time
    if 'Defensive Casualties' in agg_df.columns:
        with st.expander("Total Defensive Casualties by Alliance Over Time"):
            pivot_defense = agg_df.pivot(index='date', columns='Alliance', values='Defensive Casualties')
            st.line_chart(pivot_defense)
    
    # 5. Average Defensive Casualties by Alliance Over Time
    with st.expander("Average Defensive Casualties by Alliance Over Time"):
        pivot_avg_defense = agg_df.pivot(index='date', columns='Alliance', values='avg_defensive_casualties')
        st.line_chart(pivot_avg_defense)
    
    # 6. Total Infrastructure by Alliance Over Time (sums)
    if 'Infrastructure' in agg_df.columns:
        with st.expander("Total Infrastructure by Alliance Over Time"):
            pivot_infra = agg_df.pivot(index='date', columns='Alliance', values='Infrastructure')
            st.line_chart(pivot_infra)
    
    # 7. Average Infrastructure by Alliance Over Time
    with st.expander("Average Infrastructure by Alliance Over Time"):
        pivot_avg_infra = agg_df.pivot(index='date', columns='Alliance', values='avg_infrastructure')
        st.line_chart(pivot_avg_infra)
    
    # 8. Total Technology by Alliance Over Time (sums)
    if 'Technology' in agg_df.columns:
        with st.expander("Total Technology by Alliance Over Time"):
            pivot_tech = agg_df.pivot(index='date', columns='Alliance', values='Technology')
            st.line_chart(pivot_tech)
    
    # 9. Average Technology by Alliance Over Time
    with st.expander("Average Technology by Alliance Over Time"):
        pivot_avg_tech = agg_df.pivot(index='date', columns='Alliance', values='avg_technology')
        st.line_chart(pivot_avg_tech)
    
    # 10. Total Base Land by Alliance Over Time (sums)
    if 'Base Land' in agg_df.columns:
        with st.expander("Total Base Land by Alliance Over Time"):
            pivot_base_land = agg_df.pivot(index='date', columns='Alliance', values='Base Land')
            st.line_chart(pivot_base_land)
    
    # 11. Average Base Land by Alliance Over Time
    with st.expander("Average Base Land by Alliance Over Time"):
        pivot_avg_base_land = agg_df.pivot(index='date', columns='Alliance', values='avg_base_land')
        st.line_chart(pivot_avg_base_land)
    
    # 12. Total Strength by Alliance Over Time (sums)
    if 'Strength' in agg_df.columns:
        with st.expander("Total Strength by Alliance Over Time"):
            pivot_strength = agg_df.pivot(index='date', columns='Alliance', values='Strength')
            st.line_chart(pivot_strength)
    
    # 13. Average Strength by Alliance Over Time
    with st.expander("Average Strength by Alliance Over Time"):
        pivot_avg_strength = agg_df.pivot(index='date', columns='Alliance', values='avg_strength')
        st.line_chart(pivot_avg_strength)
    
    # 14. Nation Activity Distribution: Average Activity Score Over Time
    if 'Activity' in df.columns:
        with st.expander("Average Alliance Activity Over Time (Days)"):
            activity_mapping = {
                "Active in the Last 3 Days": 3,
                "Active This Week": 7,
                "Active Last Week": 14,
                "Active Three Weeks Ago": 21,
                "Active More Than Three Weeks Ago": 28
            }
            df['activity_score'] = df['Activity'].map(activity_mapping)
            df_activity = df.dropna(subset=['activity_score'])
            activity_grouped = df_activity.groupby(['date'])['activity_score'].mean().reset_index()
            activity_grouped = activity_grouped.set_index('date')
            st.line_chart(activity_grouped)
            st.caption("Lower scores indicate more recent activity.")
    
if __name__ == "__main__":
    main()
