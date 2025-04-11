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
      date_token = "452025" which is interpreted as:
        month = 4, day = 5, year = 2025.
    Returns a datetime object on success, otherwise None.
    """
    pattern = r'^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$'
    match = re.match(pattern, filename)
    if not match:
        return None
    date_token = match.group(1)
    # Try possible splits: month: 1-2 digits, day: 1-2 digits, year: 4 digits.
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
    Computes:
      - Number of nations (nation_count)
      - Sum of Attacking and Defensive Casualties
      - Averages for Infra, Tech, Base Land, Strength, and Empty Trade Slots (if available)
    """
    # List of all numeric columns we want to aggregate
    numeric_cols = ['Attacking Casualties', 'Defensive Casualties', 'Infra', 'Tech', 'Base Land', 'Strength', 'Empty Trade Slots']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    agg_dict = {
        'Nation ID': 'count',
        'Attacking Casualties': 'sum',
        'Defensive Casualties': 'sum'
    }
    for col in ['Infra', 'Tech', 'Base Land', 'Strength', 'Empty Trade Slots']:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    grouped = df.groupby(['snapshot_date', 'Alliance']).agg(agg_dict).reset_index()
    grouped.rename(columns={'Nation ID': 'nation_count'}, inplace=True)
    return grouped

##############################
# STREAMLIT APP
##############################

def main():
    st.title("Cyber Nations | Nation Stats Timeline Tracker")
    st.markdown("""
        This dashboard displays a time-based stream of nation statistics grouped by alliance.
        Data is loaded from historical zip files stored in the `downloaded_zips` folder.
    """)
    
    # Load historical data
    df = load_data()
    if df.empty:
        st.error("No data loaded. Please check the 'downloaded_zips' folder.")
        return
    
    # Ensure snapshot_date is datetime
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Sidebar Filters: Filter by alliance and date range.
    st.sidebar.header("Filters")
    # Set default text to "Freehold of the Wolves"
    alliance_filter = st.sidebar.text_input("Filter Nations by Alliance", value="Freehold of the Wolves")
    if alliance_filter:
        df = df[df['Alliance'].str.contains(alliance_filter, case=False, na=False)]
    
    # Create a date column for filtering
    df['date'] = df['snapshot_date'].dt.date
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    with st.expander("Show Raw Data"):
        st.dataframe(df.head(50))
    
    # Aggregate data by alliance
    agg_df = aggregate_by_alliance(df)
    agg_df['date'] = agg_df['snapshot_date'].dt.date
    
    st.subheader("Aggregated Alliance Data")
    st.dataframe(agg_df.sort_values('date'))
    
    # Collapsible charts for different metrics
    
    # 1. Nation Count by Alliance
    with st.expander("Nation Count by Alliance Over Time"):
        pivot_count = agg_df.pivot(index='date', columns='Alliance', values='nation_count')
        st.line_chart(pivot_count)
    
    # 2. Total Attacking Casualties by Alliance
    if 'Attacking Casualties' in agg_df.columns:
        with st.expander("Total Attacking Casualties by Alliance Over Time"):
            pivot_attack = agg_df.pivot(index='date', columns='Alliance', values='Attacking Casualties')
            st.line_chart(pivot_attack)
    
    # 3. Total Defensive Casualties by Alliance
    if 'Defensive Casualties' in agg_df.columns:
        with st.expander("Total Defensive Casualties by Alliance Over Time"):
            pivot_defense = agg_df.pivot(index='date', columns='Alliance', values='Defensive Casualties')
            st.line_chart(pivot_defense)
    
    # 4. Average Infra by Alliance
    if 'Infra' in agg_df.columns:
        with st.expander("Average Infra by Alliance Over Time"):
            pivot_infra = agg_df.pivot(index='date', columns='Alliance', values='Infra')
            st.line_chart(pivot_infra)
    
    # 5. Average Tech by Alliance
    if 'Tech' in agg_df.columns:
        with st.expander("Average Tech by Alliance Over Time"):
            pivot_tech = agg_df.pivot(index='date', columns='Alliance', values='Tech')
            st.line_chart(pivot_tech)
    
    # 6. Average Base Land by Alliance
    if 'Base Land' in agg_df.columns:
        with st.expander("Average Base Land by Alliance Over Time"):
            pivot_land = agg_df.pivot(index='date', columns='Alliance', values='Base Land')
            st.line_chart(pivot_land)
    
    # 7. Average Strength by Alliance
    if 'Strength' in agg_df.columns:
        with st.expander("Average Strength by Alliance Over Time"):
            pivot_strength = agg_df.pivot(index='date', columns='Alliance', values='Strength')
            st.line_chart(pivot_strength)
    
    # 8. Average Empty Trade Slots by Alliance
    if 'Empty Trade Slots' in agg_df.columns:
        with st.expander("Average Empty Trade Slots by Alliance Over Time"):
            pivot_empty = agg_df.pivot(index='date', columns='Alliance', values='Empty Trade Slots')
            st.line_chart(pivot_empty)
    
    # 9. Nation Activity Distribution with Recoded Values
    if 'Activity' in df.columns:
        with st.expander("Nation Activity Distribution Over Time"):
            # Map the Activity values to numeric scores
            activity_mapping = {
                "Active in the Last 3 Days": 3,
                "Active This Week": 7,
                "Active Last Week": 14,
                "Active Three Weeks Ago": 21,
                "Active More Than Three Weeks Ago": 28
            }
            df['activity_score'] = df['Activity'].map(activity_mapping)
            # Drop any rows that didn't match the mapping
            df_activity = df.dropna(subset=['activity_score'])
            # Group by date and compute the average activity score (or you can compute sums/counts as needed)
            activity_score_df = df_activity.groupby('date')['activity_score'].mean().reset_index()
            activity_score_df = activity_score_df.set_index('date')
            st.line_chart(activity_score_df)

if __name__ == "__main__":
    main()
