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
      - Count the number of nations (renamed as 'nation_count').
      - Sum the metrics (Attacking Casualties, Defensive Casualties,
        Infrastructure, Technology, Base Land, and Strength).
    """
    numeric_cols = ['Technology', 'Infrastructure', 'Base Land', 'Strength', 'Attacking Casualties', 'Defensive Casualties']

    # Convert relevant columns to string, remove commas, then to numeric
    for col in numeric_cols:
        if col in df.columns:
            # Remove commas and extra spaces before converting
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    
    agg_dict = {
        'Nation ID': 'count',  # This will be renamed to nation_count.
        'Technology': 'sum',
        'Infrastructure': 'sum',
        'Base Land': 'sum',
        'Strength': 'sum',
        'Attacking Casualties': 'sum',
        'Defensive Casualties': 'sum'
    }
    
    grouped = df.groupby(['snapshot_date', 'Alliance']).agg(agg_dict).reset_index()
    grouped.rename(columns={'Nation ID': 'nation_count'}, inplace=True)
    return grouped


def aggregate_totals(df, col):
    """
    Groups the raw data by snapshot_date and Alliance and returns a pivoted DataFrame 
    with the SUM for the specified column.
    """
    grouped = df.groupby(['snapshot_date', 'Alliance'])[col].sum().reset_index()
    grouped['date'] = grouped['snapshot_date'].dt.date
    return grouped.pivot(index='date', columns='Alliance', values=col)

# Function to count empty trade slots
def count_empty_slots(row, resource_cols):
    """
    Count blank resource cells and determine trade slots (each slot covers 2 resources).
    """
    return sum(1 for x in row[resource_cols] if pd.isnull(x) or str(x).strip() == '') // 2

def get_current_resources(row, resource_cols):
    """Return a comma-separated string of non-blank resources sorted alphabetically."""
    resources = sorted([str(x).strip() for x in row[resource_cols] if pd.notnull(x) and str(x).strip() != ''])
    return ", ".join(resources)

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
    
    # Compute Empty Trade Slots
    # Define the resource columns (assumed to be "Connected Resource 1" through "Connected Resource 10")
    resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
    # Compute the empty slots for each nation using the helper function
    df['Empty Slots Count'] = df.apply(lambda row: count_empty_slots(row, resource_cols), axis=1)
    
    # Sidebar Filters: Create a dropdown with available alliances.
    st.sidebar.header("Filters")
    alliances = sorted(df['Alliance'].dropna().unique())
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
        st.dataframe(df)
    
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
    
    # Nation Count by Alliance Over Time
    with st.expander("Nation Count by Alliance Over Time"):
        pivot_count = agg_df.pivot(index='date', columns='Alliance', values='nation_count')
        st.line_chart(pivot_count)

    # Nation Activity Distribution: Average Alliance Activity Over Time (if Activity column exists)
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

    # Total Empty Trade Slots by Alliance Over Time
    with st.expander("Total Empty Trade Slots by Alliance Over Time"):
        empty_agg = df.groupby(['snapshot_date', 'Alliance'])['Empty Slots Count'].sum().reset_index()
        empty_agg['date'] = empty_agg['snapshot_date'].dt.date
        pivot_empty_total = empty_agg.pivot(index='date', columns='Alliance', values='Empty Slots Count')
        st.line_chart(pivot_empty_total)

    # Percentage of Nations with Empty Trade Slots ---
    with st.expander("% of Nations with Empty Trade Slots Over Time"):
        # Group total nations per snapshot_date and Alliance
        total_nations = df.groupby(['snapshot_date', 'Alliance']).agg(total_nations=('Nation ID', 'count')).reset_index()
        # Group nations with empty slots (> 0)
        empty_nations = df[df['Empty Slots Count'] > 0].groupby(['snapshot_date', 'Alliance']).agg(empty_nations=('Nation ID', 'count')).reset_index()
        # Merge the two groups to compute the ratio
        ratio_df = pd.merge(total_nations, empty_nations, on=['snapshot_date', 'Alliance'], how='left')
        ratio_df['empty_nations'] = ratio_df['empty_nations'].fillna(0)
        ratio_df['percent_empty'] = (ratio_df['empty_nations'] / ratio_df['total_nations']) * 100
        ratio_df['date'] = ratio_df['snapshot_date'].dt.date
        pivot_ratio = ratio_df.pivot(index='date', columns='Alliance', values='percent_empty')
        st.line_chart(pivot_ratio)

    # Total Technology by Alliance Over Time (sums)
    if 'Technology' in agg_df.columns:
        with st.expander("Total Technology by Alliance Over Time"):
            pivot_tech = agg_df.pivot(index='date', columns='Alliance', values='Technology')
            st.line_chart(pivot_tech)
    
    # Average Technology by Alliance Over Time
    with st.expander("Average Technology by Alliance Over Time"):
        pivot_avg_tech = agg_df.pivot(index='date', columns='Alliance', values='avg_technology')
        st.line_chart(pivot_avg_tech)
    
    # Total Infrastructure by Alliance Over Time (sums)
    if 'Infrastructure' in agg_df.columns:
        with st.expander("Total Infrastructure by Alliance Over Time"):
            pivot_infra = agg_df.pivot(index='date', columns='Alliance', values='Infrastructure')
            st.line_chart(pivot_infra)
    
    # Average Infrastructure by Alliance Over Time
    with st.expander("Average Infrastructure by Alliance Over Time"):
        pivot_avg_infra = agg_df.pivot(index='date', columns='Alliance', values='avg_infrastructure')
        st.line_chart(pivot_avg_infra)    
    
    # Total Base Land by Alliance Over Time (sums)
    if 'Base Land' in agg_df.columns:
        with st.expander("Total Base Land by Alliance Over Time"):
            pivot_base_land = agg_df.pivot(index='date', columns='Alliance', values='Base Land')
            st.line_chart(pivot_base_land)
    
    # Average Base Land by Alliance Over Time
    with st.expander("Average Base Land by Alliance Over Time"):
        pivot_avg_base_land = agg_df.pivot(index='date', columns='Alliance', values='avg_base_land')
        st.line_chart(pivot_avg_base_land)
    
    # Total Strength by Alliance Over Time (sums)
    if 'Strength' in agg_df.columns:
        with st.expander("Total Strength by Alliance Over Time"):
            pivot_strength = agg_df.pivot(index='date', columns='Alliance', values='Strength')
            st.line_chart(pivot_strength)
    
    # Average Strength by Alliance Over Time
    with st.expander("Average Strength by Alliance Over Time"):
        pivot_avg_strength = agg_df.pivot(index='date', columns='Alliance', values='avg_strength')
        st.line_chart(pivot_avg_strength)

    # Total Attacking Casualties by Alliance Over Time
    if 'Attacking Casualties' in agg_df.columns:
        with st.expander("Total Attacking Casualties by Alliance Over Time"):
            pivot_attack = agg_df.pivot(index='date', columns='Alliance', values='Attacking Casualties')
            st.line_chart(pivot_attack)

    # Average Attacking Casualties by Alliance Over Time
    with st.expander("Average Attacking Casualties by Alliance Over Time"):
        pivot_avg_attack = agg_df.pivot(index='date', columns='Alliance', values='avg_attacking_casualties')
        st.line_chart(pivot_avg_attack)
    
    # Total Defensive Casualties by Alliance Over Time
    if 'Defensive Casualties' in agg_df.columns:
        with st.expander("Total Defensive Casualties by Alliance Over Time"):
            pivot_defense = agg_df.pivot(index='date', columns='Alliance', values='Defensive Casualties')
            st.line_chart(pivot_defense)
    
    # Average Defensive Casualties by Alliance Over Time
    with st.expander("Average Defensive Casualties by Alliance Over Time"):
        pivot_avg_defense = agg_df.pivot(index='date', columns='Alliance', values='avg_defensive_casualties')
        st.line_chart(pivot_avg_defense)
    
if __name__ == "__main__":
    main()
