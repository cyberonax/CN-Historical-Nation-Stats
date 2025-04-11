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
    
    The dateToken is a concatenation of month, day, and year. Since month and day
    may be 1 or 2 digits, we try possible splits.
    Example:
      - For "CyberNations_SE_Nation_Stats_452025510002.zip":
          date_token = "452025" which we interpret as:
            month = 4, day = 5, year = 2025.
            
    Returns a datetime object if parsing is successful, otherwise None.
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
                    # Quick validation
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month, day)
                except Exception:
                    continue
    return None

@st.cache_data(show_spinner="Loading historical data...")
def load_data():
    """
    Loads data from all zip files in the "downloaded_zips" folder.
    Each zip file is expected to contain a CSV file with a '|' delimiter
    and encoding 'ISO-8859-1'. A new column 'snapshot_date' is added based on
    the filename date.
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
    Aggregates nation stats by 'snapshot_date' and 'Alliance'.
    In this example, we simply count the number of nations per alliance.
    You may extend this to average/sum other metrics (e.g., casualties).
    """
    # Convert casualty columns to numeric if available.
    for col in ['Attacking Casualties', 'Defensive Casualties']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    grouped = df.groupby(['snapshot_date', 'Alliance']).agg(
        nation_count=('Nation ID', 'count'),
        total_attacking_casualties=('Attacking Casualties', 'sum'),
        total_defensive_casualties=('Defensive Casualties', 'sum')
    ).reset_index()
    return grouped

##############################
# STREAMLIT APP
##############################

def main():
    st.title("CyberNations: Historical Nation Stats by Alliance")
    st.markdown("""
        This dashboard displays a time-based stream of nation statistics grouped by alliance.
        Data is loaded from historical zip files stored in the `downloaded_zips` folder.
    """)

    # Load historical data
    df = load_data()
    if df.empty:
        st.error("No data loaded. Please check the 'downloaded_zips' folder.")
        return
    
    # Ensure snapshot_date is in datetime format
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Display raw data (optional, within an expander)
    with st.expander("Show Raw Data"):
        st.dataframe(df.head(50))
    
    # Aggregate data by alliance per snapshot_date
    agg_df = aggregate_by_alliance(df)
    
    # Convert snapshot_date to date (for easier filtering)
    agg_df['date'] = agg_df['snapshot_date'].dt.date
    min_date = agg_df['date'].min()
    max_date = agg_df['date'].max()

    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (agg_df['date'] >= start_date) & (agg_df['date'] <= end_date)
        filtered = agg_df[mask]
    else:
        filtered = agg_df

    # Pivot the data for a time-series line chart (nation_count by alliance)
    pivot_df = filtered.pivot(index='date', columns='Alliance', values='nation_count')
    st.subheader("Nation Count by Alliance Over Time")
    st.line_chart(pivot_df)

    st.subheader("Aggregated Alliance Data")
    st.dataframe(filtered.sort_values('date'))

if __name__ == "__main__":
    main()
