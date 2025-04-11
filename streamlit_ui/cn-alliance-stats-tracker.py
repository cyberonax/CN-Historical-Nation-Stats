import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
import re
from datetime import datetime
import altair as alt

##############################
# HELPER FUNCTIONS
##############################

def parse_date_from_filename(filename):
    """
    Extract and parse the date from a filename that follows the pattern:
    "CyberNations_SE_Nation_Stats_<dateToken><zipid>.zip"
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

    # Convert relevant columns to numeric after removing commas.
    for col in numeric_cols:
        if col in df.columns:
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

def altair_line_chart_from_pivot(pivot_df, y_field):
    """
    Creates an Altair line chart from a pivot DataFrame.
    The pivot DataFrame is expected to have its index as dates and columns as Alliance names.
    Axis labels and chart title are removed.
    The color encoding will force:
      - CLAWS to red,
      - Freehold of The Wolves to yellow,
      - NATO to blue.
    """
    # Reset index and melt the dataframe to long format.
    df_long = pivot_df.reset_index().melt(id_vars="date", var_name="Alliance", value_name=y_field)
    color_scale = alt.Scale(
        domain=["CLAWS", "Freehold of The Wolves", "NATO"],
        range=["red", "yellow", "blue"]
    )
    chart = alt.Chart(df_long).mark_line().encode(
        x=alt.X("date:T", title=""),
        y=alt.Y(f"{y_field}:Q", title=""),
        color=alt.Color("Alliance:N", scale=color_scale, legend=alt.Legend(title="Alliance")),
        tooltip=["date:T", "Alliance", f"{y_field}:Q"]
    ).properties(
        width=700,
        height=400
    ).interactive()
    return chart

##############################
# STREAMLIT APP
##############################

def main():
    st.title("Cyber Nations | Nation Stats Timeline Tracker")
    st.markdown("""
        This dashboard displays nation statistics over time.
        Use the tabs below to switch between aggregated alliance charts and individual nation metrics.
    """)
    
    # Load raw data
    df_raw = load_data()
    if df_raw.empty:
        st.error("No data loaded. Please check the 'downloaded_zips' folder.")
        return
    df_raw['snapshot_date'] = pd.to_datetime(df_raw['snapshot_date'])
    
    # Pre-compute resource metrics (shared if needed)
    resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
    df_raw['Empty Slots Count'] = df_raw.apply(lambda row: sum(1 for x in row[resource_cols] if pd.isnull(x) or str(x).strip() == '') // 2, axis=1)
    
    # Create two tabs: one for aggregated metrics and one for individual nation metrics.
    tabs = st.tabs(["Aggregated Alliance Metrics", "Individual Nation Metrics"])
    
    #########################################
    # TAB 1: Aggregated Alliance Metrics
    #########################################
    with tabs[0]:
        st.header("Aggregated Alliance Metrics Over Time")
        
        # Sidebar filters for aggregated charts.
        st.sidebar.header("Aggregated Filters")
        alliances = sorted(df_raw['Alliance'].dropna().unique())
        default1 = alliances.index("Freehold of The Wolves") if "Freehold of The Wolves" in alliances else 0
        default2 = alliances.index("CLAWS") if "CLAWS" in alliances else 0
        default3 = alliances.index("NATO") if "NATO" in alliances else 0
        
        selected_alliance1 = st.sidebar.selectbox("Select Alliance 1", options=alliances, index=default1, key="agg1")
        selected_alliance2 = st.sidebar.selectbox("Select Alliance 2", options=alliances, index=default2, key="agg2")
        selected_alliance3 = st.sidebar.selectbox("Select Alliance 3", options=alliances, index=default3, key="agg3")
        comparison_alliances = [selected_alliance1, selected_alliance2, selected_alliance3]
        
        # Filter data for selected alliances.
        df_agg = df_raw[df_raw['Alliance'].isin(comparison_alliances)].copy()
        df_agg['date'] = df_agg['snapshot_date'].dt.date
        
        # Date range filter
        min_date = df_agg['date'].min()
        max_date = df_agg['date'].max()
        date_range = st.sidebar.date_input("Select date range", [min_date, max_date], key="agg_date")
        if isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
            df_agg = df_agg[(df_agg['date'] >= start_date) & (df_agg['date'] <= end_date)]
        
        with st.expander("Show Raw Data"):
            st.dataframe(df_agg)
        
        # Aggregate data by alliance
        agg_df = aggregate_by_alliance(df_agg)
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
        
        ##############
        # CHARTS USING ALTAIR
        ##############
        
        # Example: Nation Count by Alliance Over Time
        with st.expander("Nation Count by Alliance Over Time"):
            pivot_count = agg_df.pivot(index='date', columns='Alliance', values='nation_count')
            chart = altair_line_chart_from_pivot(pivot_count, "nation_count")
            st.altair_chart(chart, use_container_width=True)
        
        # (Retain your other aggregated charts or add new ones as needed.)
    
    ####################################################
    # TAB 2: Individual Nation Metrics Over Time
    ####################################################
    with tabs[1]:
        st.header("Individual Nation Metrics Over Time")
        
        # Sidebar filter for individual nation metrics.
        st.sidebar.header("Nation Metrics Filters")
        alliances = sorted(df_raw['Alliance'].dropna().unique())
        default_ind = alliances.index("Freehold of The Wolves") if "Freehold of The Wolves" in alliances else 0
        selected_alliance_ind = st.sidebar.selectbox("Select Alliance for Nation Metrics", options=alliances, index=default_ind, key="nation")
        
        # Filter raw data for the selected alliance
        df_indiv = df_raw[df_raw["Alliance"] == selected_alliance_ind].copy()
        df_indiv['date'] = df_indiv['snapshot_date'].dt.date
        
        # Let the user choose which metric to view.
        metric_options = ["Technology", "Infrastructure", "Base Land", "Strength", "Attacking Casualties", "Defensive Casualties"]
        selected_metric = st.selectbox("Select Metric", options=metric_options, key="metric")
        
        st.markdown(f"### {selected_metric} over time for alliance: {selected_alliance_ind}")
        
        # Remove rows where the selected metric is not available.
        chart_data = df_indiv.dropna(subset=[selected_metric])
        if chart_data.empty:
            st.info("No data available for the selected metric and alliance.")
        else:
            # Create a line chart where each line represents one Nation (by Nation ID)
            chart = alt.Chart(chart_data).mark_line().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y(f"{selected_metric}:Q", title=selected_metric),
                color=alt.Color("Nation ID:N", legend=alt.Legend(title="Nation ID")),
                tooltip=["date:T", "Nation ID", f"{selected_metric}:Q"]
            ).properties(
                width=700,
                height=400
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        
if __name__ == "__main__":
    main()
