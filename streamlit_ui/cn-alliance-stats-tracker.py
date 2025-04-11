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
      - Counts the number of nations (renamed as 'nation_count').
      - Sums the key metrics.
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
    df_long = pivot_df.reset_index().melt(id_vars="date", var_name="Alliance", value_name=y_field)
    color_scale = alt.Scale(
        domain=["CLAWS", "Freehold of The Wolves", "NATO"],
        range=["red", "yellow", "blue"]
    )
    chart = alt.Chart(df_long).mark_line().encode(
        x=alt.X("date:T", title=""),
        y=alt.Y(f"{y_field}:Q", title=""),
        color=alt.Color("Alliance:N", scale=color_scale, legend=alt.Legend(title="Alliance")),
        tooltip=["date:T", "Alliance", alt.Tooltip(f"{y_field}:Q", title=y_field)]
    ).properties(
        width=700,
        height=400
    ).interactive()
    return chart

def altair_individual_metric_chart(df, metric, title, show_ruler_on_hover=True):
    """
    Creates an Altair line chart from raw nation-level data for a given metric.
    Each line represents a Nation (displayed with its Ruler Name instead of Nation ID).

    When show_ruler_on_hover is True an interactive layer shows the Ruler Name as text on hover.
    """
    # Base chart with common encodings.
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y(f"{metric}:Q", title=title),
        color=alt.Color("Ruler Name:N", legend=alt.Legend(title="Ruler Name"))
    )
    
    # Define tooltip including Ruler Name and other info.
    tooltip = [alt.Tooltip("date:T", title="Date"), "Ruler Name", alt.Tooltip(f"{metric}:Q", title=title)]
    
    # Draw the main line.
    line = base.mark_line()
    
    if show_ruler_on_hover:
        # Create a nearest selection that reacts to mouse hover.
        nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['date'], empty='none')
        
        # Transparent selectors across the chart.
        selectors = base.mark_point().encode(
            opacity=alt.value(0)
        ).add_selection(nearest)
        
        # Points on the line that appear on hover.
        points = line.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )
        
        # Text label to display the Ruler Name.
        text = line.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, "Ruler Name:N", alt.value(''))
        )
        
        # Layer all parts together.
        layered_chart = alt.layer(line, selectors, points, text).encode(
            tooltip=tooltip
        ).properties(
            width=700,
            height=400
        ).interactive()
        chart = layered_chart
    else:
        chart = line.encode(
            tooltip=tooltip
        ).properties(
            width=700,
            height=400
        ).interactive()
        
    return chart

##############################
# STREAMLIT APP
##############################

def main():
    st.title("Cyber Nations | Alliance Stats Timeline Tracker")
    st.markdown("""
        This dashboard displays alliance and nation statistics over time.
        Use the tabs below to switch between aggregated alliance charts and individual nation metrics.
    """)
    
    # Load raw data
    df_raw = load_data()
    if df_raw.empty:
        st.error("No data loaded. Please check the 'downloaded_zips' folder.")
        return
    df_raw['snapshot_date'] = pd.to_datetime(df_raw['snapshot_date'])
    df_raw['date'] = df_raw['snapshot_date'].dt.date

    # Ensure key numeric metrics are converted properly.
    numeric_cols = ['Technology', 'Infrastructure', 'Base Land', 'Strength', 'Attacking Casualties', 'Defensive Casualties']
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    
    # Compute Empty Trade Slots for each nation.
    resource_cols = [f"Connected Resource {i}" for i in range(1, 11)]
    df_raw['Empty Slots Count'] = df_raw.apply(lambda row: sum(1 for x in row[resource_cols] if pd.isnull(x) or str(x).strip() == '') // 2, axis=1)
    
    # If Activity column exists, map activity to a numeric score.
    if 'Activity' in df_raw.columns:
        activity_mapping = {
            "Active In The Last 3 Days": 3,
            "Active This Week": 7,
            "Active Last Week": 14,
            "Active Three Weeks Ago": 21,
            "Active More Than Three Weeks Ago": 28
        }
        df_raw['activity_score'] = df_raw['Activity'].map(activity_mapping)
    
    # Create two tabs: one for aggregated metrics and one for individual nation metrics.
    tabs = st.tabs(["Aggregated Alliance Metrics", "Individual Nation Metrics"])
    
    #########################################
    # TAB 1: Aggregated Alliance Metrics
    #########################################
    with tabs[0]:
        st.header("Aggregated Alliance Metrics Over Time")
        
        # Sidebar filters for aggregated charts.
        st.sidebar.header("Alliance Metrics")
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
        
        # Date range filter.
        min_date = df_agg['date'].min()
        max_date = df_agg['date'].max()
        date_range = st.sidebar.date_input("Select date range", [min_date, max_date], key="agg_date")
        if isinstance(date_range, list) and len(date_range) == 2:
            start_date, end_date = date_range
            df_agg = df_agg[(df_agg['date'] >= start_date) & (df_agg['date'] <= end_date)]
        
        with st.expander("Show Raw Aggregated Data"):
            st.dataframe(df_agg)
        
        # Aggregate data by alliance.
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
        # CHARTS USING ALTAIR FOR AGGREGATED DATA
        ##############
        
        # 1. Nation Count by Alliance Over Time
        with st.expander("Nation Count by Alliance Over Time"):
            pivot_count = agg_df.pivot(index='date', columns='Alliance', values='nation_count')
            chart = altair_line_chart_from_pivot(pivot_count, "nation_count")
            st.altair_chart(chart, use_container_width=True)
        
        # 2. Nation Activity Distribution Over Time (if available)
        if 'activity_score' in df_agg.columns:
            with st.expander("Average Alliance Inactivity Over Time (Days)"):
                activity_grouped = df_agg.dropna(subset=['activity_score']).groupby(['date', 'Alliance'])['activity_score'].mean().reset_index()
                pivot_activity = activity_grouped.pivot(index='date', columns='Alliance', values='activity_score')
                chart = altair_line_chart_from_pivot(pivot_activity, "activity_score")
                st.altair_chart(chart, use_container_width=True)
                st.caption("Lower scores indicate more recent activity.")
        
        # 3. Total Empty Trade Slots by Alliance Over Time
        with st.expander("Total Empty Trade Slots by Alliance Over Time"):
            empty_agg = df_agg.groupby(['snapshot_date', 'Alliance'])['Empty Slots Count'].sum().reset_index()
            empty_agg['date'] = empty_agg['snapshot_date'].dt.date
            pivot_empty_total = empty_agg.pivot(index='date', columns='Alliance', values='Empty Slots Count')
            chart = altair_line_chart_from_pivot(pivot_empty_total, "Empty Slots Count")
            st.altair_chart(chart, use_container_width=True)
        
        # 4. % of Nations with Empty Trade Slots Over Time
        with st.expander("% of Nations with Empty Trade Slots Over Time"):
            total_nations = df_agg.groupby(['snapshot_date', 'Alliance']).agg(total_nations=('Nation ID', 'count')).reset_index()
            empty_nations = df_agg[df_agg['Empty Slots Count'] > 0].groupby(['snapshot_date', 'Alliance']).agg(empty_nations=('Nation ID', 'count')).reset_index()
            ratio_df = pd.merge(total_nations, empty_nations, on=['snapshot_date', 'Alliance'], how='left')
            ratio_df['empty_nations'] = ratio_df['empty_nations'].fillna(0)
            ratio_df['percent_empty'] = (ratio_df['empty_nations'] / ratio_df['total_nations']) * 100
            ratio_df['date'] = ratio_df['snapshot_date'].dt.date
            pivot_ratio = ratio_df.pivot(index='date', columns='Alliance', values='percent_empty')
            chart = altair_line_chart_from_pivot(pivot_ratio, "percent_empty")
            st.altair_chart(chart, use_container_width=True)
        
        # 5. Total Technology by Alliance Over Time
        if 'Technology' in agg_df.columns:
            with st.expander("Total Technology by Alliance Over Time"):
                pivot_tech = agg_df.pivot(index='date', columns='Alliance', values='Technology')
                chart = altair_line_chart_from_pivot(pivot_tech, "Technology")
                st.altair_chart(chart, use_container_width=True)
        
        # 6. Average Technology by Alliance Over Time
        with st.expander("Average Technology by Alliance Over Time"):
            pivot_avg_tech = agg_df.pivot(index='date', columns='Alliance', values='avg_technology')
            chart = altair_line_chart_from_pivot(pivot_avg_tech, "avg_technology")
            st.altair_chart(chart, use_container_width=True)
        
        # 7. Total Infrastructure by Alliance Over Time
        if 'Infrastructure' in agg_df.columns:
            with st.expander("Total Infrastructure by Alliance Over Time"):
                pivot_infra = agg_df.pivot(index='date', columns='Alliance', values='Infrastructure')
                chart = altair_line_chart_from_pivot(pivot_infra, "Infrastructure")
                st.altair_chart(chart, use_container_width=True)
        
        # 8. Average Infrastructure by Alliance Over Time
        with st.expander("Average Infrastructure by Alliance Over Time"):
            pivot_avg_infra = agg_df.pivot(index='date', columns='Alliance', values='avg_infrastructure')
            chart = altair_line_chart_from_pivot(pivot_avg_infra, "avg_infrastructure")
            st.altair_chart(chart, use_container_width=True)
        
        # 9. Total Base Land by Alliance Over Time
        if 'Base Land' in agg_df.columns:
            with st.expander("Total Base Land by Alliance Over Time"):
                pivot_base_land = agg_df.pivot(index='date', columns='Alliance', values='Base Land')
                chart = altair_line_chart_from_pivot(pivot_base_land, "Base Land")
                st.altair_chart(chart, use_container_width=True)
        
        # 10. Average Base Land by Alliance Over Time
        with st.expander("Average Base Land by Alliance Over Time"):
            pivot_avg_base_land = agg_df.pivot(index='date', columns='Alliance', values='avg_base_land')
            chart = altair_line_chart_from_pivot(pivot_avg_base_land, "avg_base_land")
            st.altair_chart(chart, use_container_width=True)
        
        # 11. Total Strength by Alliance Over Time
        if 'Strength' in agg_df.columns:
            with st.expander("Total Strength by Alliance Over Time"):
                pivot_strength = agg_df.pivot(index='date', columns='Alliance', values='Strength')
                chart = altair_line_chart_from_pivot(pivot_strength, "Strength")
                st.altair_chart(chart, use_container_width=True)
        
        # 12. Average Strength by Alliance Over Time
        with st.expander("Average Strength by Alliance Over Time"):
            pivot_avg_strength = agg_df.pivot(index='date', columns='Alliance', values='avg_strength')
            chart = altair_line_chart_from_pivot(pivot_avg_strength, "avg_strength")
            st.altair_chart(chart, use_container_width=True)
        
        # 13. Total Attacking Casualties by Alliance Over Time
        if 'Attacking Casualties' in agg_df.columns:
            with st.expander("Total Attacking Casualties by Alliance Over Time"):
                pivot_attack = agg_df.pivot(index='date', columns='Alliance', values='Attacking Casualties')
                chart = altair_line_chart_from_pivot(pivot_attack, "Attacking Casualties")
                st.altair_chart(chart, use_container_width=True)
        
        # 14. Average Attacking Casualties by Alliance Over Time
        with st.expander("Average Attacking Casualties by Alliance Over Time"):
            pivot_avg_attack = agg_df.pivot(index='date', columns='Alliance', values='avg_attacking_casualties')
            chart = altair_line_chart_from_pivot(pivot_avg_attack, "avg_attacking_casualties")
            st.altair_chart(chart, use_container_width=True)
        
        # 15. Total Defensive Casualties by Alliance Over Time
        if 'Defensive Casualties' in agg_df.columns:
            with st.expander("Total Defensive Casualties by Alliance Over Time"):
                pivot_defense = agg_df.pivot(index='date', columns='Alliance', values='Defensive Casualties')
                chart = altair_line_chart_from_pivot(pivot_defense, "Defensive Casualties")
                st.altair_chart(chart, use_container_width=True)
        
        # 16. Average Defensive Casualties by Alliance Over Time
        with st.expander("Average Defensive Casualties by Alliance Over Time"):
            pivot_avg_defense = agg_df.pivot(index='date', columns='Alliance', values='avg_defensive_casualties')
            chart = altair_line_chart_from_pivot(pivot_avg_defense, "avg_defensive_casualties")
            st.altair_chart(chart, use_container_width=True)
    
    ####################################################
    # TAB 2: Individual Nation Metrics Over Time
    ####################################################
    with tabs[1]:
        st.header("Individual Nation Metrics Over Time")
        
        # Sidebar filter for individual nation metrics.
        st.sidebar.header("Nation Metrics")
        alliances = sorted(df_raw['Alliance'].dropna().unique())
        default_ind = alliances.index("Freehold of The Wolves") if "Freehold of The Wolves" in alliances else 0
        selected_alliance_ind = st.sidebar.selectbox("Select Alliance for Nation Metrics", options=alliances, index=default_ind, key="nation")
        
        # New sidebar multiselect widget to filter by Nation ID.
        # Default selection is empty which means show all nations.
        nation_ids = sorted(df_raw["Nation ID"].dropna().unique())
        selected_nation_ids = st.sidebar.multiselect("Filter by Nation ID", options=nation_ids, default=[], key="filter_nation_id")
        
        # Add an option to show the Ruler Name on hover.
        show_hover = st.sidebar.checkbox("Display Ruler Name on hover", value=True, key="hover_option")
        
        # Filter raw data for the selected alliance.
        df_indiv = df_raw[df_raw["Alliance"] == selected_alliance_ind].copy()
        # If any Nation ID is selected, further filter the data.
        if selected_nation_ids:
            df_indiv = df_indiv[df_indiv["Nation ID"].isin(selected_nation_ids)]
        
        # (Optional) Apply a date range filter for individual nation charts.
        min_date_ind = df_indiv['date'].min()
        max_date_ind = df_indiv['date'].max()
        date_range_ind = st.sidebar.date_input("Select date range (Nation Metrics)", [min_date_ind, max_date_ind], key="nation_date")
        if isinstance(date_range_ind, list) and len(date_range_ind) == 2:
            start_date_ind, end_date_ind = date_range_ind
            df_indiv = df_indiv[(df_indiv['date'] >= start_date_ind) & (df_indiv['date'] <= end_date_ind)]
        
        with st.expander("Show Raw Nation Data"):
            st.dataframe(df_indiv)
        
        st.markdown(f"### Charts for Alliance: {selected_alliance_ind}")
        
        # (a) Nation Activity Distribution Over Time (if available)
        if 'activity_score' in df_indiv.columns:
            with st.expander("Nation Inctivity Over Time (Days)"):
                chart = altair_individual_metric_chart(df_indiv.dropna(subset=['activity_score']), "activity_score", "Activity Score (Days)", show_ruler_on_hover=show_hover)
                st.altair_chart(chart, use_container_width=True)
                st.caption("Lower scores indicate more recent activity.")
                
                # Compute each nation's average activity score across snapshots.
                avg_activity = (
                    df_indiv.dropna(subset=['activity_score'])
                    .groupby(["Nation ID", "Ruler Name"])["activity_score"]
                    .mean()
                    .reset_index()
                    .rename(columns={"activity_score": "All Time Average Days of Inactivity"})
                )
                st.markdown("#### All Time Average Daily Inactivity per Nation")
                st.dataframe(avg_activity)
                st.caption("Only inactive nations are shown.")
        
        # (b) Empty Trade Slots Over Time
        with st.expander("Empty Trade Slots Over Time"):
            chart = altair_individual_metric_chart(df_indiv.dropna(subset=['Empty Slots Count']), "Empty Slots Count", "Empty Trade Slots", show_ruler_on_hover=show_hover)
            st.altair_chart(chart, use_container_width=True)
            
            # Compute each nation's average empty trade slots across snapshots.
            avg_empty = (
                df_indiv.dropna(subset=['Empty Slots Count'])
                .groupby(["Nation ID", "Ruler Name"])["Empty Slots Count"]
                .mean()
                .reset_index()
                .rename(columns={"Empty Slots Count": "All Time Average Empty Trade Slots"})
            )
            st.markdown("#### All Tme Average Empty Trade Slots per Nation")
            st.dataframe(avg_empty)
        
        # (c) Technology Over Time
        if 'Technology' in df_indiv.columns:
            with st.expander("Technology Over Time"):
                chart = altair_individual_metric_chart(df_indiv.dropna(subset=['Technology']), "Technology", "Technology", show_ruler_on_hover=show_hover)
                st.altair_chart(chart, use_container_width=True)
        
        # (d) Infrastructure Over Time
        if 'Infrastructure' in df_indiv.columns:
            with st.expander("Infrastructure Over Time"):
                chart = altair_individual_metric_chart(df_indiv.dropna(subset=['Infrastructure']), "Infrastructure", "Infrastructure", show_ruler_on_hover=show_hover)
                st.altair_chart(chart, use_container_width=True)
        
        # (e) Base Land Over Time
        if 'Base Land' in df_indiv.columns:
            with st.expander("Base Land Over Time"):
                chart = altair_individual_metric_chart(df_indiv.dropna(subset=['Base Land']), "Base Land", "Base Land", show_ruler_on_hover=show_hover)
                st.altair_chart(chart, use_container_width=True)
        
        # (f) Strength Over Time
        if 'Strength' in df_indiv.columns:
            with st.expander("Strength Over Time"):
                chart = altair_individual_metric_chart(df_indiv.dropna(subset=['Strength']), "Strength", "Strength", show_ruler_on_hover=show_hover)
                st.altair_chart(chart, use_container_width=True)
        
        # (g) Attacking Casualties Over Time
        if 'Attacking Casualties' in df_indiv.columns:
            with st.expander("Attacking Casualties Over Time"):
                chart = altair_individual_metric_chart(df_indiv.dropna(subset=['Attacking Casualties']), "Attacking Casualties", "Attacking Casualties", show_ruler_on_hover=show_hover)
                st.altair_chart(chart, use_container_width=True)
        
        # (h) Defensive Casualties Over Time
        if 'Defensive Casualties' in df_indiv.columns:
            with st.expander("Defensive Casualties Over Time"):
                chart = altair_individual_metric_chart(df_indiv.dropna(subset=['Defensive Casualties']), "Defensive Casualties", "Defensive Casualties", show_ruler_on_hover=show_hover)
                st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
