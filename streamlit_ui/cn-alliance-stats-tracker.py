import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
import re
from datetime import datetime
import altair as alt
import itertools

##############################
# HELPER FUNCTIONS
##############################

def parse_date_from_filename(filename):
    """
    Extract and parse the date from a filename that follows the pattern:
    "CyberNations_SE_Nation_Stats_<dateToken><zipid>.zip"
    
    For example, from "CyberNations_SE_Nation_Stats_4132025510001.zip":
      - date_token = "4132025" is interpreted as month=4, day=13, year=2025.
      - The zipid "510001" corresponds to the first 12 hours of the day (00:00).
      - The zipid "510002" corresponds to the last 12 hours of the day (12:00).
      
    Returns a datetime object on success, otherwise None.
    """
    pattern = r'^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$'
    match = re.match(pattern, filename)
    if not match:
        return None
    date_token = match.group(1)
    zip_id = match.group(2)
    # Decide the hour offset based on the zip id.
    # 510001 -> first 12 hours: hour=0; 510002 -> last 12 hours: hour=12.
    hour = 0 if zip_id == "510001" else 12

    # Try different possibilities for the digit splits in the date_token.
    for m_digits in [1, 2]:
        for d_digits in [1, 2]:
            if m_digits + d_digits + 4 == len(date_token):
                try:
                    month = int(date_token[:m_digits])
                    day = int(date_token[m_digits:m_digits+d_digits])
                    year = int(date_token[m_digits+d_digits:m_digits+d_digits+4])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month, day, hour=hour)
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

def altair_line_chart_from_pivot(pivot_df, y_field, alliances, show_alliance_hover=True):
    """
    Creates an Altair line chart from a pivot DataFrame.
    The pivot DataFrame is expected to have its index as dates and columns as Alliance names.
    Axis labels and chart title are removed.
    
    A dynamic color scale is generated based on the provided alliances.
    The tooltip will include the Alliance name if show_alliance_hover is True.
    """
    # Generate a dynamic color scale.
    predefined_colors = ["yellow", "red", "blue", "green", "purple", "orange", "brown", "pink"]
    # If more alliances than predefined colors, cycle through the list.
    color_range = list(itertools.islice(itertools.cycle(predefined_colors), len(alliances)))
    
    df_long = pivot_df.reset_index().melt(id_vars="date", var_name="Alliance", value_name=y_field)
    
    color_scale = alt.Scale(
        domain=alliances,
        range=color_range
    )
    
    # Define tooltip based on checkbox.
    if show_alliance_hover:
        tooltip = ["date:T", "Alliance", alt.Tooltip(f"{y_field}:Q", title=y_field)]
    else:
        tooltip = ["date:T", alt.Tooltip(f"{y_field}:Q", title=y_field)]
    
    # Create base chart.
    base = alt.Chart(df_long).encode(
        x=alt.X("date:T", title=""),
        y=alt.Y(f"{y_field}:Q", title=""),
        color=alt.Color("Alliance:N", scale=color_scale, legend=alt.Legend(title="Alliance"))
    )
    line = base.mark_line()
    
    if show_alliance_hover:
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
        # Text label to display the Alliance name.
        text = line.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, "Alliance:N", alt.value(''))
        )
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

def compute_growth(df, metric):
    """
    Computes the per-day growth for a given metric for each nation.
    For each Nation ID and Ruler Name, it calculates:
       (last_value - first_value) / (number_of_days between last_date and first_date)
    If the snapshot dates are the same, the growth is set to 0.
    
    Returns a DataFrame with columns: Nation ID, Ruler Name, and '{metric} Growth Per Day'.
    """
    # Ensure the data is sorted by date to get the correct first and last snapshots.
    df_sorted = df.sort_values("date")
    
    # Group by Nation ID and Ruler Name and aggregate the first and last values and dates.
    growth_df = df_sorted.groupby(["Nation ID", "Ruler Name"]).agg(
        first_date=("date", "first"),
        last_date=("date", "last"),
        first_val=(metric, "first"),
        last_val=(metric, "last")
    ).reset_index()
    
    # Calculate the number of days between the first and last snapshot.
    growth_df["delta_days"] = (pd.to_datetime(growth_df["last_date"]) - pd.to_datetime(growth_df["first_date"])).dt.days
    
    # Calculate daily growth; if delta_days is zero, set growth to 0 to avoid division by zero.
    growth_df["daily_growth"] = (growth_df["last_val"] - growth_df["first_val"]) / growth_df["delta_days"].replace(0, 1)
    growth_df.loc[growth_df["delta_days"] == 0, "daily_growth"] = 0
    
    # Rename the column to the desired format.
    growth_df = growth_df[["Nation ID", "Ruler Name", "daily_growth"]].rename(
        columns={"daily_growth": f"{metric} Growth Per Day"}
    )
    return growth_df

# New helper functions for aggregated alliance metrics

def current_alliance_stats(agg_df, metric, metric_label):
    """
    For each alliance, get the latest (current) value of the given metric
    based on the most recent snapshot.
    Returns a DataFrame with Alliance and the metric labelled with metric_label.
    """
    current = agg_df.sort_values('date').groupby('Alliance').last().reset_index()
    table = current[['Alliance', metric]].rename(columns={metric: metric_label})
    return table

def compute_alliance_growth(agg_df, metric):
    """
    For each alliance, compute the per-day growth for the given metric
    by comparing the first and last snapshots.
    Returns a DataFrame with Alliance and the growth rate per day.
    """
    growth_list = []
    for alliance, group in agg_df.groupby('Alliance'):
        group = group.sort_values('date')
        first_date = group.iloc[0]['date']
        last_date = group.iloc[-1]['date']
        first_value = group.iloc[0][metric]
        last_value = group.iloc[-1][metric]
        delta_days = (last_date - first_date).days if (last_date - first_date).days != 0 else 1
        growth_rate = (last_value - first_value) / delta_days
        growth_list.append({'Alliance': alliance, f"{metric} Growth Per Day": growth_rate})
    return pd.DataFrame(growth_list)

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
    # Preserve the full timestamp including hour for accurate snapshot differentiation.
    df_raw['date'] = df_raw['snapshot_date']

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
        # Compute the intersection of default selections with the available alliances.
        default_defaults = ["Freehold of The Wolves", "CLAWS", "NATO"]
        default_selection = [a for a in default_defaults if a in alliances]
        selected_alliances = st.sidebar.multiselect("Filter by Alliance", options=alliances, default=default_selection, key="agg_multiselect")
        display_alliance_hover = st.sidebar.checkbox("Display Alliance Name on hover", value=True, key="agg_hover")
        if not selected_alliances:
            selected_alliances = alliances
        
        # Filter data for selected alliances.
        df_agg = df_raw[df_raw['Alliance'].isin(selected_alliances)].copy()
        
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
        # Use the full snapshot_date (with hour) so each half-day snapshot remains unique.
        agg_df['date'] = agg_df['snapshot_date']
        
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
            chart = altair_line_chart_from_pivot(pivot_count, "nation_count", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Display current and growth tables for Nation Count.
            current_nation_count = current_alliance_stats(agg_df, 'nation_count', 'Current Nation Count')
            st.markdown("#### Current Nation Count by Alliance")
            st.dataframe(current_nation_count)
            nation_count_growth = compute_alliance_growth(agg_df, 'nation_count')
            nation_count_growth.rename(columns={"nation_count Growth Per Day": "Nation Count Growth Per Day"}, inplace=True)
            st.markdown("#### Nation Count Growth Per Day")
            st.dataframe(nation_count_growth)
        
        # 2. Average Alliance Inactivity Over Time (Days)
        if 'activity_score' in df_agg.columns:
            with st.expander("Average Alliance Inactivity Over Time (Days)"):
                activity_grouped = df_agg.dropna(subset=['activity_score']).groupby(['date', 'Alliance'])['activity_score'].mean().reset_index()
                pivot_activity = activity_grouped.pivot(index='date', columns='Alliance', values='activity_score')
                chart = altair_line_chart_from_pivot(pivot_activity, "activity_score", selected_alliances, display_alliance_hover)
                st.altair_chart(chart, use_container_width=True)
                st.caption("Lower scores indicate more recent activity.")
                # Current Average Inactivity: for each alliance, use the latest snapshot and average the activity scores.
                current_inactivity = df_agg.dropna(subset=['activity_score']).groupby('Alliance').apply(
                    lambda x: x[x['date'] == x['date'].max()]['activity_score'].mean()
                ).reset_index(name='Current Average Inactivity (Days)')
                st.markdown("#### Current Average Alliance Inactivity (Days)")
                st.dataframe(current_inactivity)
                # All Time Average Inactivity
                all_time_inactivity = df_agg.dropna(subset=['activity_score']).groupby('Alliance')['activity_score'].mean().reset_index().rename(columns={'activity_score': 'All Time Average Inactivity (Days)'})
                st.markdown("#### All Time Average Alliance Inactivity (Days)")
                st.dataframe(all_time_inactivity)
        
        # 3. Total Empty Trade Slots by Alliance Over Time
        with st.expander("Total Empty Trade Slots by Alliance Over Time"):
            empty_agg = df_agg.groupby(['snapshot_date', 'Alliance'])['Empty Slots Count'].sum().reset_index()
            empty_agg['date'] = empty_agg['snapshot_date']
            pivot_empty_total = empty_agg.pivot(index='date', columns='Alliance', values='Empty Slots Count')
            chart = altair_line_chart_from_pivot(pivot_empty_total, "Empty Slots Count", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current Total Empty Trade Slots
            current_empty_total = empty_agg.sort_values('date').groupby('Alliance').last().reset_index()[['Alliance', 'Empty Slots Count']].rename(columns={'Empty Slots Count': 'Current Total Empty Trade Slots'})
            st.markdown("#### Current Total Empty Trade Slots by Alliance")
            st.dataframe(current_empty_total)
            # All Time Average Empty Trade Slots
            avg_empty_total = empty_agg.groupby('Alliance')['Empty Slots Count'].mean().reset_index().rename(columns={'Empty Slots Count': 'All Time Average Empty Trade Slots'})
            st.markdown("#### All Time Average Empty Trade Slots by Alliance")
            st.dataframe(avg_empty_total)
        
        # 4. % of Nations with Empty Trade Slots Over Time
        with st.expander("% of Nations with Empty Trade Slots Over Time"):
            total_nations = df_agg.groupby(['snapshot_date', 'Alliance']).agg(total_nations=('Nation ID', 'count')).reset_index()
            empty_nations = df_agg[df_agg['Empty Slots Count'] > 0].groupby(['snapshot_date', 'Alliance']).agg(empty_nations=('Nation ID', 'count')).reset_index()
            ratio_df = pd.merge(total_nations, empty_nations, on=['snapshot_date', 'Alliance'], how='left')
            ratio_df['empty_nations'] = ratio_df['empty_nations'].fillna(0)
            ratio_df['percent_empty'] = (ratio_df['empty_nations'] / ratio_df['total_nations']) * 100
            ratio_df['date'] = ratio_df['snapshot_date']
            pivot_ratio = ratio_df.pivot(index='date', columns='Alliance', values='percent_empty')
            chart = altair_line_chart_from_pivot(pivot_ratio, "percent_empty", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current % of Nations with Empty Trade Slots
            current_empty_percent = ratio_df.sort_values('date').groupby('Alliance').last().reset_index()[['Alliance', 'percent_empty']].rename(columns={'percent_empty': 'Current % of Nations with Empty Trade Slots'})
            st.markdown("#### Current % of Nations with Empty Trade Slots by Alliance")
            st.dataframe(current_empty_percent)
            # All Time Average % of Nations with Empty Trade Slots
            avg_empty_percent = ratio_df.groupby('Alliance')['percent_empty'].mean().reset_index().rename(columns={'percent_empty': 'All Time Average % of Empty Trade Slots'})
            st.markdown("#### All Time Average % of Nations with Empty Trade Slots by Alliance")
            st.dataframe(avg_empty_percent)
        
        # 5. Total Technology by Alliance Over Time
        if 'Technology' in agg_df.columns:
            with st.expander("Total Technology by Alliance Over Time"):
                pivot_tech = agg_df.pivot(index='date', columns='Alliance', values='Technology')
                chart = altair_line_chart_from_pivot(pivot_tech, "Technology", selected_alliances, display_alliance_hover)
                st.altair_chart(chart, use_container_width=True)
                # Current Total Technology
                current_total_tech = current_alliance_stats(agg_df, 'Technology', 'Current Total Technology')
                st.markdown("#### Current Total Technology by Alliance")
                st.dataframe(current_total_tech)
                # Technology Growth Per Day
                tech_growth = compute_alliance_growth(agg_df, 'Technology')
                tech_growth.rename(columns={"Technology Growth Per Day": "Technology Growth Per Day"}, inplace=True)
                st.markdown("#### Technology Growth Per Day")
                st.dataframe(tech_growth)
        
        # 6. Average Technology by Alliance Over Time
        with st.expander("Average Technology by Alliance Over Time"):
            pivot_avg_tech = agg_df.pivot(index='date', columns='Alliance', values='avg_technology')
            chart = altair_line_chart_from_pivot(pivot_avg_tech, "avg_technology", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current Average Technology
            current_avg_tech = current_alliance_stats(agg_df, 'avg_technology', 'Current Average Technology')
            st.markdown("#### Current Average Technology by Alliance")
            st.dataframe(current_avg_tech)
            # Average Technology Growth Per Day
            avg_tech_growth = compute_alliance_growth(agg_df, 'avg_technology')
            avg_tech_growth.rename(columns={"avg_technology Growth Per Day": "Average Technology Growth Per Day"}, inplace=True)
            st.markdown("#### Average Technology Growth Per Day")
            st.dataframe(avg_tech_growth)
        
        # 7. Total Infrastructure by Alliance Over Time
        if 'Infrastructure' in agg_df.columns:
            with st.expander("Total Infrastructure by Alliance Over Time"):
                pivot_infra = agg_df.pivot(index='date', columns='Alliance', values='Infrastructure')
                chart = altair_line_chart_from_pivot(pivot_infra, "Infrastructure", selected_alliances, display_alliance_hover)
                st.altair_chart(chart, use_container_width=True)
                # Current Total Infrastructure
                current_total_infra = current_alliance_stats(agg_df, 'Infrastructure', 'Current Total Infrastructure')
                st.markdown("#### Current Total Infrastructure by Alliance")
                st.dataframe(current_total_infra)
                # Infrastructure Growth Per Day
                infra_growth = compute_alliance_growth(agg_df, 'Infrastructure')
                infra_growth.rename(columns={"Infrastructure Growth Per Day": "Infrastructure Growth Per Day"}, inplace=True)
                st.markdown("#### Infrastructure Growth Per Day")
                st.dataframe(infra_growth)
        
        # 8. Average Infrastructure by Alliance Over Time
        with st.expander("Average Infrastructure by Alliance Over Time"):
            pivot_avg_infra = agg_df.pivot(index='date', columns='Alliance', values='avg_infrastructure')
            chart = altair_line_chart_from_pivot(pivot_avg_infra, "avg_infrastructure", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current Average Infrastructure
            current_avg_infra = current_alliance_stats(agg_df, 'avg_infrastructure', 'Current Average Infrastructure')
            st.markdown("#### Current Average Infrastructure by Alliance")
            st.dataframe(current_avg_infra)
            # Average Infrastructure Growth Per Day
            avg_infra_growth = compute_alliance_growth(agg_df, 'avg_infrastructure')
            avg_infra_growth.rename(columns={"avg_infrastructure Growth Per Day": "Average Infrastructure Growth Per Day"}, inplace=True)
            st.markdown("#### Average Infrastructure Growth Per Day")
            st.dataframe(avg_infra_growth)
        
        # 9. Total Base Land by Alliance Over Time
        if 'Base Land' in agg_df.columns:
            with st.expander("Total Base Land by Alliance Over Time"):
                pivot_base_land = agg_df.pivot(index='date', columns='Alliance', values='Base Land')
                chart = altair_line_chart_from_pivot(pivot_base_land, "Base Land", selected_alliances, display_alliance_hover)
                st.altair_chart(chart, use_container_width=True)
                # Current Total Base Land
                current_total_base_land = current_alliance_stats(agg_df, 'Base Land', 'Current Total Base Land')
                st.markdown("#### Current Total Base Land by Alliance")
                st.dataframe(current_total_base_land)
                # Total Base Land Growth Per Day
                base_land_growth = compute_alliance_growth(agg_df, 'Base Land')
                base_land_growth.rename(columns={"Base Land Growth Per Day": "Total Base Land Growth Per Day"}, inplace=True)
                st.markdown("#### Total Base Land Growth Per Day")
                st.dataframe(base_land_growth)
        
        # 10. Average Base Land by Alliance Over Time
        with st.expander("Average Base Land by Alliance Over Time"):
            pivot_avg_base_land = agg_df.pivot(index='date', columns='Alliance', values='avg_base_land')
            chart = altair_line_chart_from_pivot(pivot_avg_base_land, "avg_base_land", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current Average Base Land
            current_avg_base_land = current_alliance_stats(agg_df, 'avg_base_land', 'Current Average Base Land')
            st.markdown("#### Current Average Base Land by Alliance")
            st.dataframe(current_avg_base_land)
            # Average Base Land Growth Per Day
            avg_base_land_growth = compute_alliance_growth(agg_df, 'avg_base_land')
            avg_base_land_growth.rename(columns={"avg_base_land Growth Per Day": "Average Base Land Growth Per Day"}, inplace=True)
            st.markdown("#### Average Base Land Growth Per Day")
            st.dataframe(avg_base_land_growth)
        
        # 11. Total Strength by Alliance Over Time
        if 'Strength' in agg_df.columns:
            with st.expander("Total Strength by Alliance Over Time"):
                pivot_strength = agg_df.pivot(index='date', columns='Alliance', values='Strength')
                chart = altair_line_chart_from_pivot(pivot_strength, "Strength", selected_alliances, display_alliance_hover)
                st.altair_chart(chart, use_container_width=True)
                # Current Total Strength
                current_total_strength = current_alliance_stats(agg_df, 'Strength', 'Current Total Strength')
                st.markdown("#### Current Total Strength by Alliance")
                st.dataframe(current_total_strength)
                # Total Strength Growth Per Day
                strength_growth = compute_alliance_growth(agg_df, 'Strength')
                strength_growth.rename(columns={"Strength Growth Per Day": "Total Strength Growth Per Day"}, inplace=True)
                st.markdown("#### Total Strength Growth Per Day")
                st.dataframe(strength_growth)
        
        # 12. Average Strength by Alliance Over Time
        with st.expander("Average Strength by Alliance Over Time"):
            pivot_avg_strength = agg_df.pivot(index='date', columns='Alliance', values='avg_strength')
            chart = altair_line_chart_from_pivot(pivot_avg_strength, "avg_strength", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current Average Strength
            current_avg_strength = current_alliance_stats(agg_df, 'avg_strength', 'Current Average Strength')
            st.markdown("#### Current Average Strength by Alliance")
            st.dataframe(current_avg_strength)
            # Average Strength Growth Per Day
            avg_strength_growth = compute_alliance_growth(agg_df, 'avg_strength')
            avg_strength_growth.rename(columns={"avg_strength Growth Per Day": "Average Strength Growth Per Day"}, inplace=True)
            st.markdown("#### Average Strength Growth Per Day")
            st.dataframe(avg_strength_growth)
        
        # 13. Total Attacking Casualties by Alliance Over Time
        if 'Attacking Casualties' in agg_df.columns:
            with st.expander("Total Attacking Casualties by Alliance Over Time"):
                pivot_attack = agg_df.pivot(index='date', columns='Alliance', values='Attacking Casualties')
                chart = altair_line_chart_from_pivot(pivot_attack, "Attacking Casualties", selected_alliances, display_alliance_hover)
                st.altair_chart(chart, use_container_width=True)
                # Current Total Attacking Casualties
                current_attack = current_alliance_stats(agg_df, 'Attacking Casualties', 'Current Total Attacking Casualties')
                st.markdown("#### Current Total Attacking Casualties by Alliance")
                st.dataframe(current_attack)
                # Attacking Casualties Growth Per Day
                attack_growth = compute_alliance_growth(agg_df, 'Attacking Casualties')
                attack_growth.rename(columns={"Attacking Casualties Growth Per Day": "Attacking Casualties Growth Per Day"}, inplace=True)
                st.markdown("#### Attacking Casualties Growth Per Day")
                st.dataframe(attack_growth)

        # 14. Average Attacking Casualties by Alliance Over Time
        with st.expander("Average Attacking Casualties by Alliance Over Time"):
            pivot_avg_attack = agg_df.pivot(index='date', columns='Alliance', values='avg_attacking_casualties')
            chart = altair_line_chart_from_pivot(pivot_avg_attack, "avg_attacking_casualties", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current Average Attacking Casualties
            current_avg_attack = current_alliance_stats(agg_df, 'avg_attacking_casualties', 'Current Average Attacking Casualties')
            st.markdown("#### Current Average Attacking Casualties by Alliance")
            st.dataframe(current_avg_attack)
            # Average Attacking Casualties Growth Per Day
            avg_attack_growth = compute_alliance_growth(agg_df, 'avg_attacking_casualties')
            avg_attack_growth.rename(columns={"avg_attacking_casualties Growth Per Day": "Average Attacking Casualties Growth Per Day"}, inplace=True)
            st.markdown("#### Average Attacking Casualties Growth Per Day")
            st.dataframe(avg_attack_growth)

        # 15. Total Defensive Casualties by Alliance Over Time
        if 'Defensive Casualties' in agg_df.columns:
            with st.expander("Total Defensive Casualties by Alliance Over Time"):
                pivot_defense = agg_df.pivot(index='date', columns='Alliance', values='Defensive Casualties')
                chart = altair_line_chart_from_pivot(pivot_defense, "Defensive Casualties", selected_alliances, display_alliance_hover)
                st.altair_chart(chart, use_container_width=True)
                # Current Total Defensive Casualties
                current_defense = current_alliance_stats(agg_df, 'Defensive Casualties', 'Current Total Defensive Casualties')
                st.markdown("#### Current Total Defensive Casualties by Alliance")
                st.dataframe(current_defense)
                # Defensive Casualties Growth Per Day
                defense_growth = compute_alliance_growth(agg_df, 'Defensive Casualties')
                defense_growth.rename(columns={"Defensive Casualties Growth Per Day": "Defensive Casualties Growth Per Day"}, inplace=True)
                st.markdown("#### Defensive Casualties Growth Per Day")
                st.dataframe(defense_growth)

        # 16. Average Defensive Casualties by Alliance Over Time
        with st.expander("Average Defensive Casualties by Alliance Over Time"):
            pivot_avg_defense = agg_df.pivot(index='date', columns='Alliance', values='avg_defensive_casualties')
            chart = altair_line_chart_from_pivot(pivot_avg_defense, "avg_defensive_casualties", selected_alliances, display_alliance_hover)
            st.altair_chart(chart, use_container_width=True)
            # Current Average Defensive Casualties
            current_avg_defense = current_alliance_stats(agg_df, 'avg_defensive_casualties', 'Current Average Defensive Casualties')
            st.markdown("#### Current Average Defensive Casualties by Alliance")
            st.dataframe(current_avg_defense)
            # Average Defensive Casualties Growth Per Day
            avg_defense_growth = compute_alliance_growth(agg_df, 'avg_defensive_casualties')
            avg_defense_growth.rename(columns={"avg_defensive_casualties Growth Per Day": "Average Defensive Casualties Growth Per Day"}, inplace=True)
            st.markdown("#### Average Defensive Casualties Growth Per Day")
            st.dataframe(avg_defense_growth)

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
        
        # --- NEW FILTERING LOGIC ---
        # Instead of filtering by the alliance column in the raw data, determine which nations currently belong to the selected alliance.
        # First, filter out rows with missing alliances.
        df_valid = df_raw[df_raw["Alliance"].notnull()].copy()
        # Compute each nation's most recent snapshot from the valid data.
        latest_statuses = df_valid.sort_values("date").groupby("Nation ID").last().reset_index()
        # Identify the Nation IDs whose latest alliance equals the selected alliance.
        current_nation_ids = latest_statuses[latest_statuses["Alliance"] == selected_alliance_ind]["Nation ID"].unique()
        # Now filter the individual nation data to include all snapshots for nations that are currently in the selected alliance.
        df_indiv = df_raw[df_raw["Nation ID"].isin(current_nation_ids)].copy()
        
        # If any Nation ID is additionally selected from the sidebar, further filter the data.
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
            with st.expander("Nation Inactivity Over Time (Days)"):
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
        
        # (b) Empty Trade Slots Over Time
        with st.expander("Empty Trade Slots Over Time"):
            chart = altair_individual_metric_chart(
                df_indiv.dropna(subset=['Empty Slots Count']),
                "Empty Slots Count",
                "Empty Trade Slots",
                show_ruler_on_hover=show_hover
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Compute each nation's average empty trade slots across snapshots.
            avg_empty = (
                df_indiv.dropna(subset=['Empty Slots Count'])
                .groupby(["Nation ID", "Ruler Name"])["Empty Slots Count"]
                .mean()
                .reset_index()
                .rename(columns={"Empty Slots Count": "All Time Average Empty Trade Slots"})
            )
            st.markdown("#### All Time Average Empty Trade Slots per Nation")
            st.dataframe(avg_empty)
            
            # Compute each nation's average activity score (inactivity).
            avg_inactivity = (
                df_indiv.dropna(subset=['activity_score'])
                .groupby(["Nation ID", "Ruler Name"])["activity_score"]
                .mean()
                .reset_index()
                .rename(columns={"activity_score": "All Time Average Inactivity"})
            )
            
            # Merge the two averages on Nation ID and Ruler Name.
            avg_ratio = pd.merge(avg_empty, avg_inactivity, on=["Nation ID", "Ruler Name"], how="inner")
            
            # Compute the Empty-to-Inactivity Ratio.
            # Replace 0 inactivity with None to avoid division by zero.
            avg_ratio["Empty-to-Inactivity Ratio"] = avg_ratio["All Time Average Empty Trade Slots"] / avg_ratio["All Time Average Inactivity"].replace(0, None)
            
            st.markdown("#### Empty Slots-to-Inactivity Ratio per Nation")
            st.dataframe(avg_ratio)
        
        # (c) Technology Over Time
        if 'Technology' in df_indiv.columns:
            with st.expander("Technology Over Time"):
                chart = altair_individual_metric_chart(
                    df_indiv.dropna(subset=['Technology']),
                    "Technology",
                    "Technology",
                    show_ruler_on_hover=show_hover
                )
                st.altair_chart(chart, use_container_width=True)
                
                # Display Technology Growth Per Day Table.
                tech_growth_df = compute_growth(df_indiv.dropna(subset=['Technology']), "Technology")
                st.markdown("#### Technology Growth Per Day")
                st.dataframe(tech_growth_df)
        
        # (d) Infrastructure Over Time
        if 'Infrastructure' in df_indiv.columns:
            with st.expander("Infrastructure Over Time"):
                chart = altair_individual_metric_chart(
                    df_indiv.dropna(subset=['Infrastructure']),
                    "Infrastructure",
                    "Infrastructure",
                    show_ruler_on_hover=show_hover
                )
                st.altair_chart(chart, use_container_width=True)
                
                # Display Infrastructure Growth Per Day Table.
                infra_growth_df = compute_growth(df_indiv.dropna(subset=['Infrastructure']), "Infrastructure")
                st.markdown("#### Infrastructure Growth Per Day")
                st.dataframe(infra_growth_df)
        
        # (e) Base Land Over Time
        if 'Base Land' in df_indiv.columns:
            with st.expander("Base Land Over Time"):
                chart = altair_individual_metric_chart(
                    df_indiv.dropna(subset=['Base Land']),
                    "Base Land",
                    "Base Land",
                    show_ruler_on_hover=show_hover
                )
                st.altair_chart(chart, use_container_width=True)
                
                # Display Base Land Growth Per Day Table.
                base_land_growth_df = compute_growth(df_indiv.dropna(subset=['Base Land']), "Base Land")
                st.markdown("#### Base Land Growth Per Day")
                st.dataframe(base_land_growth_df)
        
        # (f) Strength Over Time
        if 'Strength' in df_indiv.columns:
            with st.expander("Strength Over Time"):
                chart = altair_individual_metric_chart(
                    df_indiv.dropna(subset=['Strength']),
                    "Strength",
                    "Strength",
                    show_ruler_on_hover=show_hover
                )
                st.altair_chart(chart, use_container_width=True)
                
                # Display Strength Growth Per Day Table.
                strength_growth_df = compute_growth(df_indiv.dropna(subset=['Strength']), "Strength")
                st.markdown("#### Strength Growth Per Day")
                st.dataframe(strength_growth_df)
        
        # (g) Attacking Casualties Over Time
        if 'Attacking Casualties' in df_indiv.columns:
            with st.expander("Attacking Casualties Over Time"):
                chart = altair_individual_metric_chart(
                    df_indiv.dropna(subset=['Attacking Casualties']),
                    "Attacking Casualties",
                    "Attacking Casualties",
                    show_ruler_on_hover=show_hover
                )
                st.altair_chart(chart, use_container_width=True)
                # Display Attacking Casualties Growth Per Day Table.
                attack_growth_df = compute_growth(df_indiv.dropna(subset=['Attacking Casualties']), "Attacking Casualties")
                st.markdown("#### Attacking Casualties Growth Per Day")
                st.dataframe(attack_growth_df)
        
        # (h) Defensive Casualties Over Time
        if 'Defensive Casualties' in df_indiv.columns:
            with st.expander("Defensive Casualties Over Time"):
                chart = altair_individual_metric_chart(
                    df_indiv.dropna(subset=['Defensive Casualties']),
                    "Defensive Casualties",
                    "Defensive Casualties",
                    show_ruler_on_hover=show_hover
                )
                st.altair_chart(chart, use_container_width=True)
                # Display Defensive Casualties Growth Per Day Table.
                defense_growth_df = compute_growth(df_indiv.dropna(subset=['Defensive Casualties']), "Defensive Casualties")
                st.markdown("#### Defensive Casualties Growth Per Day")
                st.dataframe(defense_growth_df)

if __name__ == "__main__":
    main()
