import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
import re
from datetime import datetime
import altair as alt

st.set_page_config(layout="wide")

##############################
# HELPER FUNCTIONS
##############################

def parse_date_from_filename(filename):
    """
    Extract and parse the date from a filename that follows the pattern:
    "CyberNations_SE_Nation_Stats_<dateToken><zipid>.zip"
    """
    pattern = r'^CyberNations_SE_Nation_Stats_([0-9]+)(510001|510002)\.zip$'
    match = re.match(pattern, filename)
    if not match:
        return None
    date_token, zip_id = match.groups()
    hour = 0 if zip_id == "510001" else 12
    for m_digits in [1,2]:
        for d_digits in [1,2]:
            if m_digits + d_digits + 4 == len(date_token):
                try:
                    month = int(date_token[:m_digits])
                    day   = int(date_token[m_digits:m_digits+d_digits])
                    year  = int(date_token[m_digits+d_digits:])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month, day, hour=hour)
                except:
                    continue
    return None

def load_data():
    """
    Load and concatenate all snapshots from the downloaded_zips folder
    on every run, ensuring always-fresh data.
    """
    dfs = []
    zip_folder = Path("downloaded_zips")
    if not zip_folder.exists():
        st.warning("Folder 'downloaded_zips' not found.")
        return pd.DataFrame()
    for zf in zip_folder.glob("CyberNations_SE_Nation_Stats_*.zip"):
        date = parse_date_from_filename(zf.name)
        if date is None:
            st.warning(f"Could not parse date from {zf.name}")
            continue
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                name = z.namelist()[0]
                with z.open(name) as f:
                    df = pd.read_csv(f, delimiter="|", encoding="ISO-8859-1", low_memory=False)
                    df['snapshot_date'] = date
                    dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {zf.name}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def map_activity_scores(df):
    mapping = {
        "Active In The Last 3 Days": 3,
        "Active This Week": 7,
        "Active Last Week": 14,
        "Active Three Weeks Ago": 21,
        "Active More Than Three Weeks Ago": 28
    }
    df['activity_score'] = df['Activity'].map(mapping)
    return df

def get_resource_1_2(row):
    """Combine Resource 1 and Resource 2 into one field."""
    r1 = str(row.get("Resource 1", "")).strip()
    r2 = str(row.get("Resource 2", "")).strip()
    if r1 and r2:
        return f"{r1}, {r2}"
    return r1 or r2 or ""

def altair_individual_metric_chart(df, metric, title, show_hover=True):
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y(f"{metric}:Q", title=title),
        color=alt.Color("Ruler Name:N", legend=alt.Legend(title="Ruler"))
    )
    line = base.mark_line()
    if show_hover:
        nearest   = alt.selection(type='single', nearest=True, on='mouseover', fields=['date'], empty='none')
        selectors = base.mark_point().encode(opacity=alt.value(0)).add_selection(nearest)
        points    = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        text      = line.mark_text(align='left', dx=5, dy=-5)\
                         .encode(text=alt.condition(nearest, "Ruler Name:N", alt.value('')))
        tooltip   = [
            alt.Tooltip("date:T", title="Date"),
            "Ruler Name",
            alt.Tooltip(f"{metric}:Q", title=title)
        ]
        return alt.layer(line, selectors, points, text)\
                  .encode(tooltip=tooltip)\
                  .properties(width=800, height=400)\
                  .interactive()
    else:
        return line.properties(width=800, height=400).interactive()

##############################
# STREAMLIT APP
##############################

def main():
    st.title("Cyber Nations | Trade Circle Tool")

    # Always load fresh data
    df = load_data()
    if df.empty:
        st.error("No data loaded.")
        return

    # Preprocess
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['date']          = df['snapshot_date']
    df['Alliance']      = df['Alliance'].fillna("None")
    df['Ruler Name']    = df['Ruler Name'].fillna("None")

    if 'Activity' in df.columns:
        df = map_activity_scores(df)

    # Alliance selector
    alliances = sorted(df['Alliance'].unique())
    default_idx = alliances.index("Freehold of The Wolves") if "Freehold of The Wolves" in alliances else 0
    selected_alliance = st.selectbox("Select Alliance", alliances, index=default_idx)

    df_all = df[df['Alliance'] == selected_alliance].copy()
    if df_all.empty:
        st.warning("No data for that alliance.")
        return

    # Raw data (collapsed by default)
    with st.expander("Raw Alliance Data"):
        st.dataframe(df_all)

    st.markdown(f"### Charts for Alliance: {selected_alliance}")

    # Most recent snapshot and majority team
    latest_date     = df_all['date'].max()
    latest_snapshot = df_all[df_all['date'] == latest_date]
    majority_team   = latest_snapshot['Team'].mode().iloc[0]

    # Build per-nation history filtered to current, non-pending, majority-team members
    df_indiv = df_all[
        (df_all['Ruler Name'].isin(latest_snapshot['Ruler Name'])) &
        (df_all['Alliance Status'] != "Pending") &
        (df_all['Team'] == majority_team)
    ].copy()

    if 'activity_score' in df_indiv.columns:
        # Compute and filter by all-time average inactivity
        avg_activity = (
            df_indiv
            .dropna(subset=['activity_score'])
            .groupby('Ruler Name')['activity_score']
            .mean()
            .reset_index()
            .rename(columns={'activity_score': 'All Time Average Days of Inactivity'})
        )

        valid = set(avg_activity[avg_activity['All Time Average Days of Inactivity'] < 14]['Ruler Name'])
        df_filtered = df_indiv[df_indiv['Ruler Name'].isin(valid)].copy()

        # Inactivity chart & table (collapsed by default)
        with st.expander("Nation Inactivity Over Time In (Days)"):
            st.altair_chart(
                altair_individual_metric_chart(
                    df_filtered,
                    "activity_score",
                    "Activity Score (Days)",
                    show_hover=True
                ),
                use_container_width=True
            )
            st.caption(
                "Lower scores indicate more recent activity. "
                "Showcasing only nations under 14 days of all time average days of inactivity."
            )

            avg_display = (
                avg_activity[avg_activity['Ruler Name'].isin(valid)]
                .sort_values('All Time Average Days of Inactivity', ascending=False)
                .reset_index(drop=True)
            )
            avg_display.index += 1
            st.markdown("#### All Time Average Days of Inactivity")
            st.dataframe(avg_display)

        # Nation details (collapsed by default), based on most recent snapshot
        with st.expander("Nation Details"):
            # start from latest_snapshot, apply same filters and valid set
            details = latest_snapshot[
                (latest_snapshot['Alliance Status'] != "Pending") &
                (latest_snapshot['Team'] == majority_team) &
                (latest_snapshot['Ruler Name'].isin(valid))
            ].copy()
            # combine resources
            details["Resource 1+2"] = details.apply(get_resource_1_2, axis=1)
            # compute Days Old since 'Created'
            details["Created"]     = pd.to_datetime(details["Created"], errors='coerce')
            details["Days Old"]    = (pd.Timestamp.now() - details["Created"]).dt.days
            # nation drill link
            details["Nation Drill Link"] = (
                "https://www.cybernations.net/nation_drill_display.asp?Nation_ID="
                + details["Nation ID"].astype(str)
            )
            # map in the all-time average inactivity as the Activity column
            avg_map = avg_activity.set_index('Ruler Name')['All Time Average Days of Inactivity']
            details["Activity"] = details["Ruler Name"].map(avg_map)

            # select and order
            details = details[
                ["Ruler Name", "Resource 1+2", "Alliance", "Team",
                 "Days Old", "Nation Drill Link", "Activity"]
            ].reset_index(drop=True)
            details.index += 1
            st.dataframe(details)

if __name__ == "__main__":
    main()
