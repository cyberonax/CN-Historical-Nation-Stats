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
    for m_digits in [1, 2]:
        for d_digits in [1, 2]:
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

    # ——— Raw data (collapsed by default) ———
    with st.expander("Raw Alliance Data"):
        st.dataframe(df_all)

    st.markdown(f"### Charts for Alliance: {selected_alliance}")

    # Identify most recent snapshot and majority team
    latest_date      = df_all['date'].max()
    latest_snapshot  = df_all[df_all['date'] == latest_date]
    majority_team    = latest_snapshot['Team'].mode().iloc[0]

    # Filter current members by non-pending & majority team
    current_snapshot_filtered = latest_snapshot[
        (latest_snapshot['Alliance Status'] != "Pending") &
        (latest_snapshot['Team'] == majority_team)
    ].copy()
    current_rulers = set(current_snapshot_filtered['Ruler Name'])

    # Build history for those current rulers
    df_indiv = df_all[df_all['Ruler Name'].isin(current_rulers)].copy()

    if 'activity_score' in df_indiv.columns:
        # Compute all-time averages for those rulers
        avg_activity = (
            df_indiv
            .dropna(subset=['activity_score'])
            .groupby('Ruler Name')['activity_score']
            .mean()
            .reset_index()
            .rename(columns={'activity_score': 'All Time Average Days of Inactivity'})
        )

        # Keep only those with avg < 14 days
        valid = set(avg_activity[avg_activity['All Time Average Days of Inactivity'] < 14]['Ruler Name'])
        df_filtered = df_indiv[df_indiv['Ruler Name'].isin(valid)].copy()

        # ——— Inactivity chart & averages (collapsed by default) ———
        with st.expander("Nation Inactivity Over Time In Days (Average Inactivity < 14 Days, Alliance Status =/= Pending, Team = Majority Alliance Team)"):
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

        # ——— Nation details for that same valid set, from the latest snapshot ———
        with st.expander("Nation Details (Ruler Name | Resource 1+2 | Alliance | Team | Days Old | Nation Drill Link | Activity)"):
            details = current_snapshot_filtered[current_snapshot_filtered['Ruler Name'].isin(valid)].copy()
            details["Resource 1+2"] = details.apply(get_resource_1_2, axis=1)
            details["Created"]      = pd.to_datetime(details["Created"], errors='coerce')
            details["Days Old"]     = (pd.Timestamp.now() - details["Created"]).dt.days
            details["Nation Drill Link"] = (
                "https://www.cybernations.net/nation_drill_display.asp?Nation_ID="
                + details["Nation ID"].astype(str)
            )
            # Map average inactivity into the Activity column
            avg_map = avg_activity.set_index('Ruler Name')['All Time Average Days of Inactivity']
            details["Activity"] = details["Ruler Name"].map(avg_map)

            # Include Alliance Status at the end
            details = details[
                ["Ruler Name", "Resource 1+2", "Alliance", "Team",
                 "Days Old", "Nation Drill Link", "Activity", "Alliance Status"]
            ].reset_index(drop=True)
            details.index += 1
            st.dataframe(details)

        # ——— Nations omitted from latest snapshot filtering ———
        with st.expander("Omitted Nations"):
            omitted = latest_snapshot[~latest_snapshot['Ruler Name'].isin(valid)].copy()
            omitted["Resource 1+2"] = omitted.apply(get_resource_1_2, axis=1)
            omitted["Created"]      = pd.to_datetime(omitted["Created"], errors='coerce')
            omitted["Days Old"]     = (pd.Timestamp.now() - omitted["Created"]).dt.days
            omitted["Nation Drill Link"] = (
                "https://www.cybernations.net/nation_drill_display.asp?Nation_ID="
                + omitted["Nation ID"].astype(str)
            )
            omitted["Activity"] = omitted["Ruler Name"].map(
                avg_activity.set_index('Ruler Name')['All Time Average Days of Inactivity']
            )

            # Include Alliance Status at the end
            omitted = omitted[
                ["Ruler Name", "Resource 1+2", "Alliance", "Team",
                 "Days Old", "Nation Drill Link", "Activity", "Alliance Status"]
            ].reset_index(drop=True)
            omitted.index += 1
            st.dataframe(omitted)

        # ——— Valid Resource Combinations ———
        with st.expander("Valid Resource Combinations"):
            st.markdown("### Valid Resource Combinations Input")
            # Text box for Peace Mode - Level A with default combinations.
            peace_a_text = st.text_area(
                "Peace Mode - Level A (one combination per line)",
                value="""Cattle, Coal, Fish, Gems, Gold, Lead, Oil, Rubber, Silver, Spices, Uranium, Wheat
Cattle, Coal, Fish, Gold, Lead, Oil, Pigs, Rubber, Spices, Sugar, Uranium, Wheat
Coal, Fish, Furs, Gems, Gold, Lead, Oil, Rubber, Silver, Uranium, Wheat, Wine
Coal, Fish, Gems, Gold, Lead, Oil, Rubber, Silver, Spices, Sugar, Uranium, Wheat
Coal, Fish, Gems, Gold, Lead, Lumber, Oil, Rubber, Silver, Spices, Uranium, Wheat
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Wheat, Wine
Coal, Fish, Gems, Gold, Lead, Oil, Pigs, Rubber, Silver, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Gold, Iron, Lead, Lumber, Marble, Oil, Rubber, Uranium, Wheat
Coal, Fish, Gems, Gold, Lead, Marble, Oil, Rubber, Silver, Spices, Uranium, Wheat
Cattle, Coal, Fish, Gold, Lead, Lumber, Oil, Rubber, Spices, Sugar, Uranium, Wheat""",
                height=100
            )
            peace_b_text = st.text_area(
                "Peace Mode - Level B (one combination per line)",
                value="""Aluminum, Cattle, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Water, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Sugar, Uranium, Wheat
Aluminum, Coal, Fish, Gems, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Pigs, Rubber, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Silver, Spices, Uranium, Wheat
Aluminum, Coal, Fish, Iron, Lumber, Marble, Oil, Rubber, Spices, Uranium, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Marble, Rubber, Silver, Spices, Uranium, Wheat, Wine
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Wheat, Wine
Aluminum, Cattle, Coal, Fish, Iron, Lumber, Marble, Rubber, Spices, Uranium, Water, Wheat""",
                height=100
            )
            peace_c_text = st.text_area(
                "Peace Mode - Level C (one combination per line)",
                value="""Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Pigs, Rubber, Silver, Spices, Uranium, Wheat, Wine
Cattle, Coal, Fish, Gems, Gold, Pigs, Rubber, Silver, Spices, Sugar, Uranium, Wheat
Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Uranium, Water, Wheat, Wine
Coal, Fish, Furs, Gems, Gold, Oil, Rubber, Silver, Spices, Uranium, Wheat, Wine
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wheat
Cattle, Coal, Fish, Furs, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wine
Cattle, Coal, Fish, Furs, Gems, Gold, Pigs, Rubber, Silver, Spices, Uranium, Wine
Cattle, Coal, Fish, Gems, Gold, Rubber, Silver, Spices, Sugar, Uranium, Wheat, Wine""",
                height=100
            )
            war_text = st.text_area(
                "War Mode (one combination per line)",
                value="""Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Spices, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium, Wheat
Aluminum, Coal, Fish, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Cattle, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Sugar, Uranium
Aluminum, Coal, Furs, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Silver, Uranium
Aluminum, Coal, Gems, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium, Wine
Aluminum, Coal, Gold, Iron, Lead, Lumber, Marble, Oil, Pigs, Rubber, Uranium, Water""",
                height=100
            )
    
            # Brief explanatory text
            st.markdown("""
    ### Understanding Peace Mode Levels
    
    - **Peace Mode Level A:**  
      Nations that are less than **1000 days old**. This level is intended for newer or rapidly developing nations, which may still be adjusting their resource management.
    
    - **Peace Mode Level B:**  
      Nations that are between **1000 and 2000 days old**. These nations are moderately established; their resource combinations may be evolving as they fine‑tune their trade strategies.
    
    - **Peace Mode Level C:**  
      Nations that are **2000 days or older**. These are mature nations with longstanding resource setups, typically expecting more stable and optimized resource combinations.
    """)

        # ——— Input Trade Circless ———
        with st.expander("Input Trade Circles"):
            # — Inputs —
            trade_input  = st.text_area("Trade Circles (one Ruler Name per line)", height=200)
            filter_input = st.text_area("Filter Out Players (one per line)", height=100)
        
            # — Parse into blocks of names; blank lines split circles, but we no longer drop 'x' lines —
            blocks = trade_input.split("\n\n")
            circles = []
            for blk in blocks:
                names = [
                    line.strip()
                    for line in blk.splitlines()
                    if line.strip()  # only drop truly blank lines
                ]
                if names:
                    circles.append(names)
        
            # — Build DataFrame with lowercase merge_key for case‑insensitive matching —
            records = []
            for i, circle in enumerate(circles):
                for name in circle:
                    records.append({
                        "Trade Circle": i+1,
                        "Ruler Name": name,
                        "merge_key": name.lower()
                    })
            tc_df = pd.DataFrame(records, columns=["Trade Circle", "Ruler Name", "merge_key"])
        
            if tc_df.empty:
                st.warning("No valid Trade Circle entries detected.")
            else:
                # — Prepare snapshot with same merge_key & combined resources —
                snap = latest_snapshot.copy()
                snap["merge_key"]    = snap["Ruler Name"].str.lower()
                snap["Resource 1+2"] = snap.apply(get_resource_1_2, axis=1)
        
                # — Merge on merge_key, bringing in your stats —
                tc_df = tc_df.merge(
                    snap[[
                        "merge_key",
                        "Resource 1+2",
                        "Alliance",
                        "Team",
                        "Created",
                        "Nation ID"
                    ]],
                    on="merge_key",
                    how="inner"
                )
        
                # — Compute Days Old, Drill Link, Activity —
                tc_df["Created"] = pd.to_datetime(tc_df["Created"], errors="coerce")
                tc_df["Days Old"] = (pd.Timestamp.now() - tc_df["Created"]).dt.days
                tc_df["Nation Drill Link"] = (
                    "https://www.cybernations.net/nation_drill_display.asp?Nation_ID="
                    + tc_df["Nation ID"].astype(str)
                )
                tc_df["Activity"] = tc_df["Ruler Name"].map(
                    avg_activity.set_index("Ruler Name")["All Time Average Days of Inactivity"]
                )
        
                # — Case‑insensitive filter‑out set —
                filter_set = {ln.strip().lower() for ln in filter_input.splitlines() if ln.strip()}
        
                # — Apply filters & drop merge_key —
                tc_df = (
                    tc_df[
                        (tc_df["Activity"] < 14) &
                        (tc_df["Alliance"] == selected_alliance) &
                        (tc_df["Team"] == majority_team) &
                        (~tc_df["merge_key"].isin(filter_set))
                    ]
                    .drop(columns="merge_key")
                    .reset_index(drop=True)
                )

                # — Processed Trade Circles —
                st.markdown("#### Processed Trade Circles")
                # Reset to 0…n‑1 then shift to 1…n
                tc_df = tc_df.reset_index(drop=True)
                tc_df.index = tc_df.index + 1
            
                st.dataframe(tc_df[[
                    "Trade Circle",
                    "Ruler Name",
                    "Resource 1+2",
                    "Alliance",
                    "Team",
                    "Days Old",
                    "Nation Drill Link",
                    "Activity"
                ]])

                # — Unmatched Players —
                st.markdown("#### Unmatched Players")
                unmatched = details[~details["Ruler Name"].isin(tc_df["Ruler Name"])].copy()
                # sort case‑insensitive and reset to 0…n‑1
                unmatched = (
                    unmatched
                    .sort_values(by="Ruler Name", key=lambda col: col.str.lower())
                    .reset_index(drop=True)
                )
                # shift the index so it starts at 1 instead of 0
                unmatched.index = unmatched.index + 1
                st.dataframe(
                    unmatched[
                        [
                            "Ruler Name",
                            "Resource 1+2",
                            "Alliance",
                            "Team",
                            "Days Old",
                            "Nation Drill Link",
                            "Activity",
                        ]
                    ]
                )

        # ——— Peace Mode Trade Circles ———
        with st.expander("Peace Mode Trade Circles"):
            # 1. Peace‑level function
            def peace_level(days):
                if days < 1000:
                    return "A"
                elif days < 2000:
                    return "B"
                else:
                    return "C"

            # 2. Build df_peace by merging tc_df with details (so Days Old always present)
            df_peace = (
                tc_df[["Trade Circle", "Ruler Name"]]
                .merge(
                    details[[
                        "Ruler Name", "Resource 1+2", "Alliance",
                        "Team", "Days Old", "Nation Drill Link", "Activity"
                    ]],
                    on="Ruler Name",
                    how="left"
                )
            )
            df_peace["Peace Level"] = df_peace["Days Old"].apply(peace_level)

            # 3. Prepare full eligible pool
            eligible = details[[
                "Ruler Name", "Resource 1+2", "Alliance",
                "Team", "Days Old", "Nation Drill Link", "Activity"
            ]].copy()
            eligible["Peace Level"] = eligible["Days Old"].apply(peace_level)
            eligible_idx = eligible.set_index("Ruler Name")

            # 4. Initialize circles by level
            circles = {lvl: {} for lvl in ["A", "B", "C"]}
            for lvl in circles:
                grp = df_peace[df_peace["Peace Level"] == lvl]
                for cid, sub in grp.groupby("Trade Circle"):
                    circles[lvl][cid] = list(sub["Ruler Name"])

            # 5. Find unmatched per level
            unmatched = {}
            for lvl in circles:
                assigned = {r for members in circles[lvl].values() for r in members}
                pool     = set(eligible[eligible["Peace Level"] == lvl]["Ruler Name"])
                unmatched[lvl] = sorted(pool - assigned, key=str.lower)

            # 6. Fill each level’s incomplete circles (fewest gaps first)
            for lvl in circles:
                for cid, members in sorted(circles[lvl].items(), key=lambda x: (6 - len(x[1]), x[0])):
                    while len(members) < 6 and unmatched[lvl]:
                        members.append(unmatched[lvl].pop(0))
                    circles[lvl][cid] = members

            # 7. Cross‑level rebalancing
            def break_and_reassign(from_lvls, to_lvl):
                # donors sorted by most empty slots
                donors = [
                    (lvl, cid, mems)
                    for lvl in from_lvls
                    for cid, mems in circles[lvl].items()
                ]
                donors.sort(key=lambda x: (6 - len(x[2])), reverse=True)
                # targets needing fill
                targets = [(cid, mems) for cid, mems in circles[to_lvl].items() if len(mems) < 6]
                for tid, tm in targets:
                    while len(tm) < 6 and donors:
                        lvl, cid, dm = donors.pop(0)
                        # move everyone from donor into target until full
                        for m in dm:
                            if len(tm) < 6:
                                tm.append(m)
                            else:
                                unmatched[lvl].append(m)
                        del circles[lvl][cid]
                    circles[to_lvl][tid] = tm

            break_and_reassign(["A"], "B")
            break_and_reassign(["A", "B"], "C")

            # 8. Build final DataFrame using eligible_idx for safe lookups
            final = []
            for lvl in ["A", "B", "C"]:
                for idx, (cid, members) in enumerate(sorted(circles[lvl].items()), 1):
                    for name in sorted(members, key=str.lower):
                        if name not in eligible_idx.index:
                            continue
                        row = eligible_idx.loc[name]
                        final.append({
                            "Trade Circle":      f"{lvl}{idx}",
                            "Ruler Name":        name,
                            "Resource 1+2":      row["Resource 1+2"],
                            "Alliance":          row["Alliance"],
                            "Team":              row["Team"],
                            "Days Old":          row["Days Old"],
                            "Nation Drill Link": row["Nation Drill Link"],
                            "Activity":          row["Activity"]
                        })
            cols = [
                "Trade Circle", "Ruler Name", "Resource 1+2",
                "Alliance", "Team", "Days Old", "Nation Drill Link", "Activity"
            ]
            final_df = pd.DataFrame(final, columns=cols)

            st.markdown("#### Peace Mode Trade Circles (Levels A, B, C)")
            st.dataframe(final_df[cols])

            # 9. Leftover players
            leftovers = []
            for lvl, names in unmatched.items():
                for name in names:
                    if name not in eligible_idx.index:
                        continue
                    r = eligible_idx.loc[name]
                    leftovers.append({
                        "Ruler Name":        name,
                        "Resource 1+2":      r["Resource 1+2"],
                        "Alliance":          r["Alliance"],
                        "Team":              r["Team"],
                        "Days Old":          r["Days Old"],
                        "Nation Drill Link": r["Nation Drill Link"],
                        "Activity":          r["Activity"],
                        "Peace Level":       lvl
                    })
            lw_cols = [
                "Ruler Name", "Resource 1+2", "Alliance", "Team",
                "Days Old", "Nation Drill Link", "Activity", "Peace Level"
            ]
            leftovers_df = pd.DataFrame(leftovers, columns=lw_cols)

            st.markdown("#### Leftover Players")
            st.dataframe(leftovers_df[lw_cols])

if __name__ == "__main__":
    main()
