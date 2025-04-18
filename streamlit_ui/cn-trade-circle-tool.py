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
            # if there's no processed Trade Circle data, skip
            if 'tc_df' not in locals() or tc_df.empty:
                st.markdown("_No Trade Circles to process. Please add entries in the Input Trade Circles section above._")
            else:
                # helper to assign Peace Mode Level
                def peace_level(days):
                    if days < 1000:
                        return 'Level A'
                    elif days < 2000:
                        return 'Level B'
                    else:
                        return 'Level C'

                # base circles and unmatched, with levels
                pm_base = tc_df.copy()
                pm_base['Peace Mode Level'] = pm_base['Days Old'].apply(peace_level)

                pm_unmatched = details[~details['Ruler Name'].isin(pm_base['Ruler Name'])].copy()
                pm_unmatched['Peace Mode Level'] = pm_unmatched['Days Old'].apply(peace_level)

                final_records = []
                leftover_records = []

                # process each Peace Mode Level in order A → B → C
                for level in ['Level A', 'Level B', 'Level C']:
                    lvl_base = pm_base[pm_base['Peace Mode Level'] == level].copy()
                    lvl_un   = pm_unmatched[pm_unmatched['Peace Mode Level'] == level].copy()

                    # count existing members in each circle
                    sizes = lvl_base.groupby('Trade Circle').size() if not lvl_base.empty else pd.Series(dtype=int)
                    incomplete = {c: 6 - cnt for c, cnt in sizes.items() if cnt < 6}
                    inc_sorted = sorted(incomplete.items(), key=lambda x: x[1])  # fewest empty slots first

                    # fill incomplete circles first
                    for circle_id, slots in inc_sorted:
                        to_add = min(slots, len(lvl_un))
                        adds = lvl_un.iloc[:to_add]
                        lvl_un = lvl_un.iloc[to_add:]
                        for _, row in adds.iterrows():
                            rec = row.to_dict()
                            rec['Trade Circle'] = circle_id
                            final_records.append(rec)

                    # keep all original members in that level
                    for _, row in lvl_base.iterrows():
                        final_records.append(row.to_dict())

                    # form new full circles from remaining unmatched
                    max_circle = int(lvl_base['Trade Circle'].max()) if not lvl_base.empty else 0
                    next_circle = max_circle + 1
                    while len(lvl_un) >= 6:
                        group = lvl_un.iloc[:6]
                        lvl_un = lvl_un.iloc[6:]
                        for _, row in group.iterrows():
                            rec = row.to_dict()
                            rec['Trade Circle'] = next_circle
                            final_records.append(rec)
                        next_circle += 1

                    # anything left here becomes leftovers
                    leftover_records += [r.to_dict() for _, r in lvl_un.iterrows()]

                # build final DataFrame
                final_df = pd.DataFrame(final_records)
                level_order = {'Level A':0, 'Level B':1, 'Level C':2}
                final_df = final_df.sort_values(
                    ['Peace Mode Level','Trade Circle','Ruler Name'],
                    key=lambda col: col.map(level_order) if col.name=='Peace Mode Level' else col
                ).reset_index(drop=True)
                final_df.index += 1

                st.markdown("#### Final Peace Mode Trade Circles")
                st.dataframe(final_df[[
                    "Peace Mode Level","Trade Circle","Ruler Name",
                    "Resource 1+2","Alliance","Team",
                    "Days Old","Nation Drill Link","Activity"
                ]])

                # build leftovers DataFrame with explicit columns to avoid KeyErrors
                leftover_cols = ["Ruler Name","Resource 1+2","Alliance","Team","Days Old","Nation Drill Link","Activity"]
                leftovers_df = pd.DataFrame(leftover_records, columns=leftover_cols)

                st.markdown("#### Players Left Over")
                if leftovers_df.empty:
                    st.markdown("_No unmatched players remain._")
                else:
                    leftovers_df.index = range(1, len(leftovers_df)+1)
                    st.dataframe(leftovers_df[leftover_cols])

        # ——— Optimal Peace Mode Trade Circles via ILP with Circle‐Usage Penalty ———
        with st.expander("Optimal Peace Mode Trade Circles"):
            # ensure there is processed data
            if 'final_df' not in locals() or final_df.empty:
                st.markdown("_No processed Trade Circles to optimize. Please fill in the Input Trade Circles above._")
            else:
                try:
                    import pulp
                except ImportError:
                    st.error(
                        "🚨 *PuLP* is not installed. "
                        "Add `pulp` to your dependencies (e.g. in requirements.txt) and redeploy."
                    )
                else:
                    import math

                    level_order = {'Level A': 0, 'Level B': 1, 'Level C': 2}
                    optimal_records = []

                    for level in ['Level A', 'Level B', 'Level C']:
                        # slice out this level
                        df_lvl = final_df[final_df['Peace Mode Level'] == level].copy()
                        if df_lvl.empty:
                            continue

                        nations     = df_lvl['Ruler Name'].tolist()
                        orig_circle = dict(zip(nations, df_lvl['Trade Circle']))
                        existing_cs = sorted(df_lvl['Trade Circle'].unique())

                        # how many circles we need at minimum
                        total_n = len(nations)
                        required_circles = math.ceil(total_n / 6)

                        # build the full set of circle IDs to choose from
                        max_existing = existing_cs[-1] if existing_cs else 0
                        # allow exactly as many new circles as needed to reach required_circles
                        new_cs_count = max(required_circles - len(existing_cs), 0)
                        new_cs = list(range(max_existing + 1, max_existing + 1 + new_cs_count))
                        all_cs = existing_cs + new_cs

                        # set up ILP
                        prob = pulp.LpProblem(f"TradeCircle_{level}", pulp.LpMaximize)
                        # x[p,c] = 1 if nation p is assigned to circle c
                        x = pulp.LpVariable.dicts(
                            "x",
                            ((p, c) for p in nations for c in all_cs),
                            cat='Binary'
                        )
                        # y[c] = 1 if circle c is used at all
                        y = pulp.LpVariable.dicts(
                            "y",
                            all_cs,
                            cat='Binary'
                        )

                        # each nation assigned to exactly one circle
                        for p in nations:
                            prob += pulp.lpSum(x[p, c] for c in all_cs) == 1

                        # capacity and usage constraints per circle
                        for c in all_cs:
                            # at most 6 per circle
                            prob += pulp.lpSum(x[p, c] for p in nations) <= 6 * y[c]
                            # no assignment unless circle is "on"
                            for p in nations:
                                prob += x[p, c] <= y[c]

                        # do not exceed the minimum needed circles
                        prob += pulp.lpSum(y[c] for c in all_cs) <= required_circles

                        # objective: maximize assignments, minimize circles used, minimize reassignments
                        flow = pulp.lpSum(x[p, c] for p in nations for c in all_cs)
                        circle_penalty = pulp.lpSum(y[c] for c in all_cs)
                        reassign_cost = pulp.lpSum(
                            x[p, c] * (
                                0 if c == orig_circle[p]
                                else (1 if c in existing_cs else 6)
                            )
                            for p in nations for c in all_cs
                        )
                        # big weight on flow, moderate on circle usage, light on reassignment
                        prob += 1000 * flow - 10 * circle_penalty - reassign_cost

                        prob.solve(pulp.PULP_CBC_CMD(msg=False))

                        # collect the optimized assignments
                        for p in nations:
                            for c in all_cs:
                                if pulp.value(x[p, c]) == 1:
                                    row = df_lvl[df_lvl['Ruler Name'] == p].iloc[0].to_dict()
                                    row['Trade Circle'] = int(c)
                                    optimal_records.append(row)
                                    break

                    # build and display the final optimized DataFrame
                    opt_df = pd.DataFrame(optimal_records)
                    renumbered = []
                    for level, grp in opt_df.groupby('Peace Mode Level', sort=False):
                        old_ids = sorted(grp['Trade Circle'].unique())
                        id_map  = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=1)}
                        grp = grp.copy()
                        grp['Trade Circle'] = grp['Trade Circle'].map(id_map)
                        renumbered.append(grp)
                    opt_df = pd.concat(renumbered, ignore_index=True)
                
                    level_order = {'Level A': 0, 'Level B': 1, 'Level C': 2}
                    opt_df = opt_df.sort_values(
                        ['Peace Mode Level','Trade Circle','Ruler Name'],
                        key=lambda col: (
                            col.map(level_order)
                            if col.name == 'Peace Mode Level' else
                            col if col.name == 'Trade Circle' else
                            col.str.lower()
                        )
                    ).reset_index(drop=True)
                    opt_df.index += 1
                
                    st.markdown("##### Optimal Trade Circles")
                    st.dataframe(opt_df[[
                        "Peace Mode Level","Trade Circle","Ruler Name",
                        "Resource 1+2","Alliance","Team",
                        "Days Old","Nation Drill Link","Activity"
                    ]])

                    # show any leftovers (if required_circles was underestimated)
                    assigned = set(opt_df['Ruler Name'])
                    leftovers = final_df[~final_df['Ruler Name'].isin(assigned)].copy()
                    st.markdown("##### Players Left Over")
                    if leftovers.empty:
                        st.markdown("_No unmatched players remain._")
                    else:
                        leftovers.index = range(1, len(leftovers) + 1)
                        st.dataframe(leftovers[[
                            "Ruler Name", "Resource 1+2", "Alliance", "Team",
                            "Days Old", "Nation Drill Link", "Activity"
                        ]])

if __name__ == "__main__":
    main()
