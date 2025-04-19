import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
import re
from datetime import datetime
import altair as alt
import numpy as np
from scipy.optimize import linear_sum_assignment
import io
import streamlit.components.v1 as components

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
    st.title("Cyber Nations | Trade Circle Optimization Tool")

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

    st.markdown(f"### Optimizing Trade Circles for Alliance: {selected_alliance}")

    # Identify most recent snapshot and majority team
    latest_date      = df_all['date'].max()
    latest_snapshot  = df_all[df_all['date'] == latest_date]
    majority_team    = latest_snapshot['Team'].mode().iloc[0]

    include_pending = st.checkbox(
        "Include Pending Status Nations",
        value=False
    )

    # Filter current members by non-pending & majority team
    if include_pending:
        current_snapshot_filtered = latest_snapshot[
            (latest_snapshot['Team'] == majority_team)
        ].copy()
    else:
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
            details = details[[
                "Ruler Name", "Resource 1+2", "Alliance", "Team",
                "Days Old", "Nation Drill Link", "Activity", "Alliance Status"
            ]].sort_values(
                by="Ruler Name",
                key=lambda col: col.str.lower()
            ).reset_index(drop=True)
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

        # ——— Input Trade Circles ———
        with st.expander("Input Trade Circles"):
            # — Inputs —
            trade_input  = st.text_area("Trade Circles (one Ruler Name per line)", height=200)
            filter_input = st.text_area("Filter Out Players (one per line)", height=100)

            # — Parse into blocks of names; blank lines split circles —
            blocks = trade_input.split("\n\n")
            records = []
            for i, blk in enumerate(blocks):
                names = [ line.strip() for line in blk.splitlines() if line.strip() ]
                for name in names:
                    records.append({
                        "Trade Circle": i+1,
                        "Ruler Name":   name,
                        "merge_key":    name.lower()
                    })
            tc_df = pd.DataFrame(records, columns=["Trade Circle","Ruler Name","merge_key"])

            # — if user left it blank, use the 'details' table to auto-build circles of 6 —
            if tc_df.empty:
                st.info("No manual Trade Circles provided; defaulting to all eligible nations from ‘Nation Details’ in groups of 6.")
                tmp = details.copy()[[
                    "Ruler Name","Resource 1+2","Alliance","Team",
                    "Days Old","Nation Drill Link","Activity"
                ]].reset_index(drop=True)
                tmp["Trade Circle"] = (tmp.index // 6) + 1
                tc_df = tmp

            # — if after fallback it’s still empty, warn and bail —
            if tc_df.empty:
                st.warning("No Trade Circles could be formed.")
            else:
                # — Prepare snapshot with same merge_key & combined resources —
                if "merge_key" not in tc_df.columns:
                    # fallback path already has full columns
                    tc_df = tc_df.copy()
                else:
                    snap = latest_snapshot.copy()
                    snap["merge_key"]    = snap["Ruler Name"].str.lower()
                    snap["Resource 1+2"] = snap.apply(get_resource_1_2, axis=1)
                    tc_df = tc_df.merge(
                        snap[["merge_key","Resource 1+2","Alliance","Team","Created","Nation ID"]],
                        on="merge_key", how="inner"
                    )
                    tc_df["Created"] = pd.to_datetime(tc_df["Created"], errors="coerce")
                    tc_df["Days Old"]= (pd.Timestamp.now() - tc_df["Created"]).dt.days
                    tc_df["Nation Drill Link"] = (
                        "https://www.cybernations.net/nation_drill_display.asp?Nation_ID="
                        + tc_df["Nation ID"].astype(str)
                    )
                    tc_df["Activity"] = tc_df["Ruler Name"].map(
                        avg_activity.set_index("Ruler Name")["All Time Average Days of Inactivity"]
                    )
                    # filter & drop merge_key if we had it
                    if "merge_key" in tc_df.columns:
                        filter_set = {ln.strip().lower() for ln in filter_input.splitlines() if ln.strip()}
                        tc_df = (
                            tc_df[
                                (tc_df["Activity"] < 14) &
                                (tc_df["Alliance"] == selected_alliance) &
                                (tc_df["Team"]     == majority_team) &
                                (~tc_df["merge_key"].isin(filter_set))
                            ]
                            .drop(columns="merge_key")
                            .reset_index(drop=True)
                        )

                # — Processed Trade Circles —
                st.markdown("#### Processed Trade Circles")
                tc_df = tc_df.reset_index(drop=True)
                tc_df.index += 1
                st.dataframe(tc_df[[
                    "Trade Circle","Ruler Name","Resource 1+2",
                    "Alliance","Team","Days Old","Nation Drill Link","Activity"
                ]])
                # … rest of the original code for Unmatched … 

        # ——— Updated Trade Circles ———
        with st.expander("Updated Trade Circles"):
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

                # tag base and unmatched with levels
                pm_base = tc_df.copy()
                pm_base['Peace Mode Level']     = pm_base['Days Old'].apply(peace_level)
                pm_unmatched = details[~details['Ruler Name'].isin(pm_base['Ruler Name'])].copy()
                pm_unmatched['Peace Mode Level'] = pm_unmatched['Days Old'].apply(peace_level)

                final_records = []

                # build full circles per level, including any partials
                for level in ['Level A', 'Level B', 'Level C']:
                    lvl_base = pm_base[pm_base['Peace Mode Level'] == level].copy()
                    lvl_un   = pm_unmatched[pm_unmatched['Peace Mode Level'] == level].copy()

                    # fill incomplete circles first
                    sizes      = lvl_base.groupby('Trade Circle').size() if not lvl_base.empty else pd.Series(dtype=int)
                    incomplete = {c: 6 - cnt for c, cnt in sizes.items() if cnt < 6}
                    for circle_id, slots in sorted(incomplete.items(), key=lambda x: x[1]):
                        to_add = min(slots, len(lvl_un))
                        adds   = lvl_un.iloc[:to_add]
                        lvl_un  = lvl_un.iloc[to_add:]
                        for _, row in adds.iterrows():
                            rec = row.to_dict()
                            rec['Trade Circle'] = circle_id
                            final_records.append(rec)

                    # keep original full members
                    for _, row in lvl_base.iterrows():
                        final_records.append(row.to_dict())

                    # form brand-new full circles
                    max_id      = int(lvl_base['Trade Circle'].max()) if not lvl_base.empty else 0
                    next_circle = max_id + 1
                    while len(lvl_un) >= 6:
                        group   = lvl_un.iloc[:6]
                        lvl_un  = lvl_un.iloc[6:]
                        for _, row in group.iterrows():
                            rec = row.to_dict()
                            rec['Trade Circle'] = next_circle
                            final_records.append(rec)
                        next_circle += 1

                    # assign any leftover (<6) into one final circle
                    if len(lvl_un) > 0:
                        for _, row in lvl_un.iterrows():
                            rec = row.to_dict()
                            rec['Trade Circle'] = next_circle
                            final_records.append(rec)
                        lvl_un = lvl_un.iloc[0:0]  # clear

                # assemble final_df
                final_df = pd.DataFrame(final_records)
                level_order = {'Level A':0, 'Level B':1, 'Level C':2}
                final_df = final_df.sort_values(
                    ['Peace Mode Level','Trade Circle','Ruler Name'],
                    key=lambda col: col.map(level_order) if col.name=='Peace Mode Level' else col
                ).reset_index(drop=True)
                final_df.index += 1

                # now optimize with PuLP (unchanged)
                try:
                    import pulp
                except ImportError:
                    st.error(
                        "🚨 *PuLP* is not installed. "
                        "Add `pulp` to your dependencies and redeploy."
                    )
                else:
                    import math

                    optimal_records = []
                    for level in ['Level A', 'Level B', 'Level C']:
                        df_lvl = final_df[final_df['Peace Mode Level'] == level].copy()
                        if df_lvl.empty:
                            continue

                        nations      = df_lvl['Ruler Name'].tolist()
                        orig_circle  = dict(zip(nations, df_lvl['Trade Circle']))
                        existing_cs  = sorted(df_lvl['Trade Circle'].unique())
                        total_n      = len(nations)
                        needed       = math.ceil(total_n / 6)
                        max_exist    = existing_cs[-1] if existing_cs else 0
                        new_cs_count = max(needed - len(existing_cs), 0)
                        all_cs       = existing_cs + list(range(max_exist+1, max_exist+1+new_cs_count))

                        prob = pulp.LpProblem(f"TradeCircle_{level}", pulp.LpMaximize)
                        x = pulp.LpVariable.dicts("x", ((p,c) for p in nations for c in all_cs), cat='Binary')
                        y = pulp.LpVariable.dicts("y", all_cs, cat='Binary')

                        for p in nations:
                            prob += pulp.lpSum(x[p,c] for c in all_cs) == 1
                        for c in all_cs:
                            prob += pulp.lpSum(x[p,c] for p in nations) <= 6*y[c]
                            for p in nations:
                                prob += x[p,c] <= y[c]
                        prob += pulp.lpSum(y[c] for c in all_cs) <= needed

                        flow = pulp.lpSum(x[p,c] for p in nations for c in all_cs)
                        penal = pulp.lpSum(y[c] for c in all_cs)
                        reassign = pulp.lpSum(
                            x[p,c] * (0 if c==orig_circle[p] else (1 if c in existing_cs else 6))
                            for p in nations for c in all_cs
                        )
                        prob += 1000*flow - 10*penal - reassign
                        prob.solve(pulp.PULP_CBC_CMD(msg=False))

                        for p in nations:
                            for c in all_cs:
                                if pulp.value(x[p,c]) == 1:
                                    row = df_lvl[df_lvl['Ruler Name']==p].iloc[0].to_dict()
                                    row['Trade Circle'] = int(c)
                                    optimal_records.append(row)
                                    break
        
                    # build opt_df
                    opt_df = pd.DataFrame(optimal_records)
                    renumbered = []
                    for lvl, grp in opt_df.groupby('Peace Mode Level', sort=False):
                        ids = sorted(grp['Trade Circle'].unique())
                        id_map = {old:new for new,old in enumerate(ids,1)}
                        tmp = grp.copy(); tmp['Trade Circle'] = tmp['Trade Circle'].map(id_map)
                        renumbered.append(tmp)
                    opt_df = pd.concat(renumbered, ignore_index=True)
        
                    opt_df = opt_df.sort_values(
                        ['Peace Mode Level','Trade Circle','Ruler Name'],
                        key=lambda col: (
                            col.map(level_order) if col.name=='Peace Mode Level'
                            else col if col.name=='Trade Circle'
                            else col.str.lower()
                        )
                    ).reset_index(drop=True)
                    opt_df.index += 1
        
                    st.markdown("##### Optimal Trade Circles")
                    st.dataframe(opt_df[[
                        "Peace Mode Level","Trade Circle","Ruler Name",
                        "Resource 1+2","Alliance","Team",
                        "Days Old","Nation Drill Link","Activity"
                    ]])
        
                    st.markdown("##### Players Left Over")
                    assigned = set(opt_df['Ruler Name'])
                    leftovers = final_df[~final_df['Ruler Name'].isin(assigned)].copy()
                    if leftovers.empty:
                        st.markdown("_No unmatched players remain._")
                    else:
                        leftovers.index = range(1, len(leftovers)+1)
                        st.dataframe(leftovers[[
                            "Ruler Name","Resource 1+2","Alliance","Team",
                            "Days Old","Nation Drill Link","Activity"
                        ]])

        # ——— Assign Peacetime Recommended Resources ———
        with st.expander("Assign Peacetime Recommended Resources"):
            # only run if we have optimized circles
            if 'opt_df' not in locals() or opt_df.empty:
                st.markdown("_No processed Trade Circles to assign resources. "
                            "Please fill in the Input Trade Circles section above._")
            else:
                import numpy as np
                from scipy.optimize import linear_sum_assignment
                from collections import Counter
        
                def find_best_match(current, combos):
                    best, best_score = None, float("inf")
                    for combo in combos:
                        missing = set(combo) - set(current)
                        extra   = set(current) - set(combo)
                        score   = len(missing) + len(extra)
                        if score < best_score:
                            best_score = score
                            best       = combo
                    return best
        
                valid_combos = {
                    "Level A": [ [r.strip() for r in line.split(",")] for line in peace_a_text.splitlines() if line.strip() ],
                    "Level B": [ [r.strip() for r in line.split(",")] for line in peace_b_text.splitlines() if line.strip() ],
                    "Level C": [ [r.strip() for r in line.split(",")] for line in peace_c_text.splitlines() if line.strip() ],
                }
        
                rec_records = []
                for level in ["Level A", "Level B", "Level C"]:
                    lvl_df = opt_df[opt_df["Peace Mode Level"] == level]
                    if lvl_df.empty:
                        continue
        
                    for circle in sorted(lvl_df["Trade Circle"].unique()):
                        group = lvl_df[lvl_df["Trade Circle"] == circle].reset_index(drop=True)
        
                        # 1) collect all current resources
                        all_res = []
                        for s in group["Resource 1+2"]:
                            all_res += [r.strip() for r in s.split(",") if r.strip()]
                        current_sorted = sorted(set(all_res))
        
                        # 2) pick best 12‑resource combo
                        best_combo = find_best_match(current_sorted, valid_combos[level])
                        combo_str  = ", ".join(best_combo)
        
                        # 3) lock‑in duplicates using a Counter
                        avail = Counter(best_combo)
                        fixed = {}
                        rem_players = []
                        for _, row in group.iterrows():
                            ruler = row["Ruler Name"]
                            curr   = tuple(r.strip() for r in row["Resource 1+2"].split(","))
                            if avail[curr[0]] > 0 and avail[curr[1]] > 0:
                                # lock this player at their current pair
                                fixed[ruler] = row["Resource 1+2"]
                                avail[curr[0]] -= 1
                                avail[curr[1]] -= 1
                            else:
                                rem_players.append(ruler)
        
                        # 4) build slices from remaining resources
                        rem_resources = list(avail.elements())
                        m = len(rem_players)
                        slices = [ rem_resources[2*i:2*i+2] for i in range(m) ]
        
                        # 5) Hungarian on remaining players
                        cost = np.zeros((m, m), dtype=int)
                        for i, ruler in enumerate(rem_players):
                            curr = sorted(r for r in group.loc[group["Ruler Name"]==ruler, "Resource 1+2"].iloc[0].split(","))
                            for j, sl in enumerate(slices):
                                common = set(curr).intersection(set(sl))
                                cost[i,j] = 2 - len(common)
                        rows, cols = linear_sum_assignment(cost)
        
                        # 6a) record fixed players (now listing their actual pair)
                        for ruler, pair in fixed.items():
                            row = group[group["Ruler Name"] == ruler].iloc[0]
                            rec_records.append({
                                "Peace Mode Level": level,
                                "Trade Circle":     circle,
                                "Ruler Name":       ruler,
                                "Current Resource 1+2": row["Resource 1+2"],
                                "Alliance":         row["Alliance"],
                                "Team":             row["Team"],
                                "Days Old":         row["Days Old"],
                                "Nation Drill Link":row["Nation Drill Link"],
                                "Activity":         row["Activity"],
                                "Assigned Resource 1+2": pair,
                                "Assigned Valid Resource Combination": combo_str
                            })
        
                        # 6b) record Hungarian assignments
                        for i, j in zip(rows, cols):
                            ruler = rem_players[i]
                            sl    = slices[j]
                            row   = group[group["Ruler Name"] == ruler].iloc[0]
                            rec_records.append({
                                "Peace Mode Level": level,
                                "Trade Circle":     circle,
                                "Ruler Name":       ruler,
                                "Current Resource 1+2": row["Resource 1+2"],
                                "Alliance":         row["Alliance"],
                                "Team":             row["Team"],
                                "Days Old":         row["Days Old"],
                                "Nation Drill Link":row["Nation Drill Link"],
                                "Activity":         row["Activity"],
                                "Assigned Resource 1+2": f"{sl[0]}, {sl[1]}",
                                "Assigned Valid Resource Combination": combo_str
                            })
        
                rec_df = pd.DataFrame(rec_records).sort_values(
                    ["Peace Mode Level", "Trade Circle", "Ruler Name"],
                    key=lambda col: (
                        col.map(level_order) if col.name == "Peace Mode Level"
                        else col if col.name == "Trade Circle"
                        else col.str.lower()
                    )
                ).reset_index(drop=True)
                rec_df.index += 1
                # round Activity to 1 decimal place
                rec_df["Activity"] = rec_df["Activity"].round(1)

                # define which columns to show
                columns = [
                    "Peace Mode Level", "Trade Circle", "Ruler Name",
                    "Current Resource 1+2", "Alliance", "Team", "Days Old",
                    "Nation Drill Link", "Activity",
                    "Assigned Resource 1+2", "Assigned Valid Resource Combination"
                ]

                styled_rec = (
                    rec_df[columns]
                      .style
                      .format({"Activity": "{:.1f}"})
                      .set_properties(
                          subset=["Assigned Valid Resource Combination"],
                          **{
                              "white-space": "normal",
                              "max-width": "1200px",
                              "text-align": "left"
                          }
                      )
                )

                st.markdown("##### Assign Peacetime Recommended Resources")
                st.dataframe(styled_rec, use_container_width=True)
                
                # Copy‐to‐Clipboard for Peacetime
                peace_csv = rec_df.to_csv(index=False)
                components.html(
                    f"""
                    <textarea id="peace-data" style="display:none;">{peace_csv}</textarea>
                    <button
                      onclick="navigator.clipboard.writeText(document.getElementById('peace-data').value)"
                      style="margin-top:10px; padding:4px 8px;"
                    >
                      Copy Peacetime Table to Clipboard
                    </button>
                    """,
                    height=60,
                )

        # ——— Assign Wartime Recommended Resources ———
        with st.expander("Assign Wartime Recommended Resources"):
            # only run if we have optimized Peace‐Mode circles
            if 'opt_df' not in locals() or opt_df.empty:
                st.markdown("_No Trade Circles to process. Please add entries in the Input Trade Circles section above._")
            else:
                import numpy as np
                from scipy.optimize import linear_sum_assignment
                from collections import Counter
        
                def find_best_match(current, combos):
                    best, best_score = None, float("inf")
                    for combo in combos:
                        missing = set(combo) - set(current)
                        extra   = set(current) - set(combo)
                        score   = len(missing) + len(extra)
                        if score < best_score:
                            best_score = score
                            best       = combo
                    return best
        
                # parse War Mode valid 12‑resource combos
                war_combos = [
                    [r.strip() for r in line.split(",")]
                    for line in war_text.splitlines() if line.strip()
                ]
        
                war_records = []
                # loop by Peace Mode Level → Trade Circle to avoid mixing circles across levels
                for level in ["Level A", "Level B", "Level C"]:
                    lvl_df = opt_df[opt_df["Peace Mode Level"] == level]
                    if lvl_df.empty:
                        continue
                    for circle in sorted(lvl_df["Trade Circle"].unique()):
                        group = lvl_df[lvl_df["Trade Circle"] == circle].reset_index(drop=True)
        
                        # gather all current resources in the circle
                        all_res = []
                        for s in group["Resource 1+2"]:
                            all_res += [r.strip() for r in s.split(",") if r.strip()]
                        current_sorted = sorted(set(all_res))
        
                        # pick best‐matching 12‐combo for War Mode
                        best_combo = find_best_match(current_sorted, war_combos)
                        combo_str  = ", ".join(best_combo)
        
                        # lock in exact‐matches first, using a Counter for availability
                        avail = Counter(best_combo)
                        fixed = {}
                        rem_players = []
                        for _, row in group.iterrows():
                            ruler     = row["Ruler Name"]
                            curr_pair = tuple(r.strip() for r in row["Resource 1+2"].split(","))
                            if avail[curr_pair[0]] > 0 and avail[curr_pair[1]] > 0:
                                fixed[ruler] = row["Resource 1+2"]
                                avail[curr_pair[0]] -= 1
                                avail[curr_pair[1]] -= 1
                            else:
                                rem_players.append(ruler)
        
                        # build slices for the remaining players
                        rem_resources = list(avail.elements())
                        m = len(rem_players)
                        slices = [ rem_resources[2*i:2*i+2] for i in range(m) ]
        
                        # build cost matrix and solve Hungarian
                        cost = np.zeros((m, m), dtype=int)
                        for i, ruler in enumerate(rem_players):
                            curr = sorted(r for r in group.loc[group["Ruler Name"]==ruler, "Resource 1+2"].iloc[0].split(","))
                            for j, sl in enumerate(slices):
                                common = set(curr).intersection(sl)
                                cost[i, j] = 2 - len(common)
                        rows, cols = linear_sum_assignment(cost)
        
                        # record locked‐in players
                        for ruler, pair in fixed.items():
                            row = group[group["Ruler Name"] == ruler].iloc[0]
                            war_records.append({
                                "Peace Mode Level": level,
                                "Trade Circle":                      circle,
                                "Ruler Name":                        ruler,
                                "Current Resource 1+2":              row["Resource 1+2"],
                                "Alliance":                          row["Alliance"],
                                "Team":                              row["Team"],
                                "Days Old":                          row["Days Old"],
                                "Nation Drill Link":                 row["Nation Drill Link"],
                                "Activity":                          row["Activity"],
                                "Assigned Resource 1+2":             pair,
                                "Assigned Valid Resource Combination": combo_str
                            })
        
                        # record Hungarian‐assigned players
                        for i, j in zip(rows, cols):
                            ruler   = rem_players[i]
                            sl      = slices[j]
                            assigned = f"{sl[0]}, {sl[1]}"
                            row     = group[group["Ruler Name"] == ruler].iloc[0]
                            war_records.append({
                                "Peace Mode Level": level,
                                "Trade Circle":                      circle,
                                "Ruler Name":                        ruler,
                                "Current Resource 1+2":              row["Resource 1+2"],
                                "Alliance":                          row["Alliance"],
                                "Team":                              row["Team"],
                                "Days Old":                          row["Days Old"],
                                "Nation Drill Link":                 row["Nation Drill Link"],
                                "Activity":                          row["Activity"],
                                "Assigned Resource 1+2":             assigned,
                                "Assigned Valid Resource Combination": combo_str
                            })
        
                # build and display the War Mode DataFrame
                war_df = pd.DataFrame(war_records).sort_values(
                    ["Peace Mode Level", "Trade Circle", "Ruler Name"],
                    key=lambda col: (
                        col.map(level_order) if col.name == "Peace Mode Level"
                        else col if col.name == "Trade Circle"
                        else col.str.lower()
                    )
                ).reset_index(drop=True)
                war_df.index += 1
                # round Activity to 1 decimal place
                war_df["Activity"] = war_df["Activity"].round(1)

                columns = [
                    "Peace Mode Level", "Trade Circle", "Ruler Name",
                    "Current Resource 1+2", "Alliance", "Team", "Days Old",
                    "Nation Drill Link", "Activity",
                    "Assigned Resource 1+2", "Assigned Valid Resource Combination"
                ]

                styled_war = (
                    war_df[columns]
                      .style
                      .format({"Activity": "{:.1f}"})
                      .set_properties(
                          subset=["Assigned Valid Resource Combination"],
                          **{
                              "white-space": "normal",
                              "max-width": "1200px",
                              "text-align": "left"
                          }
                      )
                )

                st.markdown("##### Assign Wartime Recommended Resources")
                st.dataframe(styled_war, use_container_width=True)

                # Copy‐to‐Clipboard for Wartime
                war_csv = war_df.to_csv(index=False)
                components.html(
                    f"""
                    <textarea id="war-data" style="display:none;">{war_csv}</textarea>
                    <button
                      onclick="navigator.clipboard.writeText(document.getElementById('war-data').value)"
                      style="margin-top:10px; padding:4px 8px;"
                    >
                      Copy Wartime Table to Clipboard
                    </button>
                    """,
                    height=60,
                )

        # ——— Download everything as a single XLSX (with auto‑fit columns) ———
        if 'rec_df' in locals() and 'war_df' in locals():
            buffer = io.BytesIO()
            # use xlsxwriter so we can set column widths
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                rec_df.to_excel(writer, sheet_name="Peacetime", index=False)
                war_df.to_excel(writer, sheet_name="Wartime",    index=False)

                workbook    = writer.book
                peac_sheet  = writer.sheets["Peacetime"]
                war_sheet   = writer.sheets["Wartime"]

                # auto‑fit Peacetime columns
                for idx, col in enumerate(rec_df.columns):
                    max_len = max(
                        rec_df[col].astype(str).map(len).max(),
                        len(col)
                    ) + 2
                    peac_sheet.set_column(idx, idx, max_len)

                # auto‑fit Wartime columns
                for idx, col in enumerate(war_df.columns):
                    max_len = max(
                        war_df[col].astype(str).map(len).max(),
                        len(col)
                    ) + 2
                    war_sheet.set_column(idx, idx, max_len)

            buffer.seek(0)

            # build filename: AllianceName_Optimized_Trade_Circles_YYYY-MM-DD.xlsx
            alliance_safe = selected_alliance.replace(" ", "_")
            date_str      = latest_date.strftime("%Y-%m-%d")
            download_name = f"{alliance_safe}_Optimized_Trade_Circles_{date_str}.xlsx"

            st.download_button(
                "Download All Data as Excel",
                data=buffer,
                file_name=download_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Includes Peacetime & Wartime sheets"
            )

if __name__ == "__main__":
    main()
