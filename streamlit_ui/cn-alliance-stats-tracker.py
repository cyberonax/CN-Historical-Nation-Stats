import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import re
from datetime import datetime, timedelta

# -----------------------
# HELPER FUNCTION: Combine Resource 1 and Resource 2
# -----------------------
def get_resource_1_2(row):
    """
    Combine the values from "Resource 1" and "Resource 2" into a single string.
    If either resource is missing, show only the available one.
    """
    r1 = row.get("Resource 1", "")
    r2 = row.get("Resource 2", "")
    r1 = str(r1).strip() if pd.notnull(r1) else ""
    r2 = str(r2).strip() if pd.notnull(r2) else ""
    if r1 and r2:
        return f"{r1}, {r2}"
    elif r1:
        return r1
    elif r2:
        return r2
    return ""

# -----------------------
# DOWNLOAD & DATA LOADING FUNCTIONS
# -----------------------
def download_and_extract_zip(url):
    """Download a zip file from the given URL and extract its first file as a DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_list = z.namelist()
            if not file_list:
                return None
            file_name = file_list[0]
            with z.open(file_name) as file:
                # Adjust delimiter and encoding as needed
                df = pd.read_csv(file, delimiter="|", encoding="ISO-8859-1")
                return df
    except Exception:
        return None

def load_data():
    """Try downloading data using a list of dates and URL patterns without showing debug messages."""
    today = datetime.now()
    base_url = "https://www.cybernations.net/assets/CyberNations_SE_Nation_Stats_"
    dates_to_try = [today, today - timedelta(days=1), today + timedelta(days=1)]
    
    for dt in dates_to_try:
        date_str = f"{dt.month}{dt.day}{dt.year}"
        url1 = base_url + date_str + "510001.zip"
        url2 = base_url + date_str + "510002.zip"
        
        df = download_and_extract_zip(url1)
        if df is None:
            df = download_and_extract_zip(url2)
        if df is not None:
            st.success(f"Data loaded successfully from date: {date_str}")
            return df
    return None

# -----------------------
# MAIN APP
# -----------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Cyber Nations | Nation Ruler Search")
    
    # Brief description under the main title
    st.markdown("This tool helps you simplify returning information from your list of pasted Nation Rulers. Click 'Download Nation Statistics' to proceed.")
    
    # Section: Download Nation Statistics
    if st.button("Download Nation Statistics"):
        with st.spinner("Loading data..."):
            df = load_data()
        if df is not None:
            st.session_state.df = df
        else:
            st.error("Failed to load data.")
    
    # Section: Ruler Search Interface
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        
        st.subheader("Enter Ruler or Nation Names (one per line)")
        names_input = st.text_area("Paste the names here", height=150)
        
        if st.button("Search"):
            # Convert input to a list (ignoring extra whitespace and empty lines)
            filters = [name.strip() for name in names_input.splitlines() if name.strip()]
            if not filters:
                st.info("No names entered. Please paste one or more names.")
            else:
                # Convert filters to lowercase for a case-insensitive search.
                lower_filters = [f.lower() for f in filters]
                # Create a mask where either the Ruler Name or Nation Name column matches any input.
                mask = df["Ruler Name"].str.lower().isin(lower_filters) | df["Nation Name"].str.lower().isin(lower_filters)
                result_df = df[mask].copy()
                
                if result_df.empty:
                    st.info("No matching entries found. Check your input for spelling or extra spaces.")
                else:
                    # Calculate the Resource 1+2 column.
                    result_df["Resource 1+2"] = result_df.apply(get_resource_1_2, axis=1)
                    # Build the Nation Drill Link.
                    result_df["Nation Drill Link"] = (
                        "https://www.cybernations.net/nation_drill_display.asp?Nation_ID=" +
                        result_df["Nation ID"].astype(str)
                    )
                    # Reorder columns: Nation ID first.
                    display_df = result_df[["Nation ID", "Ruler Name", "Resource 1+2", "Alliance", "Team", "Nation Drill Link"]]
                    
                    st.dataframe(display_df)
                    
                    # Provide a CSV download option.
                    csv = display_df.to_csv(index=False)
                    st.download_button("Download Results as CSV", csv, file_name="ruler_search_results.csv", mime="text/csv")
    
    st.markdown("---")
    
    # -----------------------
    # COLLAPSIBLE SECTION: Process Comma-Separated Names
    # -----------------------
    st.markdown("### Other Tools")
    with st.expander("Comma-Separated Name Processor"):
        st.markdown(
            """
            Paste a list of names, numbers, or other text below (separated by commas or new lines).

            - **Output 1:** Shows the names on separate lines.
            - **Output 2:** Shows each name wrapped in quotes with a trailing comma.
            - **Output 3:** Shows the names joined by a comma.
            """
        )
        
        names_input = st.text_area("Enter text", height=100)
        
        if names_input:
            # Split the input on commas or newlines using regex.
            names_list = [name.strip() for name in re.split(r"[,\n]+", names_input) if name.strip()]
            
            # Output 1: Each name on its own separate line.
            output1 = "\n".join(names_list)
            
            # Output 2: Each name on its separate line, wrapped in quotes and appended with a comma.
            output2 = "\n".join([f'"{name}",' for name in names_list])
            
            # Output 3: Names joined with a comma and a space.
            output3 = ", ".join(names_list)
            
            st.text_area("Output 1 (each name on a separate line)", value=output1, height=150)
            st.text_area("Output 2 (quoted names with trailing comma)", value=output2, height=150)
            st.text_area("Output 3 (names joined by a comma)", value=output3, height=100)

if __name__ == "__main__":
    main()
