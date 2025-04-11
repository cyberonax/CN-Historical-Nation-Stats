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
      date_token = "452025" which is interpreted as:
        month = 4, day = 5, year = 2025.
    Returns a datetime object on success, otherwise None.
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
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month,
