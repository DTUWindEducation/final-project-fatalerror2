"""
Check functions works as expected.
"""
import os
import sys
from pathlib import Path

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *

def test_load_site_data():
    """Check the load_site_data function."""
    # given
    site_index = 1
    power_exp = 0.1635 # expected value of power
    # when
    df, _ = load_site_data(site_index)
    # then
    assert np.isclose(df['Power'][0], power_exp)  # check power value
