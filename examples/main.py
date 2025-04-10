#Import Packages
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- User Inputs ---
variable_name = "Power"       # e.g., "temperature_2m", "Power", etc.
site_index = 2                # 1 to 4
starting_time = "2018-01-01"  # Included; YYYY-MM-DD
ending_time = "2018-01-31"    #Excluded

# --- File Paths ---
script_dir = Path(__file__).resolve().parent
inputs_dir = script_dir.parent / 'inputs'

site_files = {
    1: inputs_dir / 'Location1.csv',
    2: inputs_dir / 'Location2.csv',
    3: inputs_dir / 'Location3.csv',
    4: inputs_dir / 'Location4.csv',
}

# --- Function 1: Load and filter by site ---
def load_and_filter_by_site(site_files, site_index):
    dfs = []
    for idx, path in site_files.items():
        print(f"Trying to load: {path}")
        df = pd.read_csv(path)
        df['Site'] = f'Location{idx}'
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    site_to_process = f'Location{site_index}'
    site_df = combined_df[combined_df['Site'] == site_to_process].copy()
    site_df['Time'] = pd.to_datetime(site_df['Time'])
    return site_df, site_to_process

# --- Function 2: Filter by time and plot selected variable ---
def filter_and_plot(site_df, variable_name, start_time, end_time, site_label):
    filtered_df = site_df[
        (site_df['Time'] >= pd.to_datetime(start_time)) &
        (site_df['Time'] <= pd.to_datetime(end_time))
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['Time'], filtered_df[variable_name], label=variable_name)

    plt.title(f"{variable_name} at {site_label}")
    plt.xlabel('Time')
    plt.ylabel(variable_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Function 4: Split the dataset into training dataset and test dataset. ---
def split_site_data(site_df, test_size=0.2, random_state=42):
    """
    Splits the full site DataFrame (before time filtering) into training and test sets.
    Shuffle is False to preserve chronological order (important for time series).
    """
    train_df, test_df = train_test_split(
        site_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=False  # Keep time order for time series modeling
    )
    return train_df, test_df





# --- Main Execution ---
site_df, site_name = load_and_filter_by_site(site_files, site_index)
filter_and_plot(site_df, variable_name, starting_time, ending_time, site_name)

