import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Location:
    """
    A class to represent a location with weather data and power output.
    """

    def __init__(self, Time,temperature_2m,relativehumidity_2m,dewpoint_2m,
                 windspeed_10m,windspeed_100m,winddirection_10m,
                 winddirection_100m,windgusts_10m,Power):
        self.Time = Time
        self.temperature_2m = temperature_2m
        self.relativehumidity_2m = relativehumidity_2m
        self.dewpoint_2m = dewpoint_2m
        self.windspeed_10m = windspeed_10m
        self.windspeed_100m = windspeed_100m
        self.winddirection_10m = winddirection_10m
        self.winddirection_100m = winddirection_100m
        self.windgusts_10m = windgusts_10m
        self.Power = Power

    def plot_power(self, starting_time, ending_time, site_index):
        """
        Function to load the data from the 4 locations and plot the power output
        from a chosen location, starting time and ending time."""

        file_path = Path(__file__).parents[1] / "inputs"/f"Location{site_index}.csv"

        df = pd.read_csv(file_path, sep= ',')
        df['Time'] = pd.to_datetime(df['Time'])
        starting_time = pd.to_datetime(starting_time)
        ending_time = pd.to_datetime(ending_time)

        plt.figure(figsize=(10, 5))
        plt.plot(df['Time'], df['Power'])
        plt.xlim(starting_time, ending_time)
        plt.title(f"Power Output for Location {site_index}")
        plt.xlabel("Time")
        plt.ylabel("Power Output (kW)")
        plt.show()

#Example usage
site = Location(
    Time = None,
    temperature_2m = None,
    relativehumidity_2m = None,
    dewpoint_2m = None,
    windspeed_10m = None,
    windspeed_100m = None,
    winddirection_10m = None,
    winddirection_100m = None,
    windgusts_10m = None,
    Power = None
)

site.plot_power("2019-01-01 00:00:00", "2019-01-02 00:00:00", 1)



