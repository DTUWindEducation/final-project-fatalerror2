import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from inputs.location import Location

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



