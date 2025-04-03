import numpy as np
import matplotlib.pyplot as plt

class GeneralWindTurbine:
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name=None):
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.v_in = v_in
        self.v_rated = v_rated
        self.v_out = v_out
        self.name = name

    def get_power(self,v):
        power_array = []
        for item in v:
            if item < self.v_in or item > self.v_out:
                power_array.append(0)
            elif item < self.v_rated:
                power_array.append(self.rated_power * (item / self.v_rated) ** 3)
            elif item <= self.v_out:
                power_array.append(self.rated_power)
            else:
                power_array.append(0)
        return power_array
        

class WindTurbine(GeneralWindTurbine):
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, power_curve_data, name=None):
        super().__init__(rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name)
        self.power_curve_data = power_curve_data

    def get_power(self, v):
        power_array = []
        for item in v:
            if item < self.v_in or item > self.v_out:
                power_array.append(0)
            else:
                windspeeds = self.power_curve_data[:, 0]
                powers = self.power_curve_data[:, 1]
                power_array.append(np.interp(item, windspeeds, powers))
        return power_array


# Create a GeneralWindTurbine object
gwt = GeneralWindTurbine(
    rotor_diameter=164,
    hub_height=110,
    rated_power=8000,
    v_in=4,
    v_rated=12.5,
    v_out=25,
    name="Leanwind 8 MW RWT")

# Create a WindTurbine object with power curve data
wt = WindTurbine(
    rotor_diameter=164,
    hub_height=110,
    rated_power=8000,
    v_in=4,
    v_rated=12.5,
    v_out=25,
    power_curve_data=np.loadtxt("./homework/LEANWIND_Reference_8MW_164.csv",skiprows=1,delimiter=','),
    name="Leanwind 8 MW RWT")


# Example usage
windspeed_array = np.linspace(0, 30, 1000)
gwt_power = gwt.get_power(windspeed_array)
wt_power = wt.get_power(windspeed_array)

# plot and compare the two power curves
plt.figure(figsize=(10, 6))
plt.plot(windspeed_array, gwt_power, label="General Wind Turbine")
plt.plot(windspeed_array, wt_power, "--", label="Wind Turbine")
plt.title("Power Output vs Wind Speed")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power Output (kW)")
plt.legend()
plt.grid()
plt.show()
