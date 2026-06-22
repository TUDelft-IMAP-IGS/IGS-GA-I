import numpy as np


def dp_3_bird_mortality(var1):  # Design performance function (bird mortality per day)
    if 8 < var1 < 9:
        dp_3 = 800
        return dp_3
    else:
        dp_3 = 200
        return dp_3


def objective_profit(variables):
    """WTP=P*EP-DC - Wind turbine profit per day
    WTP — Wind turbine profit (per day)
    P — Wind turbine generated power (kWh/day)
    EP — Price of electricity (per kWh)
    DC — Wind turbine cost (per day)
    """
    var1 = variables[0]
    var2 = variables[1]
    var3 = variables[2]
    var4 = variables[3]

    rho = 1.225  # kg/m3
    e = 0.3  # efficiency
    blade_diameter = var3 * 1.3  # m
    A = np.pi * (blade_diameter / 2) ** 2  # m/s
    P = 0.5 * A * rho * e * (var4 ** 3)  # Watt
    P = P * 24 / 1000  # kWh
    EP = 0.01
    WTP = P * EP  # DC is neglected for now
    # print(f'WTP: {WTP}')
    resistance = (var3 + var4) / 12
    # print(f'z:{resistance *var1}')
    # profit = (wind turbine profit / distance) * number of turbines
    profit = (WTP * var2) - (resistance * var1)  # old: (WTP / var1) * var2

    return profit


def objective_noise(variables):
    var1 = variables[0]
    var2 = variables[1]
    var3 = variables[2]
    var4 = variables[3]

    L_W = 104  # Example sound power level of a turbine in dB

    # Adjustments
    wind_base_speed = 3  # Base wind speed for no adjustment
    height_base = 50  # Base height for no adjustment

    # Wind speed adjustment: Increase 0.2 dB per m/s over base
    wind_adj = np.maximum(0, var4 - wind_base_speed) * 0.2

    # Height adjustment: Increases 0.1 dB per meter over base height
    height_adj = np.maximum(0, var3 - height_base) * 0.1

    # Calculate the noise level for a single turbine using the inverse square law
    L_d_single = L_W + wind_adj + height_adj - 20 * np.log10(var1 * 1000 / 1)

    # Calculate the noise level for a single turbine using the inverse square law
    # L_d_single = L_W - 20 * np.log10(var1*1000/1)

    # Convert the single turbine noise level to power
    power_single = 10 ** (L_d_single / 10)

    # Calculate total power for all turbines
    total_power = var2 * power_single

    # Convert total power back to dB
    L_total = 10 * np.log10(total_power)

    return L_total


def objective_mortality(variables):
    var1 = variables[0]
    var2 = variables[1]
    var3 = variables[2]
    var4 = variables[3]

    blade_diameter = var3 * 1.3  # m
    A = np.pi * (blade_diameter / 2) ** 2  # m/s

    # bird_mortality = list(map(dp_3_bird_mortality, var1))
    bird_mortality = dp_3_bird_mortality(var1)

    mortality = bird_mortality * var2 * (A / 30000) * 0.005

    return mortality


def objective_particle(variables):
    var1 = variables[0]
    var2 = variables[1]
    var3 = variables[2]
    var4 = variables[3]

    blade_diameter = var3 * 1.3  # m
    A = np.pi * (blade_diameter / 2) ** 2  # m/s

    wind_speed_impact = (var4 ** 3)  # Coefficient for the wind speed's impact

    # Calculate the erosion rate
    particle_pollution = wind_speed_impact * (A / 30000) * var2 / 1000

    return particle_pollution
