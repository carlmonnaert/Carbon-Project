import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from numba import njit

# Initial conditions
Atmosphere_Initial = 750
CarbonateRock_Initial = 100000000
DeepOcean_Initial = 38000
FossilFuel_Initial = 7500
Plant_Initial = 560
Soil_Initial = 1500
SurfaceOcean_Initial = 890
VegLandArea_percent_Initial = 100

x0 = np.array([Atmosphere_Initial,
               CarbonateRock_Initial,
               DeepOcean_Initial,
               FossilFuel_Initial,
               Plant_Initial,
               Soil_Initial,
               SurfaceOcean_Initial,
               VegLandArea_percent_Initial
               ], dtype=float)

# Constants
Alk = 2.222446077610055
Kao = .278
SurfOcVol = .0362
Deforestation = 1.5

# Labels for the state variables, used for building output paths and plot titles
STATE_LABELS = [
    'atm',
    'rock',
    'deep',
    'fossil',
    'plant',
    'soil',
    'surf',
    'veg'
]

# Helper functions for formatting values in output paths and plot titles
def _fmt_value(value):
    if value == 0:
        return '0e0'
    sci = f'{float(value):.3e}'
    mantissa, exponent = sci.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.').replace('.', 'p')
    exponent = str(int(exponent))
    return f'{mantissa}e{exponent}'


def build_run_tag(initial_state):
    return '_'.join(f'{name}{_fmt_value(val)}' for name, val in zip(STATE_LABELS, initial_state))


def get_output_path(base_dir, initial_state):
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Helper functions for the carbon cycle model equations
# @njit
def AtmCO2(Atmosphere):
    return Atmosphere * (280/Atmosphere_Initial)
# @njit
def GlobalTemp(AtmCO2):
    return 15 + ((AtmCO2-280) * .01)
# @njit
def CO2Effect(AtmCO2):
    return 1.5 * ((AtmCO2) - 40) / ((AtmCO2) + 80)
# @njit
def WaterTemp(GlobalTemp):
    return 273+GlobalTemp
# @njit
def TempEffect(GlobalTemp):
    return ((60 - GlobalTemp) * (GlobalTemp + 15)) / (((60 + 15) / 2) ** (2))/.96
# @njit
def SurfCConc(SurfaceOcean):
    return (SurfaceOcean/12000)/SurfOcVol
# @njit
def Kcarb(WaterTemp):
    return .000575+(.000006*(WaterTemp-278))
# @njit
def KCO2(WaterTemp):
    return .035+(.0019*(WaterTemp-278))
# @njit
def HCO3(Kcarb, SurfCConc):
    denom = 1 - 4 * Kcarb
    if abs(denom) < 1e-10:
        return SurfCConc / 2  #in order to avoid dividing by zero 
    discriminant = SurfCConc**2 - Alk * (2*SurfCConc - Alk) * (1 - 4*Kcarb)
    if discriminant < 0:
        discriminant = 0  # avoid negative roots
    return (SurfCConc - np.sqrt(discriminant)) / denom
# @njit
def CO3(HCO3):
    return (Alk-HCO3)/2
# @njit
def pCO2Oc(KCO2, HCO3, CO3):
    return 280*KCO2*(HCO3**2/CO3)


# Fossil fuels
FossFuelData = np.array([[1850.0, 0.00], [1875.0, 0.30], [1900.0, 0.60], [1925.0, 1.35], [1950.0, 2.85], [1975.0, 4.95], [2000.0, 7.20], [2025.0, 10.05], [2050.0, 14.85], [2075.0, 20.70], [2100.0, 30.00]])
# CO2 equivalent of 10.05 Gt carbon is 36.88 Gt CO2


# @njit
def FossilFuelsCombustion(t):
    i = 0
    if t >= FossFuelData[-1,0]:
        return FossFuelData[-1,1]
    while i + 1 < len(FossFuelData) and t >= FossFuelData[i,0]:
        i = i + 1
    if i == 0:
        return FossFuelData[0,1]
    else:
        return FossFuelData[i-1,1] + (t - FossFuelData[i-1,0]) / (FossFuelData[i,0] - FossFuelData[i-1,0]) * (FossFuelData[i,1] - FossFuelData[i-1,1])

# @njit
def derivative(x, t):
    Atmosphere = x[0]
    CarbonateRock = x[1]
    DeepOcean = x[2]
    FossilFuelCarbon = x[3]
    Plants = x[4]
    Soils = x[5]
    SurfaceOcean = x[6]
    VegLandArea_percent = x[7]

    PlantResp = (Plants * (55/Plant_Initial)) + Deforestation/2
    Litterfall = (Plants* (55/Plant_Initial))+(Deforestation/2)
    SoilResp = Soils * (55/Soil_Initial)
    Volcanoes = 0.1
    AtmCO2_ = AtmCO2(Atmosphere)
    GlobalTemp_ = GlobalTemp(AtmCO2_)
    WaterTemp_ = WaterTemp(GlobalTemp_)
    Photosynthesis = 110 * CO2Effect(AtmCO2_) * (VegLandArea_percent/100) * TempEffect(GlobalTemp_)
    HCO3_ = HCO3(Kcarb(WaterTemp_), SurfCConc(SurfaceOcean))
    pCO2Oc_ = pCO2Oc(KCO2(WaterTemp_), HCO3_, CO3(HCO3_))
    AtmOcExchange = Kao*(AtmCO2_-pCO2Oc_)
    if x[3] > 0:
        FossilFuelsCombustion_ = FossilFuelsCombustion(t)
    else:
        FossilFuelsCombustion_ = 0
    dAtmosphere_dt = (PlantResp + SoilResp + Volcanoes + FossilFuelsCombustion_
                          - Photosynthesis - AtmOcExchange)

    Sedimentation = DeepOcean * (0.1/DeepOcean_Initial)
    dCarbonateRock_dt = Sedimentation - Volcanoes

    Downwelling = SurfaceOcean*(90.1/SurfaceOcean_Initial)
    Upwelling = DeepOcean * (90/DeepOcean_Initial)
    dDeepOcean_dt= Downwelling - Sedimentation - Upwelling

    dFossilFuelCarbon_dt = - FossilFuelsCombustion_

    dPlants_dt = Photosynthesis - PlantResp - Litterfall

    dSoils_dt = Litterfall - SoilResp

    dSurfaceOcean_dt = Upwelling + AtmOcExchange - Downwelling

    Development = (Deforestation/Plant_Initial * 0.2) * 100
    dVegLandArea_percent_dt = - Development

    derivative = np.array([
        dAtmosphere_dt,
        dCarbonateRock_dt,
        dDeepOcean_dt,
        dFossilFuelCarbon_dt,
        dPlants_dt,
        dSoils_dt,
        dSurfaceOcean_dt,
        dVegLandArea_percent_dt
        ])
    return derivative

# Perform a single step of the simulation using Euler's method
# @njit
def step(x, t, dt):
    return x + derivative(x, t) * dt

# Perform the simulation over a specified time range
# @njit
def run_simulation(x0, t0, tf, dt):
    times = np.arange(t0, tf + dt, dt)
    results = np.zeros((len(times), len(x0)))
    results[0] = x0

    for i in range(1, len(times)):
        results[i] = step(results[i-1], times[i-1], dt)

    return times, results

# Function to plot the results of the simulation, showing the dynamics of the carbon cycle reservoirs and related effects over time
def plot_results(times, results, initial_state, dt, output_base_dir='./data/plots/trajectories'):
    output_dir = get_output_path(output_base_dir, initial_state)
    run_tag = build_run_tag(initial_state)
    years = times[-1] - times[0]
    dt_tag = _fmt_value(dt)
    years_tag = _fmt_value(years)
    filename = f'plot_{run_tag}_years{years_tag}_dt{dt_tag}.pdf'
    output_path = output_dir / filename
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    axs[1, 1].axis('off')
    fig.suptitle(f'Carbon Cycle Simulation | {run_tag}', fontsize=11)

    def norm(series):
        series_min = np.min(series)
        series_max = np.max(series)
        span = series_max - series_min
        if span == 0:
            return np.full_like(series, 0.5, dtype=float)
        return (series - series_min) / span

    def range_label(series):
        return f'[{np.min(series):.3g}, {np.max(series):.3g}]'

    palette = {
        'atm': 'tab:blue',
        'fossil': 'tab:red',
        'deep': 'tab:purple',
        'plants': 'tab:green',
        'soils': 'tab:brown',
        'co2': 'tab:cyan',
        'temp': 'tab:orange',
        'photo': 'tab:pink',
        'rock': 'tab:olive',
        'surface': 'tab:gray',
        'veg': 'tab:green'
    }
    
    # Subplot 1
    line = ax1.plot(times, norm(results[:, 0]), label=f'Atmosphere {range_label(results[:, 0])}', color=palette['atm'], linewidth=2)[0]
    line = ax1.plot(times, norm(results[:, 3]), label=f'Fossil Fuel Carbon {range_label(results[:, 3])}', color=palette['fossil'], linewidth=2)[0]
    line = ax1.plot(times, norm(results[:, 2]), label=f'Deep Ocean {range_label(results[:, 2])}', color=palette['deep'], linewidth=2)[0]
    line = ax1.plot(times, norm(results[:, 4]), label=f'Plants {range_label(results[:, 4])}', color=palette['plants'], linewidth=2)[0]
    line = ax1.plot(times, norm(results[:, 5]), label=f'Soils {range_label(results[:, 5])}', color=palette['soils'], linewidth=2)[0]
    ax1.set_ylabel('Normalized (min→0, max→1)')
    ax1.set_title('Reservoir Dynamics')
    ax1.set_xlabel('Time (years)')
    legend1 = ax1.legend(ncol=1, frameon=False, loc='upper left', fontsize=9)
    for text, handle in zip(legend1.get_texts(), legend1.legend_handles):
        text.set_color(handle.get_color())

    # Subplot 2
    AtmCO2_values = np.array([AtmCO2(Atmosphere) for Atmosphere in results[:, 0]])
    TempEffect_values = np.array([TempEffect(GlobalTemp(AtmCO2)) for AtmCO2 in AtmCO2_values])
    CO2Effect_values = np.array([CO2Effect(AtmCO2) for AtmCO2 in AtmCO2_values])
    Photosynthesis_values = np.array([110 * CO2Effect(AtmCO2) * (VegLandArea_percent/100) * TempEffect(GlobalTemp(AtmCO2)) for AtmCO2, VegLandArea_percent in zip(AtmCO2_values, results[:, 7])])
    GlovalTemp_values = np.array([GlobalTemp(AtmCO2) for AtmCO2 in AtmCO2_values])
    ax2.plot(times, norm(AtmCO2_values), label=f'Atmosphere CO2 {range_label(AtmCO2_values)}', color=palette['co2'], linewidth=2)
    ax2.plot(times, norm(TempEffect_values), label=f'Temperature effect {range_label(TempEffect_values)}', color=palette['temp'], linewidth=2)
    ax2.plot(times, norm(CO2Effect_values), label=f'CO2 effect {range_label(CO2Effect_values)}', color='tab:blue', linewidth=2)
    ax2.plot(times, norm(Photosynthesis_values), label=f'Photosynthesis {range_label(Photosynthesis_values)}', color=palette['photo'], linewidth=2)
    ax2.plot(times, norm(GlovalTemp_values), label=f'Global temperature {range_label(GlovalTemp_values)}', color='tab:orange', linestyle='--', linewidth=2)
    ax2.set_ylabel('Normalized (min→0, max→1)')
    ax2.set_title('Climate and Biosphere Effects')
    ax2.set_xlabel('Time (years)')
    legend2 = ax2.legend(ncol=1, frameon=False, loc='upper left', fontsize=9)
    for text, handle in zip(legend2.get_texts(), legend2.legend_handles):
        text.set_color(handle.get_color())

    # Subplot 3
    FossilFuelsCombustion_values = np.array([FossilFuelsCombustion(t) for t in times])
    ax3.plot(times, norm(results[:, 1]), label=f'Carbonate Rock {range_label(results[:, 1])}', color=palette['rock'], linewidth=2)
    ax3.plot(times, norm(FossilFuelsCombustion_values), label=f'Fossil Fuel Combustion {range_label(FossilFuelsCombustion_values)}', color=palette['fossil'], linewidth=2)
    ax3.plot(times, norm(results[:, 2]), label=f'Deep Ocean {range_label(results[:, 2])}', color=palette['deep'], linewidth=2)
    ax3.plot(times, norm(results[:, 6]), label=f'Surface Ocean {range_label(results[:, 6])}', color=palette['surface'], linewidth=2)
    ax3.plot(times, norm(results[:, 7]), label=f'Veg Land Area % {range_label(results[:, 7])}', color=palette['veg'], linewidth=2)
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Normalized (min→0, max→1)')
    ax3.set_title('Ocean–Geology–Land Coupling')
    legend3 = ax3.legend(ncol=1, frameon=False, loc='upper left', fontsize=9)
    for text, handle in zip(legend3.get_texts(), legend3.legend_handles):
        text.set_color(handle.get_color())

    for ax in (ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f'Saved plot to: {output_path}')

# Function to plot the comparison between the atmospheric CO2 levels from the simulation with historical data
def compare_with_historical_data(times, results, historical_data_path='./data/datasets/carbon_atmosphere.csv', temperature_data_path='./data/datasets/global_temperature.csv'):
    historical_data = np.genfromtxt(historical_data_path, delimiter=',', skip_header=1)
    historical_years = historical_data[:, 0]
    historical_co2 = historical_data[:, 1]
    
    temperature_data = np.genfromtxt(temperature_data_path, delimiter=',', skip_header=1)
    temperature_years = temperature_data[:, 0]
    temperature_values = temperature_data[:, 1]
    
    simulated_atm_co2 = np.array([AtmCO2(Atmosphere) for Atmosphere in results[:, 0]])
    simulated_temp = np.array([GlobalTemp(AtmCO2(Atmosphere)) for Atmosphere in results[:, 0]])
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    # CO2 comparison
    axs[0].plot(times, simulated_atm_co2, label='Simulated', color='tab:blue', linewidth=2)
    axs[0].plot(historical_years, historical_co2, label='Historical', color='tab:red', linewidth=2, linestyle='--')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Atmospheric CO2 (ppm)')
    axs[0].set_title('Atmospheric CO2 Comparison')
    axs[0].legend()
    axs[0].grid(alpha=0.25)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    
    # Temperature comparison
    axs[1].plot(times, simulated_temp, label='Simulated', color='tab:blue', linewidth=2)
    # Only plot temperature data within the simulation time range
    mask = (temperature_years >= times[0]) & (temperature_years <= times[-1])
    axs[1].plot(temperature_years[mask], temperature_values[mask], label='Historical', color='tab:red', linewidth=2, linestyle='--')
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].set_title('Global Temperature Comparison')
    axs[1].legend()
    axs[1].grid(alpha=0.25)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    
    plt.savefig('./data/plots/comparisons/atmospheric_co2_temperature_comparison.pdf', dpi=300)
    plt.show()
    print('Saved comparison plot to: ./data/plots/comparisons/atmospheric_co2_temperature_comparison.pdf')

def step_rk4(x, t, dt): # implementation of runge kutta4
    k1 = derivative(x, t)
    k2 = derivative(x + dt/2 * k1, t + dt/2)
    k3 = derivative(x + dt/2 * k2, t + dt/2)
    k4 = derivative(x + dt * k3, t + dt)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def run_simulation_rk4(x0, t0, tf, dt):
    times = np.arange(t0, tf + dt, dt)
    results = np.zeros((len(times), len(x0)))
    results[0] = x0
    for i in range(1, len(times)):
        results[i] = step_rk4(results[i-1], times[i-1], dt)
    return times, results

def comparison_euler_rg4(): # comparison between the two methods.
    t0 = 1850
    tf = 2015

    plt.figure(figsize=(12, 5))

    for dt in [2.0, 1.0, 0.1]:
        times_euler, results_euler = run_simulation(x0, t0, tf, dt)
        times_rk4,   results_rk4   = run_simulation_rk4(x0, t0, tf, dt)

        co2_euler = np.array([AtmCO2(a) for a in results_euler[:, 0]])
        co2_rk4   = np.array([AtmCO2(a) for a in results_rk4[:, 0]])

        # On ignore les valeurs aberrantes d'Euler pour l'affichage
        co2_euler = np.clip(co2_euler, 0, 600)

        plt.plot(times_euler, co2_euler, label=f'Euler dt={dt}', linewidth=2)
        plt.plot(times_rk4,   co2_rk4,   label=f'RK4 dt={dt}',   linewidth=2, linestyle='--')

    plt.ylim(250, 600)
    plt.xlabel('Année')
    plt.ylabel('CO₂ (ppm)')
    plt.title('Euler vs RK4 regarding the step dt')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('./data/plots/comparisons/euler_rungekutta4comparison.pdf', dpi=300)
    plt.show()

