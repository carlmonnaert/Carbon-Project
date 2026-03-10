import numpy as np
import matplotlib.pyplot as plt
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
Deforestation = 0

# Helper functions
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
    return(SurfCConc-(np.sqrt(SurfCConc**2-Alk*(2*SurfCConc-Alk)*(1-4*Kcarb))))/(1-4*Kcarb)
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

def plot_results(times, results):
    fig = plt.figure(figsize=(21, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # Subplot 1
    ax1.plot(times, results[:, 0]/results[:,0].max(), label='Atmosphere')
    ax1.plot(times, results[:, 3]/results[:,3].max(), label='Fossil Fuel Carbon')
    ax1.plot(times, results[:, 2]/results[:,2].max(), label='Deep Ocean')
    ax1.plot(times, results[:, 4]/results[:,4].max(), label='Plants')
    ax1.plot(times, results[:, 5]/results[:,5].max(), label='Soils')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Carbon (Gt)')
    ax1.set_title('Carbon Cycle Simulation - Part 1')
    ax1.legend()
    ax1.grid()

    # Subplot 2
    AtmCO2_values = np.array([AtmCO2(Atmosphere) for Atmosphere in results[:, 0]])
    TempEffect_values = np.array([TempEffect(GlobalTemp(AtmCO2)) for AtmCO2 in AtmCO2_values])
    CO2Effect_values = np.array([CO2Effect(AtmCO2) for AtmCO2 in AtmCO2_values])
    Photosynthesis_values = np.array([110 * CO2Effect(AtmCO2) * (VegLandArea_percent/100) * TempEffect(GlobalTemp(AtmCO2)) for AtmCO2, VegLandArea_percent in zip(AtmCO2_values, results[:, 7])])
    GlovalTemp_values = np.array([GlobalTemp(AtmCO2) for AtmCO2 in AtmCO2_values])
    ax2.plot(times, AtmCO2_values/AtmCO2_values.max(), label='Atmosphere CO2')
    ax2.plot(times, TempEffect_values/TempEffect_values.max(), label='Temperature effect')
    ax2.plot(times, CO2Effect_values/CO2Effect_values.max(), label='CO2 effect')
    ax2.plot(times, Photosynthesis_values/Photosynthesis_values.max(), label='Photosynthesis')
    ax2.plot(times, GlovalTemp_values/GlovalTemp_values.max(), label='Global Temperature')
    ax2.set_xlabel('Time (years)')
    ax2.set_title('Carbon Cycle Simulation - Part 2')
    ax2.legend()
    ax2.grid()

    # Subplot 3
    FossilFuelsCombustion_values = np.array([FossilFuelsCombustion(t) for t in times])
    ax3.plot(times, results[:, 1]/results[:,1].max(), label='Carbonate Rock')
    ax3.plot(times, FossilFuelsCombustion_values/FossilFuelsCombustion_values.max(), label='Fossil Fuel Combustion')
    ax3.plot(times, results[:, 2]/results[:,2].max(), label='Deep Ocean')
    ax3.plot(times, results[:, 6]/results[:,6].max(), label='Surface Ocean')
    ax3.plot(times, results[:, 7]/results[:,7].max(), label='Veg Land Area %')
    ax3.set_xlabel('Time (years)')
    ax3.set_ylabel('Carbon (Gt) / Percentage (%)')
    ax3.set_title('Carbon Cycle Simulation - Part 3')
    ax3.legend()
    ax3.grid()

    plt.savefig('./data/carbon_simulation_last_plot.pdf')
    plt.tight_layout()
    plt.show()

def main():
    t0 = 1850
    tf = 2100
    dt = 0.1
    times, results = run_simulation(x0, t0, tf, dt)
    plot_results(times, results)
main()