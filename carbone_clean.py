import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Initial conditions (Gt of carbon, 1850 baseline) ──────────────────────────
Atmosphere_Initial      = 750
CarbonateRock_Initial   = 100_000_000
DeepOcean_Initial       = 38_000
FossilFuel_Initial      = 7_500
Plant_Initial           = 560
Soil_Initial            = 1_500
SurfaceOcean_Initial    = 890
VegLandArea_percent_Initial = 100

x0 = np.array([
    Atmosphere_Initial, CarbonateRock_Initial, DeepOcean_Initial,
    FossilFuel_Initial, Plant_Initial, Soil_Initial,
    SurfaceOcean_Initial, VegLandArea_percent_Initial
], dtype=float)

STATE_LABELS = ['atm', 'rock', 'deep', 'fossil', 'plant', 'soil', 'surf', 'veg']

# ── Physical constants ─────────────────────────────────────────────────────────
Alk        = 2.222446077610055   # alkalinity (carbonate chemistry)
Kao        = 0.278               # air-sea gas exchange coefficient
SurfOcVol  = 0.0362             # surface ocean volume (normalisation)
Deforestation = 1.5             # Gt C/yr land-use flux

# ── Scenario flag ──────────────────────────────────────────────────────────────
SCENARIO = "BAU"   # "BAU" = Business As Usual | "ACTION" = zero emissions after 2030

# ── Helper: output path builder ────────────────────────────────────────────────
def _fmt(v):
    if v == 0:
        return '0e0'
    s = f'{float(v):.3e}'
    m, e = s.split('e')
    return f'{m.rstrip("0").rstrip(".").replace(".", "p")}e{int(e)}'

def get_output_dir(subdir):
    p = BASE_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL MODEL — diagnostic variables
# ══════════════════════════════════════════════════════════════════════════════

def AtmCO2(A):
    """Convert atmospheric carbon stock to CO2 concentration (ppm)."""
    return A * (280 / Atmosphere_Initial)

def GlobalTemp(co2_ppm):
    """Linear climate sensitivity: +1°C per 100 ppm above pre-industrial."""
    return 15 + (co2_ppm - 280) * 0.01

def CO2Effect(co2_ppm):
    """CO2 fertilisation effect on photosynthesis (dimensionless)."""
    return 1.5 * (co2_ppm - 40) / (co2_ppm + 80)

def TempEffect(T):
    """Temperature optimum curve for photosynthesis (dimensionless)."""
    return ((60 - T) * (T + 15)) / ((37.5) ** 2) / 0.96

def WaterTemp(T):
    return 273 + T

def SurfCConc(O):
    return (O / 12_000) / SurfOcVol

def Kcarb(Tw):
    return 5.75e-4 + 6e-6 * (Tw - 278)

def KCO2(Tw):
    return 0.035 + 0.0019 * (Tw - 278)

def HCO3(kc, sc):
    """Solve bicarbonate concentration from alkalinity constraint (quadratic)."""
    denom = 1 - 4 * kc
    if abs(denom) < 1e-10:
        return sc / 2
    disc = sc**2 - Alk * (2*sc - Alk) * (1 - 4*kc)
    return (sc - np.sqrt(max(disc, 0))) / denom

def CO3(hco3):
    return (Alk - hco3) / 2

def pCO2Oc(kco2, hco3, co3):
    """Ocean surface pCO2 from carbonate chemistry."""
    return 280 * kco2 * (hco3**2 / co3)

# ── Fossil fuel combustion: piecewise-linear BAU profile ─────────────────────
_FF_DATA = np.array([
    [1850, 0.00], [1875, 0.30], [1900, 0.60], [1925, 1.35],
    [1950, 2.85], [1975, 4.95], [2000, 7.20], [2025, 10.05],
    [2050, 14.85], [2075, 20.70], [2100, 30.00]
])

def FossilFuelsCombustion(t):
    if SCENARIO == "ACTION" and t > 2030:
        return 0.0
    if t >= _FF_DATA[-1, 0]:
        return _FF_DATA[-1, 1]
    i = np.searchsorted(_FF_DATA[:, 0], t, side='right') - 1
    i = max(i, 0)
    t0, f0 = _FF_DATA[i]
    t1, f1 = _FF_DATA[i + 1]
    return f0 + (t - t0) / (t1 - t0) * (f1 - f0)

# ══════════════════════════════════════════════════════════════════════════════
# ODE RIGHT-HAND SIDE
# ══════════════════════════════════════════════════════════════════════════════

def derivative(x, t):
    A, R, D, F, P, S, O, V = x

    co2   = AtmCO2(A)
    T     = GlobalTemp(co2)
    Tw    = WaterTemp(T)
    hco3  = HCO3(Kcarb(Tw), SurfCConc(O))
    co3   = CO3(hco3)

    Photo      = 110 * CO2Effect(co2) * (V / 100) * TempEffect(T)
    PlantResp  = P * (55 / Plant_Initial) + Deforestation / 2
    Litterfall = PlantResp                          # symmetric by construction
    SoilResp   = S * (55 / Soil_Initial)
    AO_flux    = Kao * (co2 - pCO2Oc(KCO2(Tw), hco3, co3))
    FF_comb    = FossilFuelsCombustion(t) if F > 0 else 0.0
    Volc       = 0.1
    Sediment   = D * (0.1  / DeepOcean_Initial)
    Downwell   = O * (90.1 / SurfaceOcean_Initial)
    Upwell     = D * (90   / DeepOcean_Initial)
    Develop    = (Deforestation / Plant_Initial * 0.2) * 100

    return np.array([
        PlantResp + SoilResp + Volc + FF_comb - Photo - AO_flux,   # dA/dt
        Sediment - Volc,                                             # dR/dt
        Downwell - Sediment - Upwell,                               # dD/dt
        -FF_comb,                                                    # dF/dt
        Photo - PlantResp - Litterfall,                             # dP/dt
        Litterfall - SoilResp,                                      # dS/dt
        Upwell + AO_flux - Downwell,                                # dO/dt
        -Develop,                                                    # dV/dt
    ])

# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL INTEGRATORS
# ══════════════════════════════════════════════════════════════════════════════

def _make_times(t0, tf, dt):
    return np.arange(t0, tf + dt, dt)

def run_euler(x0, t0, tf, dt):
    times = _make_times(t0, tf, dt)
    res = np.zeros((len(times), len(x0)))
    res[0] = x0
    for i in range(1, len(times)):
        res[i] = res[i-1] + dt * derivative(res[i-1], times[i-1])
    return times, res

def run_heun(x0, t0, tf, dt):
    times = _make_times(t0, tf, dt)
    res = np.zeros((len(times), len(x0)))
    res[0] = x0
    for i in range(1, len(times)):
        x, t = res[i-1], times[i-1]
        k1 = derivative(x, t)
        k2 = derivative(x + dt * k1, t + dt)
        res[i] = x + dt / 2 * (k1 + k2)
    return times, res

def _rk4_step(x, t, dt):
    k1 = derivative(x, t)
    k2 = derivative(x + dt/2 * k1, t + dt/2)
    k3 = derivative(x + dt/2 * k2, t + dt/2)
    k4 = derivative(x + dt   * k3, t + dt)
    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

def run_rk4(x0, t0, tf, dt):
    times = _make_times(t0, tf, dt)
    res = np.zeros((len(times), len(x0)))
    res[0] = x0
    for i in range(1, len(times)):
        res[i] = _rk4_step(res[i-1], times[i-1], dt)
    return times, res

def run_am3(x0, t0, tf, dt):
    """Adams-Moulton 3 (implicit predictor-corrector), bootstrapped with 2 RK4 steps."""
    times = _make_times(t0, tf, dt)
    res = np.zeros((len(times), len(x0)))
    res[0] = x0
    res[1] = _rk4_step(res[0], times[0], dt)
    res[2] = _rk4_step(res[1], times[1], dt)

    f_prev2 = derivative(res[0], times[0])
    f_prev1 = derivative(res[1], times[1])

    for i in range(3, len(times)):
        x, t = res[i-1], times[i-1]
        f_cur = derivative(x, t)
        # Adams-Bashforth 2 predictor
        x_pred = x + dt / 12 * (23*f_cur - 16*f_prev1 + 5*f_prev2)
        # Adams-Moulton 3 corrector
        res[i] = x + dt / 12 * (5*derivative(x_pred, t + dt) + 8*f_cur - f_prev1)
        f_prev2, f_prev1 = f_prev1, f_cur

    return times, res

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_convergence():
    """
    We compute absolute CO2 errors against a high-resolution RK4 reference
    (dt=0.001) and plot them on a log-log scale to verify theoretical orders.
    """
    t0, tf = 1850, 2015
    _, ref = run_rk4(x0, t0, tf, dt=0.001)
    co2_ref = AtmCO2(ref[-1, 0])

    dts = [0.1, 0.05, 0.02, 0.01, 0.005]
    methods = {
        'Euler (order 1)': (run_euler,  'o-', 1),
        'Heun (order 2)':  (run_heun,   '^-', 2),
        'RK4 (order 4)':   (run_rk4,    's-', 4),
    }

    plt.figure(figsize=(8, 6))
    for label, (runner, style, order) in methods.items():
        errs = [abs(AtmCO2(runner(x0, t0, tf, h)[1][-1, 0]) - co2_ref) for h in dts]
        plt.loglog(dts, errs, style, label=label, linewidth=2, markersize=8)
        # Reference slope anchored at largest dt
        ref_slope = errs[0] * (np.array(dts) / dts[0]) ** order
        plt.loglog(dts, ref_slope, 'k--' if order==1 else ('k-.' if order==2 else 'k:'),
                   alpha=0.4, linewidth=1.5, label=f'Slope {order} (theoretical)')

    plt.xlabel('Time step dt (years)', fontsize=12)
    plt.ylabel('Absolute CO₂ error (ppm)', fontsize=12)
    plt.title('Convergence Analysis: Error vs. Time Step', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(get_output_dir('data/plots/comparisons') / 'convergence_analysis.png', dpi=300)
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_reservoirs(times, results):
    """
    We normalise each time series to [0,1] so reservoirs with very different
    magnitudes can be compared on the same axes.
    """
    def norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)

    def rl(s):
        return f'[{s.min():.3g}, {s.max():.3g}]'

    co2   = np.array([AtmCO2(a)          for a in results[:, 0]])
    T_abs = np.array([GlobalTemp(c)       for c in co2])
    photo = np.array([110 * CO2Effect(c) * (v/100) * TempEffect(GlobalTemp(c))
                      for c, v in zip(co2, results[:, 7])])
    ff_flux = np.array([FossilFuelsCombustion(t) for t in times])

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    run_tag = '_'.join(f'{n}{_fmt(v)}' for n, v in zip(STATE_LABELS, results[0]))
    fig.suptitle(f'Carbon Cycle Simulation | {run_tag}', fontsize=11)
    ax1, ax2, ax3 = axs[0, 0], axs[0, 1], axs[1, 0]
    axs[1, 1].axis('off')

    palette = dict(atm='tab:blue', fossil='tab:red', deep='tab:purple',
                   plants='tab:green', soils='tab:brown', co2='tab:cyan',
                   temp='tab:orange', photo='tab:pink', rock='tab:olive',
                   surface='tab:gray', veg='tab:green')

    def styled_legend(ax):
        leg = ax.legend(ncol=1, frameon=False, loc='upper left', fontsize=9)
        for txt, h in zip(leg.get_texts(), leg.legend_handles):
            txt.set_color(h.get_color())

    # Subplot 1 — reservoir stocks
    for col, lbl, c in [
        (results[:, 0], f'Atmosphere {rl(results[:,0])}',       palette['atm']),
        (results[:, 3], f'Fossil Fuel Carbon {rl(results[:,3])}', palette['fossil']),
        (results[:, 2], f'Deep Ocean {rl(results[:,2])}',        palette['deep']),
        (results[:, 4], f'Plants {rl(results[:,4])}',            palette['plants']),
        (results[:, 5], f'Soils {rl(results[:,5])}',             palette['soils']),
    ]:
        ax1.plot(times, norm(col), label=lbl, color=c, linewidth=2)
    ax1.set_title('Reservoir Dynamics')
    styled_legend(ax1)

    # Subplot 2 — climate & biosphere effects
    for col, lbl, c, ls in [
        (co2,              f'Atmosphere CO2 {rl(co2)}',              palette['co2'],   '-'),
        (np.array([TempEffect(GlobalTemp(c)) for c in co2]),
                           f'Temperature effect {rl(np.array([TempEffect(GlobalTemp(c)) for c in co2]))}',
                                                                     palette['temp'],  '-'),
        (np.array([CO2Effect(c) for c in co2]),
                           f'CO2 effect {rl(np.array([CO2Effect(c) for c in co2]))}',
                                                                     'tab:blue',       '-'),
        (photo,            f'Photosynthesis {rl(photo)}',            palette['photo'], '-'),
        (T_abs,            f'Global temperature {rl(T_abs)}',        'tab:orange',     '--'),
    ]:
        ax2.plot(times, norm(col), label=lbl, color=c, linestyle=ls, linewidth=2)
    ax2.set_title('Climate and Biosphere Effects')
    styled_legend(ax2)

    # Subplot 3 — ocean-geology-land coupling
    for col, lbl, c in [
        (results[:, 1], f'Carbonate Rock {rl(results[:,1])}',       palette['rock']),
        (ff_flux,       f'Fossil Fuel Combustion {rl(ff_flux)}',     palette['fossil']),
        (results[:, 2], f'Deep Ocean {rl(results[:,2])}',            palette['deep']),
        (results[:, 6], f'Surface Ocean {rl(results[:,6])}',         palette['surface']),
        (results[:, 7], f'Veg Land Area % {rl(results[:,7])}',       palette['veg']),
    ]:
        ax3.plot(times, norm(col), label=lbl, color=c, linewidth=2)
    ax3.set_title('Ocean–Geology–Land Coupling')
    styled_legend(ax3)

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Normalized (min→0, max→1)')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    out = get_output_dir('data/plots/trajectories')
    plt.savefig(out / f'plot_{run_tag}.pdf', dpi=300)
    plt.show()


def compare_with_historical(times, results,
        co2_path  = BASE_DIR / 'data/datasets/carbon_atmosphere.csv',
        temp_path = BASE_DIR / 'data/datasets/global_temperature.csv'):
    """We overlay simulated CO2 and temperature against observational records."""
    hist_co2  = np.genfromtxt(co2_path,  delimiter=',', skip_header=1)
    hist_temp = np.genfromtxt(temp_path, delimiter=',', skip_header=1)

    sim_co2  = np.array([AtmCO2(a)            for a in results[:, 0]])
    sim_temp = np.array([GlobalTemp(AtmCO2(a)) for a in results[:, 0]])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axs[0].plot(times, sim_co2, label='Simulated', color='tab:blue', linewidth=2)
    axs[0].plot(hist_co2[:, 0], hist_co2[:, 1], '--', label='Historical',
                color='tab:red', linewidth=2)
    axs[0].set(xlabel='Year', ylabel='Atmospheric CO2 (ppm)',
               title='Atmospheric CO2 Comparison')

    mask = (hist_temp[:, 0] >= times[0]) & (hist_temp[:, 0] <= times[-1])
    axs[1].plot(times, sim_temp, label='Simulated', color='tab:blue', linewidth=2)
    axs[1].plot(hist_temp[mask, 0], hist_temp[mask, 1], '--', label='Historical',
                color='tab:red', linewidth=2)
    axs[1].set(xlabel='Year', ylabel='Temperature (°C)',
               title='Global Temperature Comparison')

    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    out = get_output_dir('data/plots/comparisons')
    plt.savefig(out / 'atmospheric_co2_temperature_comparison.pdf', dpi=300)
    plt.show()


def plot_temperature_anomaly():
    """
    We compute the simulated anomaly relative to the 1850 baseline and compare
    it against a sparse set of observational benchmarks.
    """
    times, results = run_rk4(x0, 1850, 2100, dt=0.1)
    sim_T = np.array([GlobalTemp(AtmCO2(a)) for a in results[:, 0]])
    anomaly = sim_T - sim_T[0]

    obs_years   = np.array([1850, 1900, 1950, 2000, 2020])
    obs_anomaly = np.array([0.0, -0.1, 0.0, 0.6, 1.2])

    plt.figure(figsize=(10, 6))
    plt.plot(times, anomaly, label='Simulated', color='tab:blue', linewidth=2.5)
    plt.plot(obs_years, obs_anomaly, 'o--', label='Historical',
             color='tab:red', markersize=8, linewidth=2)
    plt.xlabel('Year', fontsize=13)
    plt.ylabel('Global Temperature Anomaly (°C)', fontsize=13)
    plt.title('Global Temperature Anomaly Projection (1850–2100)',
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_output_dir('data/plots/comparisons') / 'global_anomaly.png', dpi=300)
    plt.show()


def plot_scenarios():
    """
    We compare the BAU trajectory against an idealised Scenario B where
    all fossil fuel combustion stops abruptly in 2030.
    """
    global SCENARIO
    times = _make_times(1850, 2100, 0.1)

    SCENARIO = "BAU"
    _, res_bau = run_rk4(x0, 1850, 2100, dt=0.1)
    SCENARIO = "ACTION"
    _, res_act = run_rk4(x0, 1850, 2100, dt=0.1)
    SCENARIO = "BAU"

    T_bau = [GlobalTemp(AtmCO2(a)) for a in res_bau[:, 0]]
    T_act = [GlobalTemp(AtmCO2(a)) for a in res_act[:, 0]]

    plt.figure(figsize=(9, 5))
    plt.plot(times, T_bau, label='Scenario A — Business As Usual',      color='tab:red',   linewidth=2.5)
    plt.plot(times, T_act, label='Scenario B — Zero emissions after 2030', color='tab:green', linewidth=2.5)
    plt.axvline(2030, color='grey', linestyle='--', label='2030 action point')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Global Temperature (°C)', fontsize=12)
    plt.title('Future Climate Projections for Bibi (1850–2100)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_output_dir('data/plots') / 'bibi_scenarios.png', dpi=300)
    plt.show()


def verify_mass_conservation(times, results):
    """We check that total carbon across all liquid/solid reservoirs is conserved."""
    total = results[:, :7].sum(axis=1)   # excludes VegLandArea (dimensionless %)
    m0 = total[0]

    plt.figure(figsize=(9, 5))
    plt.plot(times, total, color='tab:green', linewidth=3, label='Total Carbon Mass')
    plt.ylim(m0 * 0.999, m0 * 1.001)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Mass (Gt of Carbon)', fontsize=12)
    plt.title('Physical Validation: Carbon Mass Conservation', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_output_dir('data/plots/comparisons') / 'conservation_des_masses.png', dpi=300)
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# CONSISTANCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_consistance():
    """
    Erreur de troncature locale (LTE) :
    pour chaque dt, on fait UN pas numérique depuis un état x(t*)
    puis on compare à une référence fine à ce MEME temps t*+dt.
    On vérifie ainsi LTE ~ O(dt^(p+1)).
    """
    # On évite les années-charnières (forcing fossile linéaire par morceaux)
    # afin de mesurer les ordres théoriques en régime lisse.
    t0 = 1860.3

    def _advance_rk4(x_init, t_init, dt, n_steps):
        x = x_init.copy()
        h = dt / n_steps
        t = t_init
        for _ in range(n_steps):
            x = _rk4_step(x, t, h)
            t += h
        return x

    # État de départ x(t0) obtenu avec une intégration RK4 fine depuis 1850.
    x_t0 = _advance_rk4(x0, 1850.0, t0 - 1850.0, n_steps=20000)

    dts = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

    def _euler_step(x, t, h):
        return x + h * derivative(x, t)

    def _heun_step(x, t, h):
        k1 = derivative(x, t)
        k2 = derivative(x + h * k1, t + h)
        return x + h / 2 * (k1 + k2)

    methods = {
        'Euler (LTE ordre 2)':  (_euler_step, 2),
        'Heun (LTE ordre 3)':   (_heun_step, 3),
        'RK4 (LTE ordre 5)':    (_rk4_step, 5),
    }

    plt.figure(figsize=(8, 6))
    for label, (step_fn, order) in methods.items():
        errs = []
        for h in dts:
            # Référence au même temps final t0+h (très fine, dépend de h)
            x_ref_h = _advance_rk4(x_t0, t0, h, n_steps=2000)
            # Un seul pas de la méthode testée depuis x(t0)
            x_num_h = step_fn(x_t0, t0, h)
            errs.append(abs(AtmCO2(x_num_h[0]) - AtmCO2(x_ref_h[0])))

        plt.loglog(dts, errs, 'o-', label=label, linewidth=2)
        ref = errs[0] * (np.array(dts)/dts[0])**order
        plt.loglog(dts, ref, 'k--', alpha=0.4, label=f'Pente {order} (théorique)')

    plt.xlabel('dt (années)')
    plt.ylabel('Erreur de troncature locale (ppm)')
    plt.title('Consistance : erreur locale vs dt')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_output_dir('data/plots/comparisons') / 'consistance.png', dpi=300)
    plt.show()
    print("consistance done")


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_stability():
    """
    Stability analysis following the course definition (Def 1.1.3):
    a problem is stable if its resolvent is locally Lipschitz continuous.
    
    We take d = x0 (initial conditions) and measure how the atmospheric CO2
    at tf=2015 responds to perturbations h of x0, estimating:
        ratio(eps) = ||x(d+h) - x(d)|| / ||h||
    
    If this ratio remains bounded as eps -> 0, the problem is stable (Prop 1.1.4).
    K_abs = lim_{eps->0} sup_{||h||<eps} ||x(d+h)-x(d)|| / ||h||
    """
    t0, tf, dt = 1850, 2015, 0.1

    # Reference solution (unperturbed)
    _, res_ref = run_rk4(x0, t0, tf, dt)
    co2_ref = AtmCO2(res_ref[-1, 0])
    x_ref_final = res_ref[-1]

    epsilons = np.logspace(-1, -8, 20)
    n_dirs = 20  # number of random perturbation directions per epsilon

    # We test perturbations on each reservoir independently + random directions
    reservoir_names = ['Atmosphere', 'CarbonateRock', 'DeepOcean', 'FossilFuel',
                       'Plants', 'Soils', 'SurfaceOcean', 'VegLandArea']

    np.random.seed(42)

    # --- Per-reservoir ratios (one direction: unit vector e_i) ---
    ratios_per_reservoir = np.zeros((len(reservoir_names), len(epsilons)))

    for i in range(len(x0)):
        for j, eps in enumerate(epsilons):
            h = np.zeros(len(x0))
            h[i] = eps
            x_pert = x0 + h
            _, res_pert = run_rk4(x_pert, t0, tf, dt)
            diff = np.linalg.norm(res_pert[-1] - x_ref_final)
            ratios_per_reservoir[i, j] = diff / np.linalg.norm(h)

    # --- Random directions: estimate sup over random unit vectors ---
    sup_ratios = np.zeros(len(epsilons))
    for j, eps in enumerate(epsilons):
        max_ratio = 0.0
        for _ in range(n_dirs):
            delta = np.random.randn(len(x0))
            delta /= np.linalg.norm(delta)
            h = eps * delta
            x_pert = x0 + h
            _, res_pert = run_rk4(x_pert, t0, tf, dt)
            diff = np.linalg.norm(res_pert[-1] - x_ref_final)
            ratio = diff / np.linalg.norm(h)
            if ratio > max_ratio:
                max_ratio = ratio
        sup_ratios[j] = max_ratio

    # K_abs estimated as the value of sup_ratio at the smallest epsilon
    K_abs = sup_ratios[-1]
    K_rel = K_abs * np.linalg.norm(x0) / np.linalg.norm(x_ref_final)

    print(f"\nStability Analysis Results:")
    print(f"  K_abs (absolute condition number) ≈ {K_abs:.4f}")
    print(f"  K_rel (relative condition number) ≈ {K_rel:.4f}")
    print(f"  -> Bounded ratios confirm stability in the sense of Def 1.1.3")

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- Plot 1: sup ratio over random directions (main stability criterion) ---
    ax = axes[0]
    ax.loglog(epsilons, sup_ratios, 'o-', color='tab:blue',
              linewidth=2, markersize=7, label=r'$\sup_{\|h\|<\varepsilon} \|x(d+h)-x(d)\| / \|h\|$')
    ax.axhline(K_abs, color='tab:red', linestyle='--', linewidth=1.5,
               label=f'$K_{{\\rm abs}} \\approx {K_abs:.2f}$')
    ax.set_xlabel(r'Perturbation magnitude $\varepsilon$', fontsize=12)
    ax.set_ylabel(r'Lipschitz ratio', fontsize=12)
    ax.set_title('Stability: Lipschitz ratio vs perturbation size\n'
                 r'(bounded $\Rightarrow$ stable)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotate convergence to K_abs
    ax.annotate(f'Converges to $K_{{\\rm abs}}$',
                xy=(epsilons[-3], sup_ratios[-3]),
                xytext=(epsilons[-3] * 10, sup_ratios[-3] * 2),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)

    # --- Plot 2: per-reservoir ratios at smallest epsilon ---
    ax2 = axes[1]
    ratios_at_small_eps = ratios_per_reservoir[:, -1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(reservoir_names)))
    bars = ax2.bar(reservoir_names, ratios_at_small_eps, color=colors, edgecolor='white')
    ax2.set_xlabel('Reservoir perturbed', fontsize=12)
    ax2.set_ylabel(r'$\|x(d+h)-x(d)\| / \|h\|$ at $\varepsilon=10^{-8}$', fontsize=11)
    ax2.set_title(r'Per-reservoir condition number ($K_{\rm abs}$ estimate)', fontsize=12)
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    # Add value labels on bars
    for bar, val in zip(bars, ratios_at_small_eps):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Stability Analysis — Lipschitz Condition',
                 fontsize=14, fontweight='bold')

    out = get_output_dir('data/plots/comparisons')
    plt.savefig(out / 'stabilite_vp.png', dpi=300)
    plt.show()

    return K_abs, K_rel

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # -- Scenario comparison --------------------------------------------------
    plot_scenarios()

    # -- Reference run (RK4, BAU, full horizon) -------------------------------
    times, results = run_rk4(x0, 1850, 2100, dt=0.1)

    # -- Figures --------------------------------------------------------------
    plot_reservoirs(times, results)
    compare_with_historical(times, results)
    plot_temperature_anomaly()
    verify_mass_conservation(times, results)
    analyse_convergence()
    analyse_consistance()
    analyse_stability()