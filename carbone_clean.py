import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

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
def _times_grid(t0, tf, dt):
    n_steps = int(round((tf - t0) / dt))
    return np.linspace(t0, tf, n_steps + 1)


def _make_times(t0, tf, dt):
    n_steps = int(round((tf - t0) / dt))
    return np.linspace(t0, tf, n_steps + 1)

def analyse_convergence():
    t0, tf = 1850, 2015 
    dt_ref = 0.0005
    
    # z(t) : Trajectoire de référence complète
    _, ref = run_rk4(x0, t0, tf, dt=dt_ref)

    dts = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01] 
    
    methods = {
        'Euler (order 1)': (run_euler,  'o-', 1),
        'Heun (order 2)':  (run_heun,   '^-', 2),
        'RK4 (order 4)':   (run_rk4,    's-', 4),
    }

    plt.figure(figsize=(8, 6))
    for label, (runner, style, order) in methods.items():
        errs = []
        for h in dts:
            # y_n : Trajectoire grossière calculée avec le pas h
            _, res = runner(x0, t0, tf, h)
            
            # Pour comparer y_n et z(t_n) au même instant, on extrait les points
            # de la référence qui correspondent exactement aux instants de la méthode grossière.
            step_ratio = int(round(h / dt_ref))
            ref_alignee = ref[::step_ratio] 
            
            # Application stricte de la définition 4.3.5 : 
            # 1. On calcule la différence pour chaque 'n' (en norme relative pour la lisibilité)
            diff = (res - ref_alignee) / x0
            
            # 2. On calcule la norme Euclidienne ||y_n - z(t_n)|| pour chaque instant n
            normes_par_instant = np.linalg.norm(diff, axis=1)
            
            # 3. On prend le MAX sur tout l'intervalle 0 <= n <= N
            erreur_globale_max = np.max(normes_par_instant)
            
            errs.append(erreur_globale_max)
            
        plt.loglog(dts, errs, style, label=label, linewidth=2, markersize=8)
        
        # Pente de référence
        ref_slope = errs[0] * (np.array(dts) / dts[0]) ** order
        plt.loglog(dts, ref_slope, 'k--' if order==1 else ('k-.' if order==2 else 'k:'),
                   alpha=0.4, linewidth=1.5, label=f'Slope {order} (theoretical)')

    plt.xlabel('Time step dt (years)', fontsize=12)
    plt.ylabel(r'Global error $\max_{0 \leq n \leq N} \left\| (y_n - z(t_n)) / z(t_0) \right\|_2$', fontsize=12)
    
    
    plt.title('Convergence Analysis: Error vs. Time Step (1850-2015)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right') 
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(get_output_dir('data/plots/comparisons') / 'convergence_analysis_strict.png', dpi=300)
    plt.show()




def _solve_ivp_reference(t0, tf, t_eval, *, method="DOP853", rtol=1e-13, atol=1e-13):
    def f(t, y):
        return derivative(y, t)
    sol = solve_ivp(
        f, (t0, tf), x0,
        method=method,
        t_eval=t_eval,
        rtol=rtol, atol=atol,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T  # (N, dim)

def analyse_convergence_scipy():
    t0, tf = 1850.0, 2015.0
    dts = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    scales = np.abs(x0).copy()
    scales[scales == 0] = 1.0

    methods = {
        "Euler (order 1)": (run_euler, "o-", 1),
        "Heun (order 2)":  (run_heun,  "^-", 2),
        "RK4 (order 4)":   (run_rk4,   "s-", 4),
    }

    plt.figure(figsize=(8, 6))

    for label, (runner, style, order) in methods.items():
        errs = []
        for h in dts:
            t_coarse = _times_grid(t0, tf, h)

            # Référence "intégration SciPy" évaluée exactement aux mêmes instants
            _, ref = _solve_ivp_reference(
                t0, tf, t_eval=t_coarse,
                method="DOP853",   # bon pour référence non-raide
                rtol=1e-11, atol=1e-13
            )

            # Solution numérique avec pas h
            _, res = runner(x0, t0, tf, h)

            n = min(len(res), len(ref))
            diff = (res[:n] - ref[:n]) / scales
            normes = np.linalg.norm(diff, axis=1)
            errs.append(np.max(normes))

        plt.loglog(dts, errs, style, label=label, linewidth=2, markersize=8)

        # pente théorique
        ref_slope = errs[0] * (np.array(dts) / dts[0]) ** order
        plt.loglog(
            dts, ref_slope,
            "k--" if order == 1 else ("k-." if order == 2 else "k:"),
            alpha=0.4, linewidth=1.5,
            label=f"Slope {order} (theoretical)"
        )

    plt.xlabel("Time step dt (years)")
    plt.ylabel(r"Global error $\max_n \| (y_n - y_{ref}(t_n))/x_0 \|_2$")
    plt.title("Convergence vs dt (reference = solve_ivp DOP853)")
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
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
    Vérifie la VRAIE consistance (Définition 4) : 
    
    Une méthode φ est consistante si pour toute solution exacte z,
    la somme des erreurs de consistance tend vers 0 quand h_max → 0:
    
        lim_{h→0} sum_{n=0}^{N-1} ||e_n|| = 0
    
    où e_n = z(t_{n+1}) - z(t_n) - h·φ(t_n, z(t_n), h)

    - On intègre une solution de référence ultra-fine avec RK4
    - Pour chaque pas h testé (Euler, Heun, RK4):
      * On évalue chaque étape ISOLÉE sur la grille [t_n, t_{n+1}]
      * On compare z(t_{n+1}) (exact) vs z(t_n) + h·φ(t_n, z(t_n), h) (schéma)
      * On calcule e_n à partir de la VRAIE solution (référence RK4 ultra-fine)
    - On somme les normes ||e_n|| : Σ||e_n||
    - Si la méthode est d'ordre p, alors Σ||e_n|| = O(h^p)
    """
    t0, tf = 1850.0, 2015.0
    
    # Pas de discrétisation pour chaque niveau
    dt_ultra_fine = 0.0001  # Solution de référence (très fine)
    dts_coarse = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]  # Pas grossiers à tester

    reservoir_idx = np.arange(7)
    reservoir_names = ['atm', 'rock', 'deep', 'fossil', 'plant', 'soil', 'surf']
    scales = np.abs(x0[reservoir_idx])
    scales[scales == 0] = 1.0  # Éviter division par zéro

    # ÉTAPE 1 : Intégrer une solution de référence ultra-fine (considérée comme "exacte")
    print(f"  Calcul solution de référence (dt={dt_ultra_fine})...")
    _, z_ultra_fine = run_rk4(x0, t0, tf, dt=dt_ultra_fine)
    times_ultra_fine = np.arange(t0, tf + dt_ultra_fine, dt_ultra_fine)

    # Définir les méthodes d'un pas
    def _euler_step(x, t, h):
        """Une étape d'Euler : x_{n+1} = x_n + h·f(t_n, x_n)"""
        return x + h * derivative(x, t)

    def _heun_step(x, t, h):
        """Une étape de Heun (RK2) : utilise k1 et k2"""
        k1 = derivative(x, t)
        k2 = derivative(x + h * k1, t + h)
        return x + h / 2 * (k1 + k2)

    methods = {
        'Euler': (_euler_step, 1),
        'Heun': (_heun_step, 2),
        'RK4': (_rk4_step, 4),
    }

    method_sum_by_h = {}
    rk4_component_sums = []

    # ÉTAPE 2 : Pour chaque pas h, mesurer la somme des défauts locaux
    for label, (step_fn, order) in methods.items():
        print(f"  Analyse {label}...")
        sums_for_h = []
        component_sums_for_h = []

        for h in dts_coarse:
            n_steps = int(round((tf - t0) / h))
            times_h = np.linspace(t0, tf, n_steps + 1)
            
            sum_norm_e = 0.0
            component_sum_abs = np.zeros(len(reservoir_idx))

            for n in range(len(times_h) - 1):
                tn = times_h[n]
                tnp1 = times_h[n + 1]
                h_actual = tnp1 - tn

                idx_n = np.argmin(np.abs(times_ultra_fine - tn))
                idx_np1 = np.argmin(np.abs(times_ultra_fine - tnp1))

                z_n_exact = z_ultra_fine[idx_n]
                z_np1_exact = z_ultra_fine[idx_np1]
                
                # Applique le schéma UNE FOIS à partir du point exact
                # z_scheme(t_{n+1}) = z_n_exact + h·φ(t_n, z_n_exact, h)
                z_np1_scheme = step_fn(z_n_exact, tn, h_actual)
                
                # Défaut local : e_n = z(t_{n+1}) - z(t_n) - h·φ(t_n, z(t_n), h)
                #            = z_exact(t_{n+1}) - z_scheme(t_{n+1})
                e_n = z_np1_exact - z_np1_scheme
                
                # Norme euclidienne relative (normalisée par les valeurs initiales)
                e_rel = e_n[reservoir_idx] / scales
                norm_e = np.linalg.norm(e_rel)
                sum_norm_e += norm_e
                
                # Accumulation par réservoir pour RK4
                component_sum_abs += np.abs(e_rel)

            sums_for_h.append(sum_norm_e)
            if label == 'RK4':
                component_sums_for_h.append(component_sum_abs)

        method_sum_by_h[label] = np.array(sums_for_h)
        if label == 'RK4':
            rk4_component_sums = component_sums_for_h

    plt.figure(figsize=(11, 7))
    styles = {'Euler': 'o-', 'Heun': 's-', 'RK4': '^-'}
    colors = {'Euler': 'C0', 'Heun': 'C1', 'RK4': 'C2'}
    slope_styles = {1: '--', 2: '-.', 4: ':'}
    
    for label, (_, order) in methods.items():
        vals = method_sum_by_h[label]
        plt.loglog(dts_coarse, vals, styles[label], color=colors[label], 
                   linewidth=2.5, markersize=9,
                   label=f'{label} (order {order}): $\\sum_n \\|e_n\\|$')
        
        # Référence : pente théorique O(h^order)
        if len(vals) > 1:
            # Prendre le dernier point comme référence
            h_ref, val_ref = dts_coarse[-1], vals[-1]
            ref_slope = val_ref * (np.array(dts_coarse) / h_ref) ** order
            plt.loglog(dts_coarse, ref_slope, slope_styles[order], 
                      color=colors[label], alpha=0.4, linewidth=1.5)

    plt.xlabel('lenght of h (years)', fontsize=12, fontweight='bold')
    plt.ylabel(r'$\sum_{n=0}^{N-1}\|e_n\|$', fontsize=12, fontweight='bold')
    plt.title('Consistency checking\n', 
              fontsize=13, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.legend(fontsize=10, loc='upper left', framealpha=0.95)
    plt.tight_layout()
    
    out = get_output_dir('data/plots/comparisons')
    plt.savefig(out / 'consistance_definition4.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ÉTAPE 4 : Graphique 2 - Contribution par réservoir (RK4 uniquement)
    if rk4_component_sums:
        rk4_component_sums = np.array(rk4_component_sums)
        plt.figure(figsize=(11, 7))
        
        for j, name in enumerate(reservoir_names):
            plt.loglog(dts_coarse, rk4_component_sums[:, j], 'o-', 
                      linewidth=2.2, markersize=7, label=f'{name}')
        
        plt.xlabel('Pas de temps h (années)', fontsize=12, fontweight='bold')
        plt.ylabel(r'$\sum_{n=0}^{N-1}|e_n^{(i)}| / x_0^{(i)}$', fontsize=12, fontweight='bold')
        plt.title('Consistency checking for all reservoirs\n',
                 fontsize=13, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
        plt.legend(fontsize=9, ncol=2, loc='upper left', framealpha=0.95)
        plt.tight_layout()
        plt.savefig(out / 'consistance_by_reservoir.png', dpi=300, bbox_inches='tight')
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_stability():
    """
    Vérifie numériquement la définition 5 de stabilité.

    Pour chaque horizon N, on calcule

        R_N = max_{0<=j<=N} ||y~_j-y_j||
              -----------------------------------------------
              ||y~_0-y_0|| + sum_{k=0}^{N-1} ||eps_k||

    puis on estime une constante S indépendante de N par

        S_est = max_N R_N

    sur un ensemble de trajectoires perturbées.
    """
    t0, tf, dt = 1850, 2015, 0.1
    times = _make_times(t0, tf, dt)
    n_steps = len(times) - 1
    n_trials = 80
    method_name = 'RK4'
    state_dim = len(x0)

    delta0_size = 1e-6
    target_eps_norm = 1e-8
    eps_std = target_eps_norm / np.sqrt(state_dim)
    eps_variance = eps_std ** 2

    rng = np.random.default_rng(42)

    def step_method(y, t, h):
        return _rk4_step(y, t, h)

    def run_pair(delta0, epsilons):
        y = np.zeros((n_steps + 1, len(x0)))
        y_pert = np.zeros((n_steps + 1, len(x0)))
        y[0] = x0
        y_pert[0] = x0 + delta0

        for n in range(n_steps):
            tn = times[n]
            y[n + 1] = step_method(y[n], tn, dt)
            y_pert[n + 1] = step_method(y_pert[n], tn, dt) + epsilons[n]

        return y, y_pert

    ratio_by_N = np.zeros(n_steps + 1)
    representative_lhs = None
    representative_rhs_base = None
    representative_ratio = -np.inf
    representative_delta0_norm = None

    for _ in range(n_trials):
        delta0 = rng.normal(size=state_dim)
        delta0 /= np.linalg.norm(delta0)
        delta0 *= delta0_size

        # i.i.d. Gaussian perturbations: epsilon_n ~ N(0, sigma^2 I)
        epsilons = rng.normal(loc=0.0, scale=eps_std, size=(n_steps, state_dim))

        y, y_pert = run_pair(delta0, epsilons)

        diff_norms = np.linalg.norm(y_pert - y, axis=1)
        lhs_N = np.maximum.accumulate(diff_norms)
        epsilon_norms = np.linalg.norm(epsilons, axis=1)
        delta0_norm = np.linalg.norm(delta0)
        rhs_base_N = delta0_norm + np.concatenate(([0.0], np.cumsum(epsilon_norms)))
        ratios = np.divide(lhs_N, rhs_base_N,
                           out=np.zeros_like(lhs_N),
                           where=rhs_base_N > 0)

        ratio_by_N = np.maximum(ratio_by_N, ratios)

        trial_peak_ratio = np.max(ratios)
        if trial_peak_ratio > representative_ratio:
            representative_ratio = trial_peak_ratio
            representative_lhs = lhs_N.copy()
            representative_rhs_base = rhs_base_N.copy()
            representative_delta0_norm = delta0_norm

    stability_constant = np.max(ratio_by_N)
    representative_rhs = stability_constant * representative_rhs_base

    print(f"\nStability Analysis Results ({method_name}, definition 5):")
    print(f"  dt = {dt}")
    print(f"  N_max = {n_steps}")
    print(f"  epsilon_n ~ N(0, sigma^2 I), sigma = {eps_std:.3e}")
    print(f"  epsilon variance per component sigma^2 = {eps_variance:.3e}")
    print(f"  R_Nmax ≈ {ratio_by_N[-1]:.4f}")
    print(f"  ||y~_0 - y_0|| (representative) ≈ {representative_delta0_norm:.3e}")
    print(f"  Estimated stability constant S ≈ {stability_constant:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    ax1 = axes[0]
    N_values = np.arange(n_steps + 1)
    ax1.plot(N_values, ratio_by_N, color='tab:blue', linewidth=2.5)
    ax1.axhline(stability_constant, color='tab:red', linestyle='--', linewidth=1.5,
                label=rf'$S \approx {stability_constant:.2f}$')
    ax1.set_xlabel('Number of steps $N$', fontsize=12)
    ax1.set_ylabel(r'Worst-case ratio $R_N$', fontsize=12)
    ax1.set_title(f'{method_name}: verification of definition 5', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2 = axes[1]
    ax2.plot(N_values, representative_lhs, color='tab:orange', linewidth=2.2,
             label=r'$\max_{0\leq j\leq N}\|\tilde y_j-y_j\|$')
    ax2.plot(N_values, representative_rhs, color='tab:green', linestyle='--', linewidth=1.8,
             label=r'$S(\|\tilde y_0-y_0\|+\sum_{k=0}^{N-1}\|\varepsilon_k\|)$')
    ax2.set_xlabel('Number of steps $N$', fontsize=12)
    ax2.set_ylabel('Inequality terms', fontsize=12)
    ax2.set_title(f'{method_name}: left-hand side vs stability bound', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Stability Analysis — Definition 5', fontsize=14, fontweight='bold')

    out = get_output_dir('data/plots/comparisons')
    plt.savefig(out / 'stabilite_vp.png', dpi=300)
    plt.show()

    return ratio_by_N[-1], stability_constant

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    SCENARIO = "BAU"   # reset to default for reference run
    # -- Reference run (RK4, BAU, full horizon) -------------------------------
    times, results = run_rk4(x0, 1850, 2600, dt=0.1)

    # -- Figures --------------------------------------------------------------
    # plot_scenarios()
    #plot_reservoirs(times, results)
    # compare_with_historical(times, results)
    # plot_temperature_anomaly()
    # verify_mass_conservation(times, results)
    #analyse_convergence()
    analyse_consistance()
    #analyse_stability()
    #analyse_convergence_scipy()