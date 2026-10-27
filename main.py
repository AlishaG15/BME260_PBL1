"""TPT -> calcitriol -> calcium model.      Run with:  python main.py
(keep main.py, tpt.py, calcitriol.py and calcium.py in the same folder)

WHAT IS MODELLED  (three linked compartment models, solved one after another)
  1. TPT (teriparatide) pharmacokinetics   injection site, blood, kidney, liver, bone
  2. Calcitriol                            kidney, blood, liver, gut        <- driven by TPT in kidney
  3. Calcium                               intake, gut, blood, kidney, bone <- driven by TPT (kidney, bone) and calcitriol (blood)

UNITS   TPT: mmol | calcitriol and calcium: mg | time: min | volume: L   (the figures show concentrations: ng/mL, pg/mL, mg/dL)

HOW THE RESULTS ARE READ
  Every model is solved twice: TREATED (TPT injection) and CONTROL (no injection). Every compartment starts at
  its no-drug steady state, so the control run stays flat. The TPT effect is always TREATED minus CONTROL.

RUN ORDER
  For each case state (healthy, osteoporosis, renal failure, grey market): apply_conditions -> run_simulation ->
  compute_outputs -> print_summary -> figures. Then the Monte Carlo (run_monte_carlo), if switched on.

Search for "ASSUMPTION" and "SWITCH" to find modelling assumptions and options.
"""

import importlib
import os
import math
from types import SimpleNamespace

import numpy as np

# must come before the parameter imports so the sampled draws repeat

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import tpt
import calcitriol
import calcium

# GREY MARKET is now a patient/case state ("grey_market") alongside healthy, osteoporosis and renal_failure.
# ASSUMPTION: the grey-market dose is 7.7-14% (potency) of a dose that is itself 28-39% above the 20 ug reference
# (users take 1.28-1.39x the label dose), so the effective dose = reference x U(1.28, 1.39) x U(0.077, 0.14).
GREY_POTENCY_RANGE = (0.077, 0.14)      # fraction of the dose that is actually active drug
GREY_OVERDOSE_RANGE = (1.28, 1.39)      # +28% to +39% increase in the amount taken


def apply_conditions(condition="healthy", seed = 0):
    """Apply the disease conditions, then recompute everything the parameter modules derived from the changed values.
    Call once after importing, and again to switch states.
    Returns this draw's DOSE SCALE (dose relative to the 20 ug reference): 1.0 for every state except grey_market
    (about 0.10-0.19) and control (0.0)."""
    np.random.seed(seed)   # same random draws for every condition, so differences come from the condition only
    dose_scale = 1.0
    # Reset modules to their base state for clean runs
    for module in (tpt, calcitriol, calcium):
        importlib.reload(module)

    if condition == "osteoporosis":
        calcium.calcitriol_sensitivity = np.random.uniform(0.2, 0.6)
        calcium.intial_bone_Ca = (1.0e6) * (0.844)    # mg; 1 kg = 1,000,000 mg; bone mass reduced by 15.6% in osteoporosis
        tpt.k_elimination_TPT = tpt.k_elimination_TPT / 2

    if condition == "renal_failure":
        if calcitriol.basal_calcitriol_produce_kidney is None or calcitriol.calcitriol_half_life is None:
            raise ValueError("Fill in healthy calcitriol production and half-life first.")
        kidney_flow_scale = np.random.uniform(0.10, 0.15)   # reduced kidney blood flow in renal failure (applied below)
        tpt.k_diffuse_TPT = tpt.k_diffuse_TPT / 1.73
        tpt.k_elimination_TPT = tpt.k_elimination_TPT / 1.77
        calcitriol.basal_calcitriol_produce_kidney *= 14.2 / 31.8
        calcitriol.calcitriol_half_life /= 0.45 / 0.58

    if condition == "control":
        dose_scale = 0.0   # healthy physiology, no TPT injected: treated = control, so every TPT effect should be 0

    if condition == "grey_market":
        # No calcitriol or calcium parameter changes are assumed; only the delivered dose differs.
        # The dose is passed to run_simulation(dose_scale=...), which is what the solver uses (not tpt.dose).
        dose_scale = np.random.uniform(*GREY_POTENCY_RANGE) * np.random.uniform(*GREY_OVERDOSE_RANGE)

    # --- recompute derived TPT rates (same formulas as tpt.py) ---
    tpt.k_blood_to_kidney_TPT = tpt.frac_elim_kidney_TPT * tpt.k_elimination_TPT
    if condition == "renal_failure":
        tpt.k_blood_to_kidney_TPT *= kidney_flow_scale   # reduced kidney blood flow, applied after the recompute so it isn't overwritten
    tpt.k_kidney_degrade_TPT = tpt.k_blood_to_kidney_TPT
    tpt.k_bloodtoliver_TPT = tpt.frac_elim_toliver_TPT * tpt.k_elimination_TPT
    tpt.k_blood_to_other_tissue = (
        (1 - tpt.frac_elim_kidney_TPT - tpt.frac_elim_toliver_TPT) * tpt.k_elimination_TPT - tpt.TPT_to_bone_receptor)

    if tpt.k_blood_to_other_tissue < 0:
        # Bone uptake alone exceeds total clearance, so "other tissue" would act as a SOURCE of TPT (mass created).
        raise ValueError(f"tpt.k_blood_to_other_tissue = {tpt.k_blood_to_other_tissue:.3g} min^-1 is negative: bone uptake "
                         "exceeds total TPT clearance for this draw/condition. Resample or lower bone uptake.")

    # --- recompute calcitriol degradation and its steady-state starting amounts (same formulas as calcitriol.py) ---
    c = calcitriol
    c.k_degrade_calcitriol = math.log(2) / c.calcitriol_half_life * c.total_over_kidney
    c.Initial_calcitriol_blood = (c.basal_calcitriol_produce_kidney * c.k_kidney_to_blood_calcitriol
                                  / (c.k_degrade_calcitriol * c.k_blood_to_kidney_calcitriol))
    c.Initial_calcitriol_kidney = (c.k_blood_to_kidney_calcitriol / c.k_kidney_to_blood_calcitriol) * c.Initial_calcitriol_blood
    c.initial_calcitriol_liver = (c.k_blood_to_liver_calcitriol / c.K_liver_to_blood_calcitriol) * c.Initial_calcitriol_blood
    c.initial_calcitriol = [c.Initial_calcitriol_kidney, c.Initial_calcitriol_blood,
                            c.initial_calcitriol_liver, c.Initial_calcitriol_gut]

    # --- recompute calcium quantities that depend on calcitriol baseline and bone mass ---
    calcium.calcitriol_basal_level = c.Initial_calcitriol_blood / tpt.Vol_blood_in_bone   # effect = 1 at baseline
    calcium.k_bone_to_blood_Ca = calcium.k_blood_to_bone_Ca * calcium.initial_blood_Ca / calcium.intial_bone_Ca
    calcium.initial_calcium = [calcium.initial_intake_Ca, calcium.initial_gut, calcium.initial_blood_Ca,
                               calcium.initial_kidney_Ca, calcium.intial_bone_Ca]
    return dose_scale


apply_conditions("healthy")

# =====================================================================
# 0. SETTINGS  (model options)
# =====================================================================
SIM_DAYS = 7                  # simulated time: one injection at t = 0 of each day
SIM_MINUTES = SIM_DAYS * 24 * 60
# DOSING: one injection every DOSE_INTERVAL_MIN, starting at t = 0. Each case state has its own dose scale (see apply_conditions).
# The calibrations (Hill constants, calcitriol coefficient, Ca-effect scale) still use ONE reference dose, so
# they describe a single injection and are not inflated by accumulation over several days.
DOSE_INTERVAL_MIN = 24 * 60        # daily
N_DOSES = SIM_DAYS                 # one per day for the whole run; set 1 for a single injection
PLOT_DETAIL_DAYS = 1               # TPT and calcitriol figures zoom to this many days (None = whole run)
SINGLE_DOSE_DETAIL_HOURS = 24      # calcium figure: fast panels (amounts, blood Ca) zoom to this window; bone panels show all days
BONE_PLOT_HOURS = 24               # bone-repair figure x-axis window in hours (None = whole run)
# DOSE SCALE is relative to the standard 20 ug injection (1.0 = reference). It is set per case state by apply_conditions():
# 1.0 for healthy / osteoporosis / renal_failure, potency x overdose (about 0.10-0.19) for grey_market, and 0.0 for
# control (healthy, no TPT added).

# MONTE CARLO: re-draw every uncertain parameter N_RUNS times and run each condition with the same draws.
RUN_MONTE_CARLO = True             # set False to skip
MC_CONDITIONS = ["healthy", "osteoporosis", "renal_failure", "grey_market", "control"]   # case states run (healthy first: it sets the calibration)
N_RUNS = 100
STEP_MIN = 2                       # output time step (min)

# SWITCH: how the Hill half-saturation constant is read.
#   "fraction_of_peak": H = 0.316 x the peak TPT amount in that organ after the injection (ASSUMPTION, see below)
#   "table":            H = calcium.half_saturation_concen = 0.316 pmol (value from the parameter table)
# With the table value the Hill terms are saturated at all times (baseline kidney TPT is ~30 pmol), so the
# injection changes nothing, which is why fraction_of_peak is used.
HALF_SATURATION_MODE = "fraction_of_peak"
HALF_SATURATION_FRACTION = 0.316   # H = 0.316 x peak organ TPT

# SWITCH: half-saturation constant for the TPT -> kidney Ca REABSORPTION Hill term only (calcitriol production
# and bone release keep the constants above). Set ONE of these, or leave both None to reuse the kidney value above.
#   KIDNEY_REABS_H_MMOL:     absolute value in mmol of TPT in kidney (e.g. 5e-9 = 5 pmol)
#   KIDNEY_REABS_H_FRACTION: fraction of the peak kidney TPT amount after the injection (e.g. 0.5)
KIDNEY_REABS_H_MMOL = None
KIDNEY_REABS_H_FRACTION = None

# SWITCH: calibrate the TPT -> calcitriol production coefficient so one dose raises blood calcitriol ~80%.
CALIBRATE_CALCITRIOL_TPT = True
CALCITRIOL_RISE_TARGET = 1.8       # peak (treated / control) blood calcitriol within the run

# INDIRECT RESPONSE (enzyme-mediated) calcitriol production. PTH/TPT does not make calcitriol; it induces the
# kidney enzyme 1-alpha-hydroxylase that makes it. A(t) = relative enzyme activity above baseline (0 = baseline,
# 1 = full Hill saturation):   dA/dt = k_hyd * (hill(TPT_kidney) - A),   production = P0 + M * A,   A(0) = 0.
# k_hyd = ln2 / half-life of the enzyme. Slow onset and slow return to baseline, so production outlives the drug
# (Dayneka, Garg & Jusko 1993, indirect response model III).
# ASSUMPTION: half-life 15 h = 900 min, inside the 8-24 h window consistent with the ~80% calcitriol rise at 24 h
# reported by Shiraki et al. (the enzyme half-life itself was not reported, so it is swept 8-24 h as a sensitivity band).
HYDROXYLASE_HALF_LIFE_MIN = 15 * 60
RUN_HYDROXYLASE_SWEEP = True                 # SWITCH: sensitivity sweep of the enzyme half-life (healthy state)
HYDROXYLASE_SWEEP_HOURS = [8, 12, 15, 18, 24]

# SWITCH: the model equations have no blood -> gut calcitriol term, so it is left out by default.
INCLUDE_BLOOD_TO_GUT_CALCITRIOL = False

# SWITCH (mass balance): the homeostatic Ca loss k_Ca_feedback*(blood - baseline) turns NEGATIVE when blood Ca is
# below baseline, i.e. it would create calcium from nothing. True = only remove EXCESS calcium (never add any).
FEEDBACK_ONLY_ABOVE_BASELINE = True

# SWITCH (bone repair): alkaline phosphatase (ALP) is a marker of bone formation. ASSUMPTION: ALP rises by up to
# ALP_MAX_INCREASE (78%) when blood Ca is elevated, linearly with the Ca rise, reaching the full 78% at a rise of
# ALP_FULL_EFFECT_RISE_MG_PER_DL (default = the +0.4 mg/dL label peak) and staying there above it. The ALP multiplier
# scales calcium FORMATION into bone (blood -> bone). It responds instantly (no ALP lag), because no ALP half-life
# was given. True = this feeds back into the calcium model (the +0.4 mg/dL calibration is redone with it on);
# False = formation stays constant, so the ALP effect is not applied at all.
ALP_FEEDS_BONE_FORMATION = True
ALP_MAX_INCREASE = 0.78
ALP_FULL_EFFECT_RISE_MG_PER_DL = 0.4   # mg/dL; same as LABEL_CALCIUM_PEAK_RISE_MG_PER_DL below

# PLOT VOLUMES: figures show concentration = amount / volume. Only the blood volume (tpt.Vol_blood_in_bone) is defined in
# the parameter files, so ASSUMPTION: every compartment is divided by it unless you give its own volume (L) here, e.g.
# VOLUME_L = {"kidney": 0.3, "liver": 1.5, "bone": 4.0}. Keys: injection, blood, kidney, liver, gut, intake, bone.
VOLUME_L = {}

SAVE_FIGURES = True
FIGURE_FORMAT = "png"             # "png": each figure as its own PNG; "pdf": one PDF per condition. Both save into the figures folder
FIGURE_FOLDER = "figures"
LABEL_CALCIUM_PEAK_RISE_MG_PER_DL = 0.4   # FORTEO label: ~+0.4 mg/dL serum Ca, shown on the plot for reference

# SWITCH: scale the three TPT -> calcium effects (bone release, kidney reabsorption, calcitriol -> gut)
# by ONE factor so the REFERENCE dose raises serum Ca by +0.4 mg/dL (Satterwhite 2010: median +0.4 mg/dL at
# ~4.25 h, back to baseline by 16-24 h). Found once at the reference dose and held fixed for every case state.
CALIBRATE_CA_EFFECT = True
CA_EFFECT_SCALE = 1.0   # 1.0 = sampled values unchanged; replaced by the calibrated value

TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x renamed trapz


# =====================================================================
# 1. SMALL HELPERS
# =====================================================================
def hill(amount, half_saturation, n):
    """Saturating 0-1 response: amount^n / (amount^n + H^n)."""
    amount = np.maximum(amount, 0.0)
    return amount**n / (amount**n + half_saturation**n)


def vol(compartment):
    """Volume (L) used to turn a compartment amount into a concentration for the figures."""
    return VOLUME_L.get(compartment, tpt.Vol_blood_in_bone)


def sci_axes(fig):
    """Scientific notation on every axis of a figure (including twin axes), so tick labels need fewer zeros."""
    for a in fig.axes:
        for which in ("x", "y"):
            try:
                a.ticklabel_format(axis=which, style="sci", scilimits=(-2, 4), useOffset=False, useMathText=True)
            except AttributeError:      # categorical axis (text tick labels): nothing to convert
                pass


def at_time(solution, row):
    """Return f(t): the solved amount in compartment `row`, linearly interpolated to any time."""
    return lambda t: np.interp(t, solution.t, solution.y[row])


def integrate(rhs, initial, time_min, atol, args=()):
    """Solve an ODE system and stop with a clear error if the solver fails."""
    solution = solve_ivp(rhs, (time_min[0], time_min[-1]), initial, t_eval=time_min,
                         args=args, method="LSODA", rtol=1e-8, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


# =====================================================================
# 2. MODEL 1 - TPT PHARMACOKINETICS   (amounts in mmol)
# =====================================================================
def TPT_model(t, amounts):
    injection, blood, kidney, liver, bone = amounts[:5]   # amounts[5] is the mass-balance tracer

    # Injection site: TPT leaves into blood.
    d_injection = -tpt.k_diffuse_TPT * injection

    # Blood: + from injection site, + endogenous PTH, + returns from kidney (0) and bone,
    #        - to kidney, liver, bone and other tissues.
    d_blood = (
        tpt.k_diffuse_TPT * injection
        + tpt.PTH_endogenous_blood_mmol_per_min
        + tpt.k_kidney_to_blood_TPT * kidney
        + tpt.k_bone_to_blood_TPT * bone
        - (tpt.k_blood_to_kidney_TPT + tpt.k_bloodtoliver_TPT
           + tpt.k_blood_to_bone_TPT + tpt.k_blood_to_other_tissue) * blood
    )

    # Kidney: + from blood, - back to blood, - degradation (ASSUMPTION from tpt.py: same rate as blood -> kidney).
    d_kidney = (
        tpt.k_blood_to_kidney_TPT * blood
        - (tpt.k_kidney_to_blood_TPT + tpt.k_kidney_degrade_TPT) * kidney
    )

    # Liver: + from blood, - degradation (ASSUMPTION from tpt.py: same rate as blood -> liver).
    d_liver = tpt.k_bloodtoliver_TPT * blood - tpt.k_bloodtoliver_TPT * liver

    # Bone: + from blood, - back to blood, - lost to bone receptors (internalised).
    d_bone = (
        tpt.k_blood_to_bone_TPT * blood
        - (tpt.k_bone_to_blood_TPT + tpt.TPT_to_bone_receptor) * bone
    )
    # MASS-BALANCE TRACER: net external flux = endogenous input - every real loss. Integrated by the solver itself,
    # so the check below has no trapezoid/interpolation error. Internal transfers (blood<->kidney, blood<->bone,
    # injection -> blood) cancel in the sum and are deliberately NOT listed here.
    d_net = tpt.PTH_endogenous_blood_mmol_per_min - (
        tpt.k_kidney_degrade_TPT * kidney + tpt.k_bloodtoliver_TPT * liver
        + tpt.k_blood_to_other_tissue * blood + tpt.TPT_to_bone_receptor * bone)
    return [d_injection, d_blood, d_kidney, d_liver, d_bone, d_net]


def dose_times_min(time_min, n_doses=None):
    """Injection times (min) inside the simulated window."""
    times = np.arange(N_DOSES if n_doses is None else n_doses) * DOSE_INTERVAL_MIN
    return times[times < time_min[-1]]


def solve_TPT(time_min, injection_scale, n_doses=None):
    """injection_scale = 1: injected run;  0: control (endogenous PTH only).
    n_doses = None: the N_DOSES schedule;  1: a single dose (used for calibration).
    The solver is restarted at every dose; each new dose is added to the injection site AND to the mass-balance
    tracer (the last state), because it is an external input to the body."""
    dose_amount = tpt.initial_TPT_injection * injection_scale
    schedule = dose_times_min(time_min, n_doses)
    state = list(tpt.initial_TPT) + [0.0]
    state[0] = dose_amount if len(schedule) > 0 else 0.0                # first dose is already in the starting amount
    bounds = list(schedule[1:]) + [time_min[-1]]
    t_start, times, amounts = time_min[0], [], []
    for i, t_end in enumerate(bounds):
        seg_t = np.unique(np.concatenate(([t_start], time_min[(time_min >= t_start) & (time_min <= t_end)], [t_end])))
        seg = integrate(TPT_model, state, seg_t, atol=1e-18)
        last = i == len(bounds) - 1
        times.append(seg.t if last else seg.t[:-1])                    # the boundary sample reappears, post-dose, as the next start
        amounts.append(seg.y if last else seg.y[:, :-1])
        state = seg.y[:, -1].copy()
        if not last:
            state[0] += dose_amount
            state[5] += dose_amount
            t_start = t_end
    t_all, y_all = np.concatenate(times), np.concatenate(amounts, axis=1)
    keep = np.isin(t_all, time_min)                                    # drop helper samples that are not on the output grid
    return SimpleNamespace(t=t_all[keep], y=y_all[:, keep], success=True)


# =====================================================================
# 3. MODEL 2 - CALCITRIOL   (amounts in mg)
# =====================================================================
def calcitriol_model(t, amounts, TPT_kidney_at, max_extra_production, H_kidney):
    kidney, blood, liver, gut = amounts[:4]               # amounts[4] is the mass-balance tracer
    A = amounts[5]                                        # relative 1-alpha-hydroxylase activity (dimensionless, 0 -> 1)
    k_degrade_kidney = calcitriol.k_degrade_calcitriol   # from the whole-body half-life (calcitriol.py)
    k_blood_to_gut = calcitriol.k_blood_to_gut_calcitriol if INCLUDE_BLOOD_TO_GUT_CALCITRIOL else 0.0

    # TPT in kidney induces the enzyme 1-alpha-hydroxylase (Hill-driven, first-order turnover); the enzyme makes calcitriol.
    # ASSUMPTION: M = max extra production = k_increase * H, so k_increase is the slope at low TPT (n = 1).
    k_hyd = math.log(2) / HYDROXYLASE_HALF_LIFE_MIN
    dA = k_hyd * (hill(TPT_kidney_at(t), H_kidney, calcium.n) - A)
    TPT_stimulated_production = max_extra_production * A     # M * A

    # Kidney: + basal production, + TPT-stimulated production, + from blood, - to blood, - degradation.
    d_kidney = (
        calcitriol.basal_calcitriol_produce_kidney
        + TPT_stimulated_production
        + calcitriol.k_blood_to_kidney_calcitriol * blood
        - (calcitriol.k_kidney_to_blood_calcitriol + k_degrade_kidney) * kidney
    )
    # Blood: + from kidney, liver, gut; - to kidney, liver (and gut if switched on).
    d_blood = (
        calcitriol.k_kidney_to_blood_calcitriol * kidney
        + calcitriol.K_liver_to_blood_calcitriol * liver
        + calcitriol.k_gut_to_blood_calcitriol * gut
        - (calcitriol.k_blood_to_kidney_calcitriol + calcitriol.k_blood_to_liver_calcitriol + k_blood_to_gut) * blood
    )
    # Liver: + from blood, - back to blood (reabsorption), - to gut.
    d_liver = (
        calcitriol.k_blood_to_liver_calcitriol * blood
        - (calcitriol.K_liver_to_blood_calcitriol + calcitriol.k_liver_to_gut_calcitriol) * liver
    )
    # Gut: + from liver (bile) and blood (if switched on), - to blood.
    d_gut = (
        calcitriol.k_liver_to_gut_calcitriol * liver
        + k_blood_to_gut * blood
        - calcitriol.k_gut_to_blood_calcitriol * gut
    )
    # MASS-BALANCE TRACER: production in minus degradation (the only real source and sink).
    d_net = calcitriol.basal_calcitriol_produce_kidney + TPT_stimulated_production - k_degrade_kidney * kidney
    return [d_kidney, d_blood, d_liver, d_gut, d_net, dA]


def solve_calcitriol(time_min, TPT_solution, k_increase, H_kidney):
    return integrate(calcitriol_model, list(calcitriol.initial_calcitriol) + [0.0, 0.0], time_min, atol=1e-14,
                     args=(at_time(TPT_solution, 2), k_increase * H_kidney, H_kidney))


# =====================================================================
# 4. MODEL 3 - CALCIUM   (amounts in mg)
# =====================================================================
def alp_multiplier(blood_Ca_mg):
    """ALP relative to baseline (1 = baseline): +ALP_MAX_INCREASE at/above the full-effect Ca rise, linear below it."""
    if not ALP_FEEDS_BONE_FORMATION:
        return 1.0 + 0.0 * np.asarray(blood_Ca_mg, dtype=float)
    rise = np.maximum(blood_Ca_mg - calcium.initial_blood_Ca, 0.0) / tpt.Vol_blood_in_bone / 10   # mg/dL above baseline
    return 1.0 + ALP_MAX_INCREASE * np.minimum(rise / ALP_FULL_EFFECT_RISE_MG_PER_DL, 1.0)


def calcium_rates(TPT_kidney, TPT_bone, calcitriol_blood_mg, H, blood_Ca_mg):
    """Every time-varying calcium rate constant (min^-1), in one place.
    Works for single values (inside the ODE) and for whole arrays (diagnostics and plots)."""
    # (a) Calcitriol in blood boosts gut absorption.  E = 1 at the basal calcitriol level.
    calcitriol_concen = calcitriol_blood_mg / tpt.Vol_blood_in_bone          # mg/L, same unit as the basal level
    calcitriol_impact_gut_absorb = np.maximum(
        1 + CA_EFFECT_SCALE * calcium.calcitriol_sensitivity * (calcitriol_concen / calcium.calcitriol_basal_level - 1), 0.0)
    k_gut_blood_abs = (calcium.basal_absorb_Ca * calcitriol_impact_gut_absorb
                       * calcium.gut_absorption_turnover_rate)
    k_gut_waste = (1 - calcium.basal_absorb_Ca) * calcium.gut_waste_turnover_rate

    # (b) TPT in kidney boosts calcium reabsorption (Hill).
    hill_kidney = hill(TPT_kidney, H["kidney_reabs"], calcium.n)
    # TPT lowers the fraction of kidney Ca lost to urine (99% is already reabsorbed, so only the 1% can change)
    frac_excreted = (1 - calcium.basal_kidney_reabsorb_Ca) * (
        1 - CA_EFFECT_SCALE * calcium.max_frac_renalreabs_TPT_impact * hill_kidney)
    k_kidney_urine_loss = frac_excreted * calcium.kidney_turnover_rate  # min^-1
    k_kidney_blood_reabs = (1 - frac_excreted) * calcium.kidney_turnover_rate  # min^-1

    # (c) TPT in bone boosts calcium release from bone (Hill).
    hill_bone = hill(TPT_bone, H["bone"], calcium.n)
    k_bone_to_blood_Ca_with_TPT = calcium.k_bone_to_blood_Ca * (1 + CA_EFFECT_SCALE * calcium.ca_release_incr_TPT * hill_bone)

    # (d) Elevated blood Ca raises ALP, which speeds bone formation (blood -> bone).
    alp = alp_multiplier(blood_Ca_mg)
    k_blood_to_bone_with_ALP = calcium.k_blood_to_bone_Ca * alp

    return dict(gut_abs=k_gut_blood_abs, alp_mult=alp, bone_formation=k_blood_to_bone_with_ALP, gut_waste=k_gut_waste, kidney_reabs=k_kidney_blood_reabs,
                urine=k_kidney_urine_loss, bone_release=k_bone_to_blood_Ca_with_TPT,
                calcitriol_impact=calcitriol_impact_gut_absorb, hill_kidney=hill_kidney, hill_bone=hill_bone)


def calcium_model(t, amounts, TPT_kidney_at, TPT_bone_at, calcitriol_blood_at, H):
    intake, gut, blood, kidney, bone = amounts[:5]        # amounts[5] is the mass-balance tracer
    k = calcium_rates(TPT_kidney_at(t), TPT_bone_at(t), calcitriol_blood_at(t), H, blood)

    # Intake (food) -> gut.
    excess = blood - calcium.initial_blood_Ca
    if FEEDBACK_ONLY_ABOVE_BASELINE:
        excess = max(excess, 0.0)                        # never create calcium when blood Ca is below baseline
    d_intake = calcium.Ca_intake_rate - calcium.k_speed_to_gut_transfer * intake
    # Gut: + from intake, - absorbed into blood, - lost as waste.
    d_gut = calcium.k_speed_to_gut_transfer * intake - (k["gut_abs"] + k["gut_waste"]) * gut
    # Blood: + from gut, kidney (reabsorbed) and bone (release); - to kidney and to bone (formation).
    # Bone formation flux = k_blood_to_bone_Ca * ALP multiplier * blood Ca (more Ca around bone and more ALP -> more formation).
    d_blood = (
        k["gut_abs"] * gut + k["kidney_reabs"] * kidney + k["bone_release"] * bone
        - (calcium.k_blood_kidney_Ca + k["bone_formation"]) * blood
        - calcium.k_Ca_feedback * excess                 # homeostasis: extra urine loss of EXCESS Ca
    )
    # Kidney: + from blood, - reabsorbed to blood, - lost in urine.
    d_kidney = calcium.k_blood_kidney_Ca * blood - (k["kidney_reabs"] + k["urine"]) * kidney
    # Bone: + formation from blood, - release to blood.
    d_bone = k["bone_formation"] * blood - k["bone_release"] * bone
    # MASS-BALANCE TRACER: food in minus waste, urine and homeostatic loss (the only real source and sinks).
    d_net = calcium.Ca_intake_rate - k["gut_waste"] * gut - k["urine"] * kidney - calcium.k_Ca_feedback * excess
    return [d_intake, d_gut, d_blood, d_kidney, d_bone, d_net]


def solve_calcium(time_min, TPT_solution, calcitriol_solution, H):
    return integrate(calcium_model, list(calcium.initial_calcium) + [0.0], time_min, atol=1e-6,
                     args=(at_time(TPT_solution, 2), at_time(TPT_solution, 4), at_time(calcitriol_solution, 1), H))


# =====================================================================
# 5. SIMULATION DRIVER
# =====================================================================
def half_saturation(TPT_treated):
    """Hill half-saturation constants in mmol TPT, per organ / effect."""
    if HALF_SATURATION_MODE == "table":
        H = {"kidney": calcium.half_saturation_concen, "bone": calcium.half_saturation_concen}
    else:
        H = {"kidney": HALF_SATURATION_FRACTION * TPT_treated.y[2].max(),
             "bone": HALF_SATURATION_FRACTION * TPT_treated.y[4].max()}
    # Separate constant for kidney calcium reabsorption (defaults to the kidney value).
    if KIDNEY_REABS_H_MMOL is not None:
        H["kidney_reabs"] = KIDNEY_REABS_H_MMOL
    elif KIDNEY_REABS_H_FRACTION is not None:
        H["kidney_reabs"] = KIDNEY_REABS_H_FRACTION * TPT_treated.y[2].max()
    else:
        H["kidney_reabs"] = H["kidney"]
    return H


def calibrate_calcitriol_coefficient(time_min, TPT_treated, TPT_control, H):
    """Find k_increase_calcitriol_produce_by_TPT so peak (treated / control) blood calcitriol = target."""
    def peak_ratio(k_increase):
        treated = solve_calcitriol(time_min, TPT_treated, k_increase, H["kidney"])
        control = solve_calcitriol(time_min, TPT_control, k_increase, H["kidney"])
        return np.max(treated.y[1] / control.y[1])

    try:
        return brentq(lambda k: peak_ratio(k) - CALCITRIOL_RISE_TARGET, 1e-2, 1e9, xtol=1e-3, rtol=1e-6), True
    except ValueError:
        # calcitriol.py leaves this value as None, so stop with a clear message instead of a TypeError.
        if calcitriol.k_increase_calcitriol_produce_by_TPT is None:
            raise RuntimeError("Calibration of k_increase_calcitriol_produce_by_TPT found no solution for "
                               f"a peak calcitriol ratio of {CALCITRIOL_RISE_TARGET}, and calcitriol.py has no value "
                               "to fall back on. Set one in calcitriol.py or change CALCITRIOL_RISE_TARGET.")
        return calcitriol.k_increase_calcitriol_produce_by_TPT, False


def calibrate_calcium_effect(time_min, TPT_reference, TPT_control, calcitriol_reference, calcitriol_control, H):
    """Find CA_EFFECT_SCALE so the reference dose raises blood Ca by LABEL_CALCIUM_PEAK_RISE_MG_PER_DL at its peak."""
    global CA_EFFECT_SCALE
    control = solve_calcium(time_min, TPT_control, calcitriol_control, H)   # no TPT, so the scale has no effect

    def peak_rise(scale):
        global CA_EFFECT_SCALE
        CA_EFFECT_SCALE = scale
        treated = solve_calcium(time_min, TPT_reference, calcitriol_reference, H)
        return np.max(treated.y[2] - control.y[2]) / tpt.Vol_blood_in_bone / 10   # mg/dL

    CA_EFFECT_SCALE = brentq(lambda s: peak_rise(s) - LABEL_CALCIUM_PEAK_RISE_MG_PER_DL, 1e-3, 10, xtol=1e-4)
    return CA_EFFECT_SCALE


def run_simulation(fixed=None, dose_scale=1.0):
    """dose_scale: dose relative to the 20 ug reference for the treated run (from apply_conditions).
    fixed = None: calibrate H, k_increase and CA_EFFECT_SCALE at the reference dose.
    fixed = dict(H=..., k_increase=..., ca_scale=...): reuse those values instead (Monte Carlo: calibrated once,
    on the default draw, then held fixed so the parameter uncertainty shows up in the results)."""
    global CA_EFFECT_SCALE
    time_min = np.arange(0, SIM_MINUTES + STEP_MIN, STEP_MIN)

    # --- Model 1: TPT, treated and control -------------------------------------------------
    TPT_reference = solve_TPT(time_min, 1.0, n_doses=1)   # ONE standard 20 ug dose; ONLY used to set H and the calibrations
    TPT_treated = solve_TPT(time_min, dose_scale)     # the dose for this case state
    TPT_control = solve_TPT(time_min, 0.0)            # no injection
    H = half_saturation(TPT_reference) if fixed is None else fixed["H"]   # set at the reference dose, then held fixed

    # --- Model 2: calcitriol, treated and control ------------------------------------------
    k_increase, calibrated = calcitriol.k_increase_calcitriol_produce_by_TPT, False
    if k_increase is None and not CALIBRATE_CALCITRIOL_TPT:
        raise ValueError("CALIBRATE_CALCITRIOL_TPT is False but calcitriol.k_increase_calcitriol_produce_by_TPT is None.")
    if fixed is not None:
        k_increase, calibrated = fixed["k_increase"], True
    elif CALIBRATE_CALCITRIOL_TPT:   # calibrated at the REFERENCE dose, not this state's dose
        k_increase, calibrated = calibrate_calcitriol_coefficient(time_min, TPT_reference, TPT_control, H)
    calcitriol_treated = solve_calcitriol(time_min, TPT_treated, k_increase, H["kidney"])
    calcitriol_control = solve_calcitriol(time_min, TPT_control, k_increase, H["kidney"])

    # --- Model 3: calcium, treated and control ---------------------------------------------
    if fixed is not None:
        CA_EFFECT_SCALE = fixed["ca_scale"]
    elif CALIBRATE_CA_EFFECT:        # calibrated at the REFERENCE dose, not this state's dose
        calcitriol_reference = solve_calcitriol(time_min, TPT_reference, k_increase, H["kidney"])
        calibrate_calcium_effect(time_min, TPT_reference, TPT_control, calcitriol_reference, calcitriol_control, H)
    calcium_treated = solve_calcium(time_min, TPT_treated, calcitriol_treated, H)
    calcium_control = solve_calcium(time_min, TPT_control, calcitriol_control, H)

    return dict(time=time_min, n_doses=len(dose_times_min(time_min)), H=H, k_increase=k_increase, calibrated=calibrated, ca_scale=CA_EFFECT_SCALE,
                TPT=TPT_treated, TPT_control=TPT_control,
                calcitriol=calcitriol_treated, calcitriol_control=calcitriol_control,
                calcium=calcium_treated, calcium_control=calcium_control)


# =====================================================================
# 6. OUTPUTS AND CHECKS
# =====================================================================
def compute_outputs(sim):
    t, H = sim["time"], sim["H"]
    T, Tc = sim["TPT"], sim["TPT_control"]
    C, Cc = sim["calcitriol"], sim["calcitriol_control"]
    A, Ac = sim["calcium"], sim["calcium_control"]
    out = {}

    # TPT outputs (mmol/L; ng/mL: mmol/L * mg/mmol = mg/L, and 1 mg/L = 1000 ng/mL)
    out["TPT_blood_concen_mmol_per_L"] = T.y[1] / tpt.Vol_blood_in_bone
    out["TPT_blood_concen_ng_per_mL"] = out["TPT_blood_concen_mmol_per_L"] * tpt.mol_weight_TPT * 1000
    out["TPT_peak_blood_concen_mmol_per_L"] = out["TPT_blood_concen_mmol_per_L"].max()
    first = t <= DOSE_INTERVAL_MIN                                   # peak timing is read from the first dose only
    out["TPT_time_to_peak_min"] = t[first][np.argmax(out["TPT_blood_concen_mmol_per_L"][first])]
    out["TPT_blood_AUC_mmol_min_per_L"] = TRAPEZOID(out["TPT_blood_concen_mmol_per_L"], t)
    out["total_TPT_mmol"] = T.y[:5].sum(axis=0)          # row 5 is the mass-balance tracer, not a compartment
    out["TPT_peak_kidney_mmol"] = T.y[2].max()
    out["TPT_time_to_kidney_peak_min"] = t[first][np.argmax(T.y[2][first])]
    out["TPT_kidney_ng"] = T.y[2] * tpt.mol_weight_TPT * 1e6   # mmol * mg/mmol = mg; 1 mg = 1e6 ng
    out["TPT_kidney_ng_per_mL"] = out["TPT_kidney_ng"] / vol("kidney") / 1000   # ng/L -> ng/mL

    # Calcitriol and calcium concentrations, treated vs control
    out["calcitriol_blood_pg_per_mL"] = C.y[1] / tpt.Vol_blood_in_bone * 1e6
    out["calcitriol_blood_control_pg_per_mL"] = Cc.y[1] / tpt.Vol_blood_in_bone * 1e6
    out["Ca_blood_concen_mg_per_L"] = A.y[2] / tpt.Vol_blood_in_bone
    out["Ca_blood_control_mg_per_L"] = Ac.y[2] / tpt.Vol_blood_in_bone
    out["Ca_blood_TPT_effect_mg_per_dL"] = (out["Ca_blood_concen_mg_per_L"] - out["Ca_blood_control_mg_per_L"]) / 10
    out["bone_Ca_levels_TPT_effect"] = A.y[4] - Ac.y[4]   # treated minus control

    # Rates over time (for plots)
    out["rates"] = calcium_rates(T.y[2], T.y[4], C.y[1], H, A.y[2])
    out["rates_control"] = calcium_rates(Tc.y[2], Tc.y[4], Cc.y[1], H, Ac.y[2])

    # Bone repair (formation) and resorption (release) fluxes, mg/min, and their running totals, treated vs control
    rc = out["rates_control"]
    out["ALP_increase_pct"] = (out["rates"]["alp_mult"] - 1) * 100
    out["bone_formation_flux"] = out["rates"]["bone_formation"] * A.y[2]
    out["bone_formation_flux_control"] = rc["bone_formation"] * Ac.y[2]
    out["bone_release_flux"] = out["rates"]["bone_release"] * A.y[4]
    out["bone_release_flux_control"] = rc["bone_release"] * Ac.y[4]
    def running_total(flux):
        return np.concatenate(([0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(t))))
    out["extra_formation_mg"] = running_total(out["bone_formation_flux"] - out["bone_formation_flux_control"])
    out["extra_release_mg"] = running_total(out["bone_release_flux"] - out["bone_release_flux_control"])

    # ---- mass-balance checks -------------------------------------------------------------------------
    # Change in total amount must equal the integrated net external flux. The net flux is integrated by the ODE
    # solver (last row of each solution), so the only remaining error is the solver tolerance, not the output grid.
    def balance_error(solution, n_compartments):
        total = solution.y[:n_compartments].sum(axis=0)
        return np.max(np.abs((total - total[0]) - solution.y[n_compartments]))

    r = out["rates"]
    out["check_TPT_mass_error_mmol"] = balance_error(T, 5)
    out["check_calcitriol_mass_error_mg"] = balance_error(C, 4)
    out["check_calcium_mass_error_mg"] = balance_error(A, 5)
    # relative to the amount in the system, so the three models can be compared
    out["check_TPT_rel"] = out["check_TPT_mass_error_mmol"] / (T.y[0, 0] if T.y[0, 0] > 0 else T.y[:5, 0].sum())
    out["check_calcitriol_rel"] = out["check_calcitriol_mass_error_mg"] / C.y[:4, 0].sum()
    out["check_calcium_rel"] = out["check_calcium_mass_error_mg"] / A.y[:5, 0].sum()
    production = (calcitriol.basal_calcitriol_produce_kidney
                  + sim["k_increase"] * H["kidney"] * C.y[5])   # M * A, same as the ODE
    out["hydroxylase_A"] = C.y[5]
    out["rates"]["production_calcitriol"] = production
    return out


def print_summary(sim, out):
    H, t = sim["H"], sim["time"]
    base = out["rates_control"]
    print("=" * 78)
    print(f"Hill half-saturation mode: {HALF_SATURATION_MODE}  (kidney {H['kidney']*1e9:.3g} pmol, bone {H['bone']*1e9:.3g} pmol, kidney reabsorption {H['kidney_reabs']*1e9:.3g} pmol)")
    print(f"  Hill value with NO injection (kidney, bone): {base['hill_kidney'][-1]:.2f}, {base['hill_bone'][-1]:.2f}"
          f"   | at the injected peak: {out['rates']['hill_kidney'].max():.2f}, {out['rates']['hill_bone'].max():.2f}")
    if base["hill_kidney"][-1] > 0.9:
        print("  WARNING: Hill terms are saturated without any injection, so TPT effects cannot differ between runs.")
    print(f"TPT: peak blood {out['TPT_peak_blood_concen_mmol_per_L']*tpt.mol_weight_TPT*1000:.3f} ng/mL at {out['TPT_time_to_peak_min']} min")
    status = "calibrated" if sim["calibrated"] else "NOT calibrated (no solution found); using the value from calcitriol.py"
    print(f"Calcitriol: k_increase_calcitriol_produce_by_TPT = {sim['k_increase']:.4g} ({status}; file value {calcitriol.k_increase_calcitriol_produce_by_TPT})")
    ratio = out['calcitriol_blood_pg_per_mL'] / out['calcitriol_blood_control_pg_per_mL']
    print(f"  1-alpha-hydroxylase: half-life {HYDROXYLASE_HALF_LIFE_MIN/60:.1f} h | peak activity A = {out['hydroxylase_A'].max():.2f} at {t[np.argmax(out['hydroxylase_A'])]/60:.1f} h")
    print(f"  peak treated/control, first dose = {np.max(ratio[t <= DOSE_INTERVAL_MIN]):.2f} (target {CALCITRIOL_RISE_TARGET}); over all doses {np.max(ratio):.2f}")
    print(f"  CONTROL drift: blood calcitriol {out['calcitriol_blood_control_pg_per_mL'][0]:.1f} -> {out['calcitriol_blood_control_pg_per_mL'][-1]:.1f} pg/mL over the run (basal level {calcium.calcitriol_basal_level*1e6:.1f})")
    ca_c = out["Ca_blood_control_mg_per_L"]
    print(f"Calcium: CONTROL drift: blood Ca {ca_c[0]/10:.2f} -> {ca_c[-1]/10:.2f} mg/dL over the run")
    effect = out["Ca_blood_TPT_effect_mg_per_dL"]
    first, final = t <= DOSE_INTERVAL_MIN, t >= t[-1] - DOSE_INTERVAL_MIN
    print(f"  TPT effect on blood Ca (treated - control), first dose: peak {effect[first].max():+.3f} mg/dL at {t[first][np.argmax(effect[first])]/60:.1f} h (label ~ +{LABEL_CALCIUM_PEAK_RISE_MG_PER_DL})")
    print(f"  Doses: {sim['n_doses']} x every {DOSE_INTERVAL_MIN/60:.0f} h | final cycle: Ca rise peak {effect[final].max():+.3f}, trough {effect[final].min():+.3f} mg/dL"
          f" | blood calcitriol peak {out['calcitriol_blood_pg_per_mL'][final].max():.1f}, trough {out['calcitriol_blood_pg_per_mL'][final].min():.1f} pg/mL")
    bone_eff = out['bone_Ca_levels_TPT_effect']
    print(f"  Bone Ca gained per day in the final cycle: {bone_eff[-1] - bone_eff[final][0]:+.1f} mg")
    print(f"  TPT effect on bone Ca after {t[-1]/1440:.0f} days: {out['bone_Ca_levels_TPT_effect'][-1]:+.1f} mg "
          f"(extra formation {out['extra_formation_mg'][-1]:+.1f} mg, extra release {out['extra_release_mg'][-1]:.1f} mg lost; "
          f"peak ALP rise {out['ALP_increase_pct'].max():.0f}%)")
    k_abs_basal = calcium.basal_absorb_Ca * calcium.gut_absorption_turnover_rate       # rate at calcitriol effect = 1
    absorbed_share = k_abs_basal / (k_abs_basal + base["gut_waste"])
    print(f"  Gut: absorbed share at basal calcitriol = {absorbed_share:.2f} (basal_absorb_Ca = {calcium.basal_absorb_Ca:.2f})")
    print(f"  Bone exchange flux = {calcium.k_blood_to_bone_Ca*calcium.initial_blood_Ca*1440/1000:.2f} g Ca/day (physiological remodelling is ~0.3-0.6 g/day)")
    print(f"  Kidney Ca starts at {calcium.initial_kidney_Ca:.2f} mg (steady state); calcitriol effect on gut absorption at baseline = {base['calcitriol_impact'][0]:.2f}")
    print(f"Mass-balance errors: TPT {out['check_TPT_mass_error_mmol']:.2e} mmol | calcitriol {out['check_calcitriol_mass_error_mg']:.2e} mg | calcium {out['check_calcium_mass_error_mg']:.2e} mg")
    print(f"  relative to system total: TPT {out['check_TPT_rel']:.1e} | calcitriol {out['check_calcitriol_rel']:.1e} | calcium {out['check_calcium_rel']:.1e}")
    print("=" * 78)


# =====================================================================
# 7. PLOTS
# =====================================================================
def make_plots_daily(sim, out):
    """Figures 1-2 (TPT, calcitriol) from the daily-dosing run. X-axis in MINUTES."""
    t, T, Tc = sim["time"], sim["TPT"], sim["TPT_control"]
    C = sim["calcitriol"]
    xmax = PLOT_DETAIL_DAYS * 1440 if PLOT_DETAIL_DAYS else t[-1]
    doses = dose_times_min(t, sim["n_doses"])                  # minutes
    figures = {}

    def finish(axes, xlabel="Time since first injection (min)"):
        """Common x-range, label and faint vertical lines at each injection."""
        for a in np.ravel(axes):
            a.set_xlim(0, xmax)
            a.set_xlabel(xlabel)
            for x in doses[doses <= xmax]:
                a.axvline(x, color="gray", lw=0.4, alpha=0.35, zorder=0)

    # ---- Figure 1: TPT pharmacokinetics ----------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for index, (key, label) in enumerate([("injection", "Injection site"), ("blood", "Blood"), ("kidney", "Kidney"),
                                          ("liver", "Liver"), ("bone", "Bone")]):
        ax[0, 0].plot(t, T.y[index] / vol(key) * tpt.mol_weight_TPT * 1000, label=label)   # mmol/L * mg/mmol * 1000 = ng/mL
    ax[0, 0].set(ylabel="TPT concentration in each compartment (ng/mL)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, out["TPT_blood_concen_ng_per_mL"])
    peak_c = np.max(out["TPT_blood_concen_ng_per_mL"])
    ax[0, 1].set(ylabel="TPT concentration in blood (ng/mL)")
    ax[1, 0].plot(t, out["TPT_kidney_ng_per_mL"])
    ax[1, 0].set(ylabel="TPT concentration in kidney (ng/mL)")
    ax[1, 1].plot(t, out["rates"]["hill_kidney"], label="kidney Hill (treated)")
    ax[1, 1].plot(t, out["rates"]["hill_bone"], label="bone Hill (treated)")
    ax[1, 1].plot(t, out["rates_control"]["hill_kidney"], "--", label="kidney Hill (control)")
    ax[1, 1].plot(t, out["rates_control"]["hill_bone"], "--", label="bone Hill (control)")
    ax[1, 1].plot(t, out["hydroxylase_A"], "k", label="1-alpha-hydroxylase activity A")
    ax[1, 1].set(ylabel="TPT effect signal (Hill fraction) / enzyme activity A (0 to 1)", ylim=(-0.02, 1.05))
    ax[1, 1].legend(fontsize=8)
    finish(ax)
    sci_axes(fig); fig.tight_layout(); figures["1_TPT"] = fig

    # ---- Figure 2: calcitriol ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for index, (key, label) in enumerate([("kidney", "Kidney"), ("blood", "Blood"), ("liver", "Liver"), ("gut", "Gut")]):
        ax[0].plot(t, C.y[index] / vol(key) * 1e6, label=label)         # mg/L -> pg/mL
    ax[0].set(ylabel="Calcitriol concentration in each compartment (pg/mL)"); ax[0].legend(fontsize=8)
    ax[1].plot(t, out["calcitriol_blood_pg_per_mL"], label="treated")
    ax[1].plot(t, out["calcitriol_blood_control_pg_per_mL"], "--", label="control")
    ax[1].axhline(calcium.calcitriol_basal_level * 1e6, color="gray", lw=0.7, label="basal level")
    ax[1].set(ylabel="Calcitriol concentration in blood (pg/mL)"); ax[1].legend(fontsize=8)
    ax[2].plot(t, out["rates"]["calcitriol_impact"], label="treated")
    ax[2].plot(t, out["rates_control"]["calcitriol_impact"], "--", label="control")
    ax[2].set(ylabel="Calcitriol effect on gut Ca absorption\n(multiplier of baseline)"); ax[2].legend(fontsize=8)
    finish(ax)
    sci_axes(fig); fig.tight_layout(); figures["2_calcitriol"] = fig

    return figures


def make_plots_single(sim, out):
    """Figures 3-4 from the daily-dosing run. Figure 3 (calcium): DAYS. Figure 4 (bone repair): HOURS."""
    t = sim["time"]
    A, Ac = sim["calcium"], sim["calcium_control"]
    d = t / 1440                                   # days
    hr = t / 60                                    # hours
    detail_days = SINGLE_DOSE_DETAIL_HOURS / 24    # zoom window for the fast calcium panels
    bone_xmax = BONE_PLOT_HOURS if BONE_PLOT_HOURS else hr[-1]
    figures = {}
    to_conc = 1 / vol("bone") / 10                  # mg -> mg/dL in bone

    # ---- Figure 3: calcium (days) -----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for index, (key, label) in enumerate([("intake", "Intake"), ("gut", "Gut"), ("blood", "Blood"), ("kidney", "Kidney")]):
        ax[0, 0].plot(d, A.y[index] / vol(key) / 10, label=label)       # mg/L -> mg/dL
    ax[0, 0].set(xlabel="Time since first injection (days)", ylabel="Calcium concentration in each compartment (mg/dL)", xlim=(0, detail_days))
    ax[0, 0].legend(fontsize=8)
    bone_treated, bone_control = A.y[4] / vol("bone") / 10, Ac.y[4] / vol("bone") / 10      # mg/dL
    ax[0, 1].plot(d, bone_treated, label="treated")
    ax[0, 1].plot(d, bone_control, "--", label="control")
    lo, hi = min(bone_treated.min(), bone_control.min()), max(bone_treated.max(), bone_control.max())
    pad = 0.15 * (hi - lo) or 1e-3 * abs(hi) or 1.0
    ax[0, 1].set_ylim(lo - pad, hi + pad)
    ax[0, 1].set(xlabel="Time since first injection (days)", ylabel="Calcium concentration in bone (mg/dL)")
    ax[0, 1].legend(fontsize=8)
    for x in range(N_DOSES):
        ax[0, 1].axvline(x, color="gray", lw=0.4, alpha=0.4, zorder=0)
    
    ax[1, 0].plot(d, out["Ca_blood_concen_mg_per_L"] / 10, label="treated")
    ax[1, 0].plot(d, out["Ca_blood_control_mg_per_L"] / 10, "--", label="control")
    ax[1, 0].set(xlabel="Time since first injection (days)", ylabel="Calcium concentration in blood (mg/dL)", xlim=(0, detail_days)); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(d, out["Ca_blood_TPT_effect_mg_per_dL"])
    ax[1, 1].axhline(LABEL_CALCIUM_PEAK_RISE_MG_PER_DL, color="gray", ls=":", label="label peak (+0.4)")
    ax[1, 1].set(xlabel="Time since first injection (days)", ylabel="TPT-driven rise in blood calcium\n(treated minus control, mg/dL)", xlim=(0, detail_days))
    ax[1, 1].legend(fontsize=8)
    sci_axes(fig); fig.tight_layout(); figures["3_calcium"] = fig
    # ---- Figure 4: bone repair (ALP-driven formation) vs resorption (hours) ------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(hr, out["ALP_increase_pct"], color="tab:green", label="ALP increase (%)")
    ax[0, 0].set(xlabel="Time since first injection (h)",
                 ylabel="ALP increase vs baseline (%)")
    twin = ax[0, 0].twinx()
    twin.plot(hr, out["Ca_blood_TPT_effect_mg_per_dL"], color="tab:red", ls="--", label="blood Ca rise")
    twin.set_ylabel("TPT-driven rise in blood calcium (mg/dL)")
    ax[0, 0].legend(loc="upper left", fontsize=8); twin.legend(loc="upper right", fontsize=8)
    ax[0, 1].plot(hr, out["bone_formation_flux"] * 60 * to_conc, label="formation (repair), treated")
    ax[0, 1].plot(hr, out["bone_formation_flux_control"] * 60 * to_conc, "--", label="formation, control")
    ax[0, 1].plot(hr, out["bone_release_flux"] * 60 * to_conc, label="release (resorption), treated")
    ax[0, 1].plot(hr, out["bone_release_flux_control"] * 60 * to_conc, "--", label="release, control")
    ax[0, 1].set(xlabel="Time since first injection (h)", ylabel="Bone calcium flux (mg/dL per hour)")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(hr, out["extra_formation_mg"] * to_conc, label="extra formation (repair)")
    ax[1, 0].plot(hr, -out["extra_release_mg"] * to_conc, label="extra release (resorption), shown negative")
    ax[1, 0].plot(hr, out["bone_Ca_levels_TPT_effect"] * to_conc, "k:", label="net change in bone Ca")
    ax[1, 0].axhline(0, color="gray", lw=0.7)
    ax[1, 0].set(xlabel="Time since first injection (h)", ylabel="Cumulative TPT-driven change in\nbone calcium concentration (mg/dL)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(hr, out["bone_Ca_levels_TPT_effect"] * to_conc)
    ax[1, 1].axhline(0, color="gray", lw=0.7)
    ax[1, 1].set(xlabel="Time since first injection (h)", ylabel="Net TPT effect on bone calcium concentration\n(treated minus control, mg/dL)")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlim(0, bone_xmax)
    twin.set_xlim(0, bone_xmax)
    ax[1, 1].set_xlim(0, hr[-1])                       # net bone calcium over all days
    for x in range(0, int(hr[-1]) + 1, 24):
        ax[1, 1].axvline(x, color="gray", lw=0.4, alpha=0.4, zorder=0)

    sci_axes(fig); fig.tight_layout(); figures["4_bone"] = fig
    return figures


# =====================================================================
# 7b. SENSITIVITY: 1-alpha-hydroxylase half-life
# =====================================================================
def run_hydroxylase_sweep():
    """Healthy state, reference dose: repeat the full calibrate-and-run for each enzyme half-life in
    HYDROXYLASE_SWEEP_HOURS (k_increase and CA_EFFECT_SCALE are re-calibrated each time, so every case still hits the
    calcitriol and label-calcium targets) and report how the time course changes. Saves one figure."""
    global HYDROXYLASE_HALF_LIFE_MIN
    saved_half_life = HYDROXYLASE_HALF_LIFE_MIN
    apply_conditions("healthy", seed=0)
    rows, curves = [], {}
    try:
        for hours in HYDROXYLASE_SWEEP_HOURS:
            HYDROXYLASE_HALF_LIFE_MIN = hours * 60
            sim = run_simulation()
            out = compute_outputs(sim)
            t = sim["time"]
            first = t <= DOSE_INTERVAL_MIN
            ratio = out["calcitriol_blood_pg_per_mL"] / out["calcitriol_blood_control_pg_per_mL"]
            effect = out["Ca_blood_TPT_effect_mg_per_dL"]
            rows.append(dict(hours=hours, k_inc=sim["k_increase"], ratio_peak=ratio[first].max(),
                             ratio_t=t[first][np.argmax(ratio[first])] / 60, ratio_24=ratio[np.searchsorted(t, DOSE_INTERVAL_MIN)],
                             ca_peak=effect[first].max(), ca_t=t[first][np.argmax(effect[first])] / 60,
                             ca_scale=sim["ca_scale"], bone=out["bone_Ca_levels_TPT_effect"][-1]))
            curves[hours] = (t, ratio, effect)
    finally:
        HYDROXYLASE_HALF_LIFE_MIN = saved_half_life
    print("=" * 78)
    print("SENSITIVITY: 1-alpha-hydroxylase half-life (healthy, daily dosing, k_increase and Ca scale re-calibrated each time)")
    print(f"  {'t1/2 (h)':>8} {'k_increase':>11} {'calcitriol peak':>16} {'at (h)':>7} {'ratio@24h':>10} {'Ca peak (mg/dL)':>16} {'at (h)':>7} {'Ca scale':>9} {'bone Ca (mg)':>13}")
    for r in rows:
        print(f"  {r['hours']:>8} {r['k_inc']:>11.4g} {r['ratio_peak']:>16.2f} {r['ratio_t']:>7.1f} {r['ratio_24']:>10.2f} {r['ca_peak']:>16.3f} {r['ca_t']:>7.1f} {r['ca_scale']:>9.3f} {r['bone']:>13.1f}")
    # Figure: calcitriol ratio and Ca rise over the first 3 days, sensitivity band + each half-life
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, idx, label in ((ax[0], 1, "Blood calcitriol, treated / control"), (ax[1], 2, "TPT-driven rise in blood Ca (mg/dL)")):
        t0 = curves[HYDROXYLASE_SWEEP_HOURS[0]][0]
        window = t0 <= 3 * 1440
        stack = np.array([curves[h][idx][window] for h in HYDROXYLASE_SWEEP_HOURS])
        a.fill_between(t0[window] / 60, stack.min(axis=0), stack.max(axis=0), color="tab:blue", alpha=0.2, label="sensitivity band")
        for h in HYDROXYLASE_SWEEP_HOURS:
            a.plot(t0[window] / 60, curves[h][idx][window], lw=2 if h == 15 else 0.9, label=f"t1/2 = {h} h")
        a.set(xlabel="Time since first injection (h)", ylabel=label)
        a.legend(fontsize=7)
    sci_axes(fig)
    fig.tight_layout()
    if SAVE_FIGURES:
        os.makedirs(FIGURE_FOLDER, exist_ok=True)
        fig.savefig(os.path.join(FIGURE_FOLDER, f"fig_6_hydroxylase_sweep.{FIGURE_FORMAT}"), dpi=110)
    plt.close(fig)
    return rows


# =====================================================================
# 8. MONTE CARLO
# =====================================================================
def run_monte_carlo():
    """Calibrate once on the healthy default draw (seed 0) at the reference dose, then re-draw the uncertain
    parameters N_RUNS times and run every case state with the calibration held fixed.
    Each draw uses the same seed for every condition, so differences between conditions come from the condition only
    (grey_market additionally draws its dose from the potency and overdose ranges).
    Prints median [5th-95th percentile] and saves one boxplot of the peak Ca rise."""
    apply_conditions("healthy", seed=0)                     # reloads the modules with the default draw
    nominal = run_simulation()
    fixed = dict(H=nominal["H"], k_increase=nominal["k_increase"], ca_scale=nominal["ca_scale"])
    results = {condition: {"dose (x 20 ug)": [], "TPT peak (ng/mL)": [], "Ca rise peak (mg/dL)": [],
                           "Ca peak time (h)": [], "calcitriol peak (x baseline)": []}
               for condition in MC_CONDITIONS}
    for run in range(N_RUNS):
        for condition in MC_CONDITIONS:
            dose_scale = apply_conditions(condition, seed=run + 1)   # reloads the modules with this draw's seed, applies the condition
            try:
                sim = run_simulation(fixed, dose_scale)
                out = compute_outputs(sim)
            except (ValueError, RuntimeError) as error:   # a draw that cannot be solved is skipped and reported
                print(f"  draw {run} ({condition}) skipped: {error}")
                continue
            effect = out["Ca_blood_TPT_effect_mg_per_dL"]
            first = sim["time"] <= DOSE_INTERVAL_MIN                      # timing of the first-dose peak
            r = results[condition]
            r["dose (x 20 ug)"].append(dose_scale)
            r["TPT peak (ng/mL)"].append(np.max(out["TPT_blood_concen_ng_per_mL"]))
            r["Ca rise peak (mg/dL)"].append(np.max(effect))
            r["Ca peak time (h)"].append(sim["time"][first][np.argmax(effect[first])] / 60 if dose_scale > 0 else np.nan)
            r["calcitriol peak (x baseline)"].append(np.max(out["calcitriol_blood_pg_per_mL"]
                                                            / out["calcitriol_blood_control_pg_per_mL"]))
    print("=" * 78)
    print(f"MONTE CARLO: {N_RUNS} draws, median [5th - 95th percentile]")
    for condition, r in results.items():
        print(f"  {condition}  (n = {len(r['Ca rise peak (mg/dL)'])})")
        for label, values in r.items():
            if np.all(np.isnan(values)):
                print(f"      {label:30s}     n/a  (no injection)")
                continue
            low, mid, high = np.nanpercentile(values, [5, 50, 95])
            print(f"      {label:30s} {mid:7.3f}  [{low:.3f} - {high:.3f}]")

    # One boxplot: one box per case state, coloured by condition
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:red", "tab:gray"]
    box = ax.boxplot([results[condition]["Ca rise peak (mg/dL)"] for condition in MC_CONDITIONS],
                     positions=range(1, len(MC_CONDITIONS) + 1), widths=0.6, whis=(5, 95), patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_xticks(range(1, len(MC_CONDITIONS) + 1))
    ax.set_xticklabels([condition.replace("_", " ") for condition in MC_CONDITIONS])
    ax.axhline(LABEL_CALCIUM_PEAK_RISE_MG_PER_DL, color="gray", ls=":", label="label peak (+0.4)")
    ax.set(xlabel=f"Case state ({N_RUNS} Monte Carlo draws; box = 25-75%, whiskers = 5-95%)",
           ylabel="Peak TPT-driven rise in blood calcium (mg/dL)")
    ax.legend(fontsize=8)
    sci_axes(fig)
    fig.tight_layout()
    if SAVE_FIGURES:
        os.makedirs(FIGURE_FOLDER, exist_ok=True)
        fig.savefig(os.path.join(FIGURE_FOLDER, f"fig_5_monte_carlo.{FIGURE_FORMAT}"), dpi=110)
    return results


# =====================================================================
# 9. RUN
# =====================================================================
if __name__ == "__main__":
    
    conditions = MC_CONDITIONS          # same four case states as the Monte Carlo
    
    for condition in conditions:
        print(f"\n\n{'='*78}\nRUNNING CONDITION: {condition.upper()}\n{'='*78}")
        dose_scale = apply_conditions(condition)
        print(f"Dose = {dose_scale:.3f} x the 20 ug reference")
        
        if condition == "healthy":
            simulation = run_simulation()                  # calibrate once on the healthy state
            fixed = dict(H=simulation["H"], k_increase=simulation["k_increase"], ca_scale=simulation["ca_scale"])
        else:
            simulation = run_simulation(fixed, dose_scale) # other states reuse the healthy calibration

        outputs = compute_outputs(simulation)
        print(f"DAILY DOSING, {SIM_DAYS} days")
        print_summary(simulation, outputs)
            
        # Gather all figures
        figures = {**make_plots_daily(simulation, outputs), **make_plots_single(simulation, outputs)}
        # Save this condition's figures into the figures folder, as PNGs or one PDF (FIGURE_FORMAT in the settings)
        if SAVE_FIGURES:
            os.makedirs(FIGURE_FOLDER, exist_ok=True)
            if FIGURE_FORMAT == "pdf":
                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(os.path.join(FIGURE_FOLDER, f"results_{condition}.pdf")) as pdf:
                    for fig in figures.values():
                        pdf.savefig(fig)
            else:
                for name, fig in figures.items():
                    fig.savefig(os.path.join(FIGURE_FOLDER, f"fig_{name}_{condition}.png"), dpi=110)
            print(f"Saved {condition} figures to {FIGURE_FOLDER}/ as {FIGURE_FORMAT}")
            
        # Clear plots to prevent overlapping/memory leak in the next loop
        for fig in figures.values():
            plt.close(fig)

    if RUN_HYDROXYLASE_SWEEP:
        run_hydroxylase_sweep()

    # MONTE CARLO: reruns every condition N_RUNS times with new parameter draws; saves one boxplot (fig_5_monte_carlo.png)
    if RUN_MONTE_CARLO:
        run_monte_carlo()