"""TPT -> calcitriol -> calcium model.      Run with:  python main.py
(keep main.py, tpt.py, calcitriol.py and calcium.py in the same folder)

WHAT IS MODELLED  (three linked compartment models, solved one after another)
  1. TPT (teriparatide) pharmacokinetics   injection site, blood, kidney, liver, bone
  2. Calcitriol                            kidney, blood, liver, gut        <- driven by TPT in kidney
  3. Calcium                               intake, gut, blood, kidney, bone <- driven by TPT (kidney, bone) and calcitriol (blood)

UNITS   TPT: mmol | calcitriol and calcium: mg | time: min | volume: L

HOW THE RESULTS ARE READ
  Every model is solved twice: TREATED (TPT injection) and CONTROL (no injection). Calcitriol and kidney calcium
  start at steady state, so they stay flat in the control run. Calcium still drifts in the control run because
  the one-off food pool (m_Ca_intake) is absorbed during the day and bone exchange is small. The TPT effect is
  therefore always TREATED minus CONTROL.

FLAGS FOR YOU: search this file for "ASSUMPTION" and "SWITCH"; they mark every interpretation I had to make.
"""

import importlib
import math

import numpy as np

np.random.seed(0)  # must come before the parameter imports so the sampled draws repeat

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import tpt
import calcitriol
import calcium




osteoporosis = False
renal_failure =False
low_estrogen = False


def apply_conditions():
    """Apply the disease conditions, then recompute everything the parameter modules derived from the changed values
    (they compute rates and starting amounts once, at import, so overriding a value alone has no effect).
    Call once after importing, and again after every importlib.reload (a reload resets the modules)."""
    if osteoporosis is True:
        calcium.calcitriol_sensitivity = np.random.uniform(0.2, 0.6)
        calcium.intial_bone_Ca = (1.0e6) * (0.844)    # mg; 1 kg = 1,000,000 mg; bone mass reduced by 15.6% in osteoporosis
        tpt.k_elimination_TPT = tpt.k_elimination_TPT / 2
        print("Osteoporosis has no assumed calcitriol parameter changes.")

    if renal_failure is True:
        if calcitriol.basal_calcitriol_produce_kidney is None or calcitriol.calcitriol_half_life is None:
            raise ValueError("Fill in healthy calcitriol production and half-life first.")
        tpt.k_diffuse_TPT = tpt.k_diffuse_TPT * 1.73
        tpt.k_elimination_TPT = tpt.k_elimination_TPT * 1.77
        calcitriol.basal_calcitriol_produce_kidney *= 14.2 / 31.8
        calcitriol.calcitriol_half_life /= 0.45 / 0.58

    if low_estrogen is True:
        print("Low estrogen has no assumed calcitriol parameter changes.")

    # --- recompute derived TPT rates (same formulas as tpt.py) ---
    tpt.k_blood_to_kidney_TPT = tpt.frac_elim_kidney_TPT * tpt.k_elimination_TPT
    tpt.k_kidney_degrade_TPT = tpt.k_blood_to_kidney_TPT
    tpt.k_bloodtoliver_TPT = tpt.frac_elim_toliver_TPT * tpt.k_elimination_TPT
    tpt.k_blood_to_other_tissue = (
        (1 - tpt.frac_elim_kidney_TPT - tpt.frac_elim_toliver_TPT) * tpt.k_elimination_TPT - tpt.TPT_to_bone_receptor)

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


apply_conditions()

# =====================================================================
# 0. SETTINGS  (the only things you should need to change here)
# =====================================================================
SIM_MINUTES = 24 * 60              # simulated time
# SCENARIO: dose relative to the standard 20 ug injection (1.0 = reference dose, 1.35 = gray-market overdose)
DOSE_SCALE = 1.0

# MONTE CARLO (item 15): re-draw every uncertain parameter N_RUNS times and run each scenario with the same draws.
RUN_MONTE_CARLO = False
N_RUNS = 100
SCENARIOS = {                                   # dose relative to the 20 ug reference
    "reference (20 ug)": 1.0,
    "gray-market overdose (+35%)": 1.35,        # problem statement: online semaglutide vials held 29-39% extra
    "degraded product (-30%)": 0.70,            # ASSUMPTION: example of peptide lost to degradation
}
STEP_MIN = 1                       # output time step

STEP_MIN = 1                       # output time step

# SWITCH: how the Hill half-saturation constant is read.
#   "fraction_of_peak": H = 0.316 x the peak TPT amount in that organ after the injection (ASSUMPTION, see below)
#   "table":            H = calcium.half_saturation_concen = 0.316 pmol, exactly as in your file
# With the table value the Hill terms are saturated at all times (baseline kidney TPT is ~30 pmol), so the
# injection changes nothing. Run once with "table" to see this for yourself.
HALF_SATURATION_MODE = "fraction_of_peak"
HALF_SATURATION_FRACTION = 0.316   # your original code used 0.316 * TPT, i.e. a fraction

# SWITCH: half-saturation constant for the TPT -> kidney Ca REABSORPTION Hill term only (calcitriol production
# and bone release keep the constants above). Set ONE of these, or leave both None to reuse the kidney value above.
#   KIDNEY_REABS_H_MMOL:     absolute value in mmol of TPT in kidney (e.g. 5e-9 = 5 pmol)
#   KIDNEY_REABS_H_FRACTION: fraction of the peak kidney TPT amount after the injection (e.g. 0.5)
KIDNEY_REABS_H_MMOL = None
KIDNEY_REABS_H_FRACTION = None

# SWITCH: calibrate the TPT -> calcitriol production coefficient so one dose raises blood calcitriol ~80%.
CALIBRATE_CALCITRIOL_TPT = True
CALCITRIOL_RISE_TARGET = 1.8       # peak (treated / control) blood calcitriol within 24 h

# SWITCH: your typed equations have no blood -> gut calcitriol term, so it is left out by default.
INCLUDE_BLOOD_TO_GUT_CALCITRIOL = False

SAVE_FIGURES = True
LABEL_CALCIUM_PEAK_RISE_MG_PER_DL = 0.4   # FORTEO label: ~+0.4 mg/dL serum Ca, shown on the plot for reference

# SWITCH (item 6/13): scale the three TPT -> calcium effects (bone release, kidney reabsorption, calcitriol -> gut)
# by ONE factor so the REFERENCE dose raises serum Ca by +0.4 mg/dL (Satterwhite 2010: median +0.4 mg/dL at
# ~4.25 h, back to baseline by 16-24 h). Found once at the reference dose and held fixed for every scenario.
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
    injection, blood, kidney, liver, bone = amounts

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
    return [d_injection, d_blood, d_kidney, d_liver, d_bone]


def solve_TPT(time_min, injection_scale):
    """injection_scale = 1: injected run;  0: control (endogenous PTH only)."""
    initial = list(tpt.initial_TPT)
    initial[0] *= injection_scale
    return integrate(TPT_model, initial, time_min, atol=1e-18)


# =====================================================================
# 3. MODEL 2 - CALCITRIOL   (amounts in mg)
# =====================================================================
def calcitriol_model(t, amounts, TPT_kidney_at, max_extra_production, H_kidney):
    kidney, blood, liver, gut = amounts
    k_degrade_kidney = calcitriol.k_degrade_calcitriol   # from the whole-body half-life (calcitriol.py)
    k_blood_to_gut = calcitriol.k_blood_to_gut_calcitriol if INCLUDE_BLOOD_TO_GUT_CALCITRIOL else 0.0

    # TPT in kidney stimulates calcitriol production with a saturating (Hill) response.
    # ASSUMPTION: max extra production = k_increase * H, so k_increase is the slope at low TPT (n = 1).
    TPT_stimulated_production = max_extra_production * hill(TPT_kidney_at(t), H_kidney, calcium.n)

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
    return [d_kidney, d_blood, d_liver, d_gut]


def solve_calcitriol(time_min, TPT_solution, k_increase, H_kidney):
    return integrate(calcitriol_model, calcitriol.initial_calcitriol, time_min, atol=1e-14,
                     args=(at_time(TPT_solution, 2), k_increase * H_kidney, H_kidney))


# =====================================================================
# 4. MODEL 3 - CALCIUM   (amounts in mg)
# =====================================================================
def calcium_rates(TPT_kidney, TPT_bone, calcitriol_blood_mg, H):
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

    return dict(gut_abs=k_gut_blood_abs, gut_waste=k_gut_waste, kidney_reabs=k_kidney_blood_reabs,
                urine=k_kidney_urine_loss, bone_release=k_bone_to_blood_Ca_with_TPT,
                calcitriol_impact=calcitriol_impact_gut_absorb, hill_kidney=hill_kidney, hill_bone=hill_bone)


def calcium_model(t, amounts, TPT_kidney_at, TPT_bone_at, calcitriol_blood_at, H):
    intake, gut, blood, kidney, bone = amounts
    k = calcium_rates(TPT_kidney_at(t), TPT_bone_at(t), calcitriol_blood_at(t), H)

    # Intake (food) -> gut.
    d_intake = calcium.Ca_intake_rate - calcium.k_speed_to_gut_transfer * intake
    # Gut: + from intake, - absorbed into blood, - lost as waste.
    d_gut = calcium.k_speed_to_gut_transfer * intake - (k["gut_abs"] + k["gut_waste"]) * gut
    # Blood: + from gut, kidney (reabsorbed) and bone (release); - to kidney and to bone (formation).
    # Bone formation flux = k_blood_to_bone_Ca * blood Ca, i.e. more calcium around bone -> more formation.
    d_blood = (
        k["gut_abs"] * gut + k["kidney_reabs"] * kidney + k["bone_release"] * bone
        - (calcium.k_blood_kidney_Ca + calcium.k_blood_to_bone_Ca) * blood
        - calcium.k_Ca_feedback * (blood - calcium.initial_blood_Ca)   # homeostasis: extra urine loss of EXCESS Ca
    )
    # Kidney: + from blood, - reabsorbed to blood, - lost in urine.
    d_kidney = calcium.k_blood_kidney_Ca * blood - (k["kidney_reabs"] + k["urine"]) * kidney
    # Bone: + formation from blood, - release to blood.
    d_bone = calcium.k_blood_to_bone_Ca * blood - k["bone_release"] * bone
    return [d_intake, d_gut, d_blood, d_kidney, d_bone]


def solve_calcium(time_min, TPT_solution, calcitriol_solution, H):
    return integrate(calcium_model, calcium.initial_calcium, time_min, atol=1e-6,
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
        # FIX: calcitriol.py leaves this value as None, which would crash the next solve with a confusing TypeError.
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


def run_simulation(fixed=None):
    """fixed = None: calibrate H, k_increase and CA_EFFECT_SCALE at the reference dose.
    fixed = dict(H=..., k_increase=..., ca_scale=...): reuse those values instead (Monte Carlo: calibrated once,
    on the default draw, then held fixed so the parameter uncertainty shows up in the results)."""
    global CA_EFFECT_SCALE
    time_min = np.arange(0, SIM_MINUTES + STEP_MIN, STEP_MIN)

    # --- Model 1: TPT, treated and control -------------------------------------------------
    TPT_reference = solve_TPT(time_min, 1.0)          # standard 20 ug dose; ONLY used to set H and k_increase
    TPT_treated = solve_TPT(time_min, DOSE_SCALE)     # the dose for this scenario
    TPT_control = solve_TPT(time_min, 0.0)            # no injection
    H = half_saturation(TPT_reference) if fixed is None else fixed["H"]   # set at the reference dose, then held fixed

    # --- Model 2: calcitriol, treated and control ------------------------------------------
    k_increase, calibrated = calcitriol.k_increase_calcitriol_produce_by_TPT, False
    if k_increase is None and not CALIBRATE_CALCITRIOL_TPT:
        raise ValueError("CALIBRATE_CALCITRIOL_TPT is False but calcitriol.k_increase_calcitriol_produce_by_TPT is None.")
    if fixed is not None:
        k_increase, calibrated = fixed["k_increase"], True
    elif CALIBRATE_CALCITRIOL_TPT:   # calibrated at the REFERENCE dose (item 11), not the scenario dose
        k_increase, calibrated = calibrate_calcitriol_coefficient(time_min, TPT_reference, TPT_control, H)
    calcitriol_treated = solve_calcitriol(time_min, TPT_treated, k_increase, H["kidney"])
    calcitriol_control = solve_calcitriol(time_min, TPT_control, k_increase, H["kidney"])

    # --- Model 3: calcium, treated and control ---------------------------------------------
    if fixed is not None:
        CA_EFFECT_SCALE = fixed["ca_scale"]
    elif CALIBRATE_CA_EFFECT:        # calibrated at the REFERENCE dose (item 6/13), not the scenario dose
        calcitriol_reference = solve_calcitriol(time_min, TPT_reference, k_increase, H["kidney"])
        calibrate_calcium_effect(time_min, TPT_reference, TPT_control, calcitriol_reference, calcitriol_control, H)
    calcium_treated = solve_calcium(time_min, TPT_treated, calcitriol_treated, H)
    calcium_control = solve_calcium(time_min, TPT_control, calcitriol_control, H)

    return dict(time=time_min, H=H, k_increase=k_increase, calibrated=calibrated, ca_scale=CA_EFFECT_SCALE,
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
    out["TPT_time_to_peak_min"] = t[np.argmax(out["TPT_blood_concen_mmol_per_L"])]
    out["TPT_blood_AUC_mmol_min_per_L"] = TRAPEZOID(out["TPT_blood_concen_mmol_per_L"], t)
    out["total_TPT_mmol"] = T.y.sum(axis=0)
    out["TPT_peak_kidney_mmol"] = T.y[2].max()
    out["TPT_time_to_kidney_peak_min"] = t[np.argmax(T.y[2])]
    out["TPT_kidney_ng"] = T.y[2] * tpt.mol_weight_TPT * 1e6   # mmol * mg/mmol = mg; 1 mg = 1e6 ng

    # Calcitriol and calcium concentrations, treated vs control
    out["calcitriol_blood_pg_per_mL"] = C.y[1] / tpt.Vol_blood_in_bone * 1e6
    out["calcitriol_blood_control_pg_per_mL"] = Cc.y[1] / tpt.Vol_blood_in_bone * 1e6
    out["Ca_blood_concen_mg_per_L"] = A.y[2] / tpt.Vol_blood_in_bone
    out["Ca_blood_control_mg_per_L"] = Ac.y[2] / tpt.Vol_blood_in_bone
    out["Ca_blood_TPT_effect_mg_per_dL"] = (out["Ca_blood_concen_mg_per_L"] - out["Ca_blood_control_mg_per_L"]) / 10
    out["bone_Ca_levels_TPT_effect"] = A.y[4] - Ac.y[4]   # treated minus control (was the treated amount only)

    # Rates over time (for plots)
    out["rates"] = calcium_rates(T.y[2], T.y[4], C.y[1], H)
    out["rates_control"] = calcium_rates(Tc.y[2], Tc.y[4], Cc.y[1], H)

    # ---- mass-balance checks: change in total amount must equal inputs minus losses -------------------
    def balance_error(total, net_flux):
        integral = np.concatenate(([0.0], np.cumsum(0.5 * (net_flux[1:] + net_flux[:-1]) * STEP_MIN)))
        return np.max(np.abs((total - total[0]) - integral))

    r = out["rates"]
    losses_TPT = (tpt.k_kidney_degrade_TPT * T.y[2] + tpt.k_bloodtoliver_TPT * T.y[3]
                  + tpt.k_blood_to_other_tissue * T.y[1] + tpt.TPT_to_bone_receptor * T.y[4])
    out["check_TPT_mass_error_mmol"] = balance_error(T.y.sum(axis=0), tpt.PTH_endogenous_blood_mmol_per_min - losses_TPT)
    k_deg = calcitriol.k_degrade_calcitriol
    max_extra = sim["k_increase"] * H["kidney"]
    production = calcitriol.basal_calcitriol_produce_kidney + max_extra * r["hill_kidney"]
    out["check_calcitriol_mass_error_mg"] = balance_error(C.y.sum(axis=0), production - k_deg * C.y[0])
    out["check_calcium_mass_error_mg"] = balance_error(A.y.sum(axis=0), calcium.Ca_intake_rate - (r["gut_waste"] * A.y[1] + r["urine"] * A.y[3])
                                                   - calcium.k_Ca_feedback * (A.y[2] - calcium.initial_blood_Ca))
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
    print(f"  peak treated/control = {np.max(out['calcitriol_blood_pg_per_mL']/out['calcitriol_blood_control_pg_per_mL']):.2f} (target {CALCITRIOL_RISE_TARGET})")
    print(f"  CONTROL drift: blood calcitriol {out['calcitriol_blood_control_pg_per_mL'][0]:.1f} -> {out['calcitriol_blood_control_pg_per_mL'][-1]:.1f} pg/mL over 24 h (basal level {calcium.calcitriol_basal_level*1e6:.1f})")
    ca_c = out["Ca_blood_control_mg_per_L"]
    print(f"Calcium: CONTROL drift: blood Ca {ca_c[0]/10:.2f} -> {ca_c[-1]/10:.2f} mg/dL over 24 h")
    effect = out["Ca_blood_TPT_effect_mg_per_dL"]
    print(f"  TPT effect on blood Ca (treated - control): peak {effect.max():+.3f} mg/dL at {t[np.argmax(effect)]/60:.1f} h (label ~ +{LABEL_CALCIUM_PEAK_RISE_MG_PER_DL})")
    print(f"  TPT effect on bone Ca after 24 h: {out['bone_Ca_levels_TPT_effect'][-1]:+.1f} mg")
    k_abs_basal = calcium.basal_absorb_Ca * calcium.gut_absorption_turnover_rate       # rate at calcitriol effect = 1
    absorbed_share = k_abs_basal / (k_abs_basal + base["gut_waste"])
    print(f"  Gut: absorbed share at basal calcitriol = {absorbed_share:.2f} (basal_absorb_Ca = {calcium.basal_absorb_Ca:.2f})")
    print(f"  Bone exchange flux = {calcium.k_blood_to_bone_Ca*calcium.initial_blood_Ca*1440/1000:.2f} g Ca/day (physiological remodelling is ~0.3-0.6 g/day)")
    print(f"  Kidney Ca starts at {calcium.initial_kidney_Ca:.2f} mg (steady state); calcitriol effect on gut absorption at baseline = {base['calcitriol_impact'][0]:.2f}")
    print(f"Mass-balance errors: TPT {out['check_TPT_mass_error_mmol']:.2e} mmol | calcitriol {out['check_calcitriol_mass_error_mg']:.2e} mg | calcium {out['check_calcium_mass_error_mg']:.2e} mg")
    print("=" * 78)


# =====================================================================
# 7. PLOTS
# =====================================================================
def make_plots(sim, out):
    t, T, Tc = sim["time"], sim["TPT"], sim["TPT_control"]
    C, A, Ac = sim["calcitriol"], sim["calcium"], sim["calcium_control"]
    h = t / 60
    figures = {}

    # ---- Figure 1: TPT pharmacokinetics ----------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for index, label in enumerate(["Injection site", "Blood", "Kidney", "Liver", "Bone"]):
        ax[0, 0].plot(t, T.y[index], label=label)
    ax[0, 0].set(title="TPT amount in each compartment", xlabel="Time (min)", ylabel="TPT amount (mmol)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, out["TPT_blood_concen_ng_per_mL"])
    peak_t = out["TPT_time_to_peak_min"]                       # time of peak blood TPT (min)
    peak_c = np.max(out["TPT_blood_concen_ng_per_mL"])         # peak blood TPT (ng/mL)
    ax[0, 1].axvline(peak_t, color="gray", ls=":", label=f"peak {peak_c:.3f} ng/mL at {peak_t:.0f} min")
    ax[0, 1].set(title="TPT blood concentration", xlabel="Time (min)", ylabel="ng/mL")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(t, out["TPT_kidney_ng"])
    ax[1, 0].set(title="TPT in kidney", xlabel="Time (min)", ylabel="ng")
    ax[1, 1].plot(t, out["rates"]["hill_kidney"], label="kidney Hill (treated)")
    ax[1, 1].plot(t, out["rates"]["hill_bone"], label="bone Hill (treated)")
    ax[1, 1].plot(t, out["rates_control"]["hill_kidney"], "--", label="kidney Hill (control)")
    ax[1, 1].plot(t, out["rates_control"]["hill_bone"], "--", label="bone Hill (control)")
    ax[1, 1].set(title="TPT effect signals (0 to 1)", xlabel="Time (min)", ylim=(-0.02, 1.05))
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("1. TPT pharmacokinetics"); fig.tight_layout(); figures["1_TPT"] = fig

    # ---- Figure 2: calcitriol ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for index, label in enumerate(["Kidney", "Blood", "Liver", "Gut"]):
        ax[0].plot(t, C.y[index], label=label)
    ax[0].set(title="Calcitriol amount (treated)", xlabel="Time (min)", ylabel="mg"); ax[0].legend(fontsize=8)
    ax[1].plot(h, out["calcitriol_blood_pg_per_mL"], label="treated")
    ax[1].plot(h, out["calcitriol_blood_control_pg_per_mL"], "--", label="control")
    ax[1].axhline(calcium.calcitriol_basal_level * 1e6, color="gray", lw=0.7, label="basal level")
    ax[1].set(title="Blood calcitriol", xlabel="Time (h)", ylabel="pg/mL"); ax[1].legend(fontsize=8)
    ax[2].plot(h, out["rates"]["calcitriol_impact"], label="treated")
    ax[2].plot(h, out["rates_control"]["calcitriol_impact"], "--", label="control")
    ax[2].set(title="Calcitriol effect on gut absorption (1 = basal)", xlabel="Time (h)", ylabel="Multiplier (k)"); ax[2].legend(fontsize=8)
    fig.suptitle("2. Calcitriol"); fig.tight_layout(); figures["2_calcitriol"] = fig

    # ---- Figure 3: calcium ------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for index, label in enumerate(["Intake", "Gut", "Blood", "Kidney"]):
        ax[0, 0].plot(t, A.y[index], label=label)
    ax[0, 0].set(title="Calcium amount, treated (bone shown separately)", xlabel="Time (min)", ylabel="mg")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(h, out["bone_Ca_levels_TPT_effect"])
    ax[0, 1].set(title="Bone calcium: treated minus control", xlabel="Time (h)", ylabel="mg")
    ax[1, 0].plot(h, out["Ca_blood_concen_mg_per_L"] / 10, label="treated")
    ax[1, 0].plot(h, out["Ca_blood_control_mg_per_L"] / 10, "--", label="control")
    ax[1, 0].set(title="Blood calcium", xlabel="Time (h)", ylabel="mg/dL"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(h, out["Ca_blood_TPT_effect_mg_per_dL"])
    ax[1, 1].axhline(LABEL_CALCIUM_PEAK_RISE_MG_PER_DL, color="gray", ls=":", label="label peak (+0.4)")
    ax[1, 1].set(title="TPT effect on blood calcium levels", xlabel="Time (h)", ylabel="mg/dL")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("3. Calcium"); fig.tight_layout(); figures["3_calcium"] = fig

    if SAVE_FIGURES:
        for name, fig in figures.items():
            fig.savefig(f"fig_{name}.png", dpi=110)
    return figures


# =====================================================================
# 8. MONTE CARLO (item 15)
# =====================================================================
def run_monte_carlo():
    """Calibrate once on the default draw (seed 0) at the reference dose, then re-draw the uncertain parameters
    N_RUNS times and run every scenario with the calibration held fixed. Prints median [5th-95th percentile]."""
    global DOSE_SCALE
    np.random.seed(0)
    for module in (tpt, calcitriol, calcium):
        importlib.reload(module)
    apply_conditions()                               # a reload resets the modules, so re-apply the conditions
    DOSE_SCALE = 1.0
    nominal = run_simulation()
    fixed = dict(H=nominal["H"], k_increase=nominal["k_increase"], ca_scale=nominal["ca_scale"])
    results = {name: {"TPT peak (ng/mL)": [], "Ca rise peak (mg/dL)": [], "Ca peak time (h)": [],
                      "calcitriol peak (x baseline)": []} for name in SCENARIOS}
    for run in range(N_RUNS):
        np.random.seed(run + 1)                      # draw number = seed - 1, so any run can be repeated
        for module in (tpt, calcitriol, calcium):    # re-import = new random draws
            importlib.reload(module)
        apply_conditions()
        try:
            for name, dose_scale in SCENARIOS.items():
                DOSE_SCALE = dose_scale
                sim = run_simulation(fixed)
                out = compute_outputs(sim)
                effect = out["Ca_blood_TPT_effect_mg_per_dL"]
                r = results[name]
                r["TPT peak (ng/mL)"].append(np.max(out["TPT_blood_concen_ng_per_mL"]))
                r["Ca rise peak (mg/dL)"].append(np.max(effect))
                r["Ca peak time (h)"].append(sim["time"][np.argmax(effect)] / 60)
                r["calcitriol peak (x baseline)"].append(np.max(out["calcitriol_blood_pg_per_mL"]
                                                                / out["calcitriol_blood_control_pg_per_mL"]))
        except (ValueError, RuntimeError) as error:  # a draw that cannot be calibrated is skipped and reported
            print(f"  draw {run} skipped: {error}")
    print("=" * 78)
    print(f"MONTE CARLO: {N_RUNS} draws, median [5th - 95th percentile]")
    for name, r in results.items():
        print(f"  {name}  (n = {len(r['Ca rise peak (mg/dL)'])})")
        for label, values in r.items():
            low, mid, high = np.percentile(values, [5, 50, 95])
            print(f"      {label:30s} {mid:7.3f}  [{low:.3f} - {high:.3f}]")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([r["Ca rise peak (mg/dL)"] for r in results.values()], whis=(5, 95))
    ax.set_xticks(range(1, len(results) + 1)); ax.set_xticklabels(list(results), fontsize=8)
    ax.axhline(LABEL_CALCIUM_PEAK_RISE_MG_PER_DL, color="gray", ls=":", label="label peak (+0.4)")
    ax.set(title=f"Peak rise in blood calcium, {N_RUNS} Monte Carlo draws", ylabel="mg/dL"); ax.legend()
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig("fig_4_monte_carlo.png", dpi=110)
    return results


# =====================================================================
# 9. RUN
# =====================================================================
if __name__ == "__main__":
    simulation = run_simulation()
    outputs = compute_outputs(simulation)
    print_summary(simulation, outputs)
    make_plots(simulation, outputs)
    if RUN_MONTE_CARLO:
        run_monte_carlo()
    plt.show()