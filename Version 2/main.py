"""Run with: python main.py (keep all four scripts in the same folder).

Units: TPT in mmol; calcitriol and calcium in mg; time in minutes; volume in L.
Fill all required None values in calcitriol.py and calcium.py before running.
The PDF's updated names are used wherever supplied; other names describe
the compound, compartment, and units. Unit corrections are documented locally.
"""

import numpy as np

np.random.seed(0)  # Set before parameter imports, preserving repeatable draws.

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import tpt
import calcium
import calcitriol


def TPT_model(time_min, TPT_amounts_mmol):
    TPT_injection_mmol, TPT_blood_mmol, TPT_kidney_mmol, TPT_liver_mmol, TPT_bone_mmol = TPT_amounts_mmol

    dTPT_injection = -tpt.k_diffuse_TPT * TPT_injection_mmol
    dTPT_blood = (
        tpt.k_diffuse_TPT * TPT_injection_mmol
        + tpt.PTH_endogenous_blood_mmol_per_min
        + tpt.k_kidney_to_blood_TPT * TPT_kidney_mmol
        + tpt.k_bone_to_blood_TPT * TPT_bone_mmol
        - (tpt.k_blood_to_kidney_TPT + tpt.k_bloodtoliver_TPT
           + tpt.k_blood_to_bone_TPT + tpt.k_blood_to_other_tissue) * TPT_blood_mmol
    )
    dTPT_kidney = (
        tpt.k_blood_to_kidney_TPT * TPT_blood_mmol
        - (tpt.k_kidney_to_blood_TPT + tpt.k_blood_to_kidney_TPT) * TPT_kidney_mmol
    )  # Kidney degradation uses k_blood_to_kidney_TPT under the equal-rate assumption.
    dTPT_liver = tpt.k_bloodtoliver_TPT * (TPT_blood_mmol - TPT_liver_mmol)
    # Liver degradation uses k_bloodtoliver_TPT under the equal-rate assumption.
    dTPT_bone = (
        tpt.k_blood_to_bone_TPT * TPT_blood_mmol
        - (tpt.k_bone_to_blood_TPT + tpt.TPT_to_bone_receptor) * TPT_bone_mmol
    )
    return [dTPT_injection, dTPT_blood, dTPT_kidney, dTPT_liver, dTPT_bone]


def calcitriol_model(time_min, calcitriol_amounts_mg, TPT_solution):
    calcitriol_kidney_mg, calcitriol_blood_mg, calcitriol_liver_mg, calcitriol_gut_mg = calcitriol_amounts_mg
    TPT_kidney_mmol = np.interp(time_min, TPT_solution.t, TPT_solution.y[2])
    k_degrade_kidney_calcitriol = np.log(2) / calcitriol.calcitriol_half_life

    dcalcitriol_kidney = (
        calcitriol.basal_calcitriol_produce_kidney
        + calcitriol.k_increase_calcitriol_produce_by_TPT * TPT_kidney_mmol
        + calcitriol.k_blood_to_kidney_calcitriol * calcitriol_blood_mg
        - (calcitriol.k_kidney_to_blood_calcitriol + k_degrade_kidney_calcitriol) * calcitriol_kidney_mg
    )
    dcalcitriol_blood = (
        calcitriol.k_kidney_to_blood_calcitriol * calcitriol_kidney_mg
        + calcitriol.K_liver_to_blood_calcitriol * calcitriol_liver_mg
        + calcitriol.k_gut_to_blood_calcitriol * calcitriol_gut_mg
        - (calcitriol.k_blood_to_kidney_calcitriol + calcitriol.k_blood_to_liver_calcitriol) * calcitriol_blood_mg
    )
    dcalcitriol_liver = (
        calcitriol.k_blood_to_liver_calcitriol * calcitriol_blood_mg
        - (calcitriol.K_liver_to_blood_calcitriol + calcitriol.k_liver_to_gut_calcitriol) * calcitriol_liver_mg
    )
    dcalcitriol_gut = (
        calcitriol.k_liver_to_gut_calcitriol * calcitriol_liver_mg
        - calcitriol.k_gut_to_blood_calcitriol * calcitriol_gut_mg
    )
    return [dcalcitriol_kidney, dcalcitriol_blood, dcalcitriol_liver, dcalcitriol_gut]


def calcium_model(time_min, calcium_amounts_mg, TPT_solution, calcitriol_solution):
    Ca_intake_mg, Ca_gut_mg, Ca_blood_mg, Ca_kidney_mg, Ca_bone_mg = calcium_amounts_mg
    calcitriol_concen = (
        np.interp(time_min, calcitriol_solution.t, calcitriol_solution.y[1])
        / tpt.Vol_blood_in_bone
    )  # Blood calcitriol in mg/L, matching calcitriol_basal_level.
    TPT_kidney_mmol = np.interp(time_min, TPT_solution.t, TPT_solution.y[2])
    TPT_bone_mmol = np.interp(time_min, TPT_solution.t, TPT_solution.y[4])

    calcitriol_impact_gut_absorb = 1 + calcium.calcitriol_sensitivity * (
        calcitriol_concen / calcium.calcitriol_basal_level - 1
    )
    k_gut_blood_abs = (
        calcium.basal_absorb_Ca * calcitriol_impact_gut_absorb
        * calcium.gut_absorption_turnover_rate
    )  # min^-1; apply the calcitriol effect once, using the PDF's blood-level formula.
    k_gut_waste = (1 - calcium.basal_absorb_Ca) * calcium.gut_waste_turnover_rate
    k_kidney_blood_reabs = (
        calcium.basal_kidney_reabsorb_Ca * calcium.kidney_turnover_rate
        * (1 + calcium.max_frac_renalreabs_TPT_impact
           * TPT_kidney_mmol ** calcium.n
           / (TPT_kidney_mmol ** calcium.n + calcium.half_saturation_concen ** calcium.n))
    )  # min^-1; the PDF's Hill expression, applied once using the kidney amount.
    k_kidney_urine_loss = calcium.kidney_turnover_rate - k_kidney_blood_reabs
    # Unresolved model issue: this table formula can give negative urine loss
    # if TPT raises reabsorption above total turnover. No clipping is imposed.
    k_bone_to_blood_Ca_with_TPT = calcium.k_bone_to_blood_Ca * (
        1 + calcium.ca_release_incr_TPT * TPT_bone_mmol / calcium.bone_TPT_reference_mmol
    )  # min^-1; the blank reference amount makes the TPT factor dimensionless.

    dCa_intake = -calcium.k_speed_to_gut_transfer * Ca_intake_mg
    dCa_gut = (
        calcium.k_speed_to_gut_transfer * Ca_intake_mg
        - (k_gut_blood_abs + k_gut_waste) * Ca_gut_mg
    )
    dCa_blood = (
        k_gut_blood_abs * Ca_gut_mg + k_kidney_blood_reabs * Ca_kidney_mg
        + k_bone_to_blood_Ca_with_TPT * Ca_bone_mg
        - (calcium.k_blood_kidney_Ca + calcium.k_blood_to_bone_Ca) * Ca_blood_mg
    )
    dCa_kidney = (
        calcium.k_blood_kidney_Ca * Ca_blood_mg
        - (k_kidney_blood_reabs + k_kidney_urine_loss) * Ca_kidney_mg
    )
    dCa_bone = calcium.k_blood_to_bone_Ca * Ca_blood_mg - k_bone_to_blood_Ca_with_TPT * Ca_bone_mg
    return [dCa_intake, dCa_gut, dCa_blood, dCa_kidney, dCa_bone]


def check_required_values():
    """Report intentional blanks together, before attempting any integration."""
    missing = [
        "calcitriol." + name for name, value in vars(calcitriol).items()
        if value is None and name != "k_blood_to_gut_calcitriol"
    ]
    missing.extend("calcium." + name for name, value in vars(calcium).items() if value is None)
    if missing:
        raise ValueError("Fill in these blank values in the indicated units:\n  " + "\n  ".join(missing))
    if calcitriol.calcitriol_half_life <= 0 or calcium.bone_TPT_reference_mmol <= 0:
        raise ValueError("Calcitriol half-life and bone TPT reference amount must be positive.")


def run_simulation():
    check_required_values()
    time_start_min = 0
    time_end_min = 24 * 60  # 24 h = 1440 min.
    time_step_min = 1
    time_min = np.arange(time_start_min, time_end_min + time_step_min, time_step_min)

    TPT_solution = solve_ivp(
        TPT_model, (time_start_min, time_end_min), tpt.initial_TPT,
        t_eval=time_min, rtol=1e-6, atol=1e-15,
    )  # Tight mmol tolerance avoids treating the small TPT amounts as numerical zero.
    if not TPT_solution.success:
        raise RuntimeError(TPT_solution.message)
    calcitriol_solution = solve_ivp(
        calcitriol_model, (time_start_min, time_end_min), calcitriol.initial_calcitriol,
        args=(TPT_solution,), t_eval=time_min, rtol=1e-6, atol=1e-12,
    )
    if not calcitriol_solution.success:
        raise RuntimeError(calcitriol_solution.message)
    calcium_solution = solve_ivp(
        calcium_model, (time_start_min, time_end_min), calcium.initial_calcium,
        args=(TPT_solution, calcitriol_solution), t_eval=time_min, rtol=1e-6, atol=1e-8,
    )
    if not calcium_solution.success:
        raise RuntimeError(calcium_solution.message)

    # OUTPUTS: retain primary units and label optional ng conversions explicitly.
    TPT_blood_concen = TPT_solution.y[1] / tpt.Vol_blood_in_bone  # mmol/L.
    TPT_blood_concen_ng_per_mL = TPT_blood_concen * tpt.mol_weight_TPT * 1000
    # mmol/L * mg/mmol = mg/L; 1 mg/L = 1000 ng/mL.
    TPT_peak_blood_concen = np.max(TPT_blood_concen)  # mmol/L.
    TPT_time_to_peak_min = time_min[np.argmax(TPT_blood_concen)]
    TPT_blood_AUC = np.trapezoid(TPT_blood_concen, time_min)  # mmol*min/L.
    total_TPT_mmol = np.sum(TPT_solution.y, axis=0)
    Ca_blood_concen = calcium_solution.y[2] / tpt.Vol_blood_in_bone  # mg/L.
    TPT_peak_kidney_mmol = np.max(TPT_solution.y[2])
    TPT_time_to_kidney_peak_min = time_min[np.argmax(TPT_solution.y[2])]
    TPT_kidney_ng = TPT_solution.y[2] * tpt.mol_weight_TPT * 1e6  # 1 mg = 1e6 ng.

    # PLOTS
    plt.figure()
    for index, label in enumerate(["Injection site", "Blood", "Kidney", "Liver", "Bone"]):
        plt.plot(time_min, TPT_solution.y[index], label=label)
    plt.xlabel("Time (min)")
    plt.ylabel("TPT amount (mmol)")
    plt.legend()

    plt.figure()
    plt.plot(time_min, TPT_solution.y[2])
    plt.xlabel("Time (min)")
    plt.ylabel("TPT amount in kidney (mmol)")

    plt.figure()
    plt.plot(time_min, TPT_blood_concen)
    plt.xlabel("Time (min)")
    plt.ylabel("TPT blood concentration (mmol/L)")

    plt.figure()
    for index, label in enumerate(["Intake", "Gut", "Blood", "Kidney", "Bone"]):
        plt.plot(time_min, calcium_solution.y[index], label=label)
    plt.xlabel("Time (min)")
    plt.ylabel("Calcium mass (mg)")
    plt.legend()
    plt.show()

    return {
        "TPT_solution": TPT_solution,
        "calcitriol_solution": calcitriol_solution,
        "calcium_solution": calcium_solution,
        "TPT_blood_concen_mmol_per_L": TPT_blood_concen,
        "TPT_blood_concen_ng_per_mL": TPT_blood_concen_ng_per_mL,
        "TPT_peak_blood_concen_mmol_per_L": TPT_peak_blood_concen,
        "TPT_time_to_peak_min": TPT_time_to_peak_min,
        "TPT_blood_AUC_mmol_min_per_L": TPT_blood_AUC,
        "total_TPT_mmol": total_TPT_mmol,
        "Ca_blood_concen_mg_per_L": Ca_blood_concen,
        "TPT_peak_kidney_mmol": TPT_peak_kidney_mmol,
        "TPT_time_to_kidney_peak_min": TPT_time_to_kidney_peak_min,
        "TPT_kidney_ng": TPT_kidney_ng,
    }


if __name__ == "__main__":
    results = run_simulation()
