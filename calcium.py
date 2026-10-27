"""Calcium parameters. Units: mg calcium, mmol TPT, minutes, and liters.

Dynamic hormone effects are evaluated in main.py, using current amounts or
concentrations rather than undefined derivatives such as dmB and dmCatrB.
"""

import numpy as np

from tpt import Vol_blood_in_bone
import calcitriol


# PARAMETERS
cardiac_out_frac_bone = np.random.uniform(0.04, 0.10)  # Dimensionless.
calcitriol_sensitivity = np.random.uniform(0.2, 1.0)  # Dimensionless.
calcitriol_basal_level = np.random.uniform(18, 78) * 1e-6  # mg/L (18-78 pg/mL); replaced by the steady-state value below.
max_frac_renalreabs_TPT_impact = np.random.uniform(0.1, 1.0)  # Dimensionless.
ca_release_incr_TPT = np.random.uniform(1.0, 2.9)  # Dimensionless.

basal_absorb_Ca = 0.3  # Dimensionless; assumed basal gut absorption fraction.
basal_kidney_reabsorb_Ca = 0.99  # Dimensionless; starting guess, RECALCULATED below so Ca in = Ca out (~0.96).
k_speed_to_gut_transfer = 0.05  # min^-1; assumed 20-minute intake-to-gut transfer.
blood_flow_kidney = 0.118924  # L/min; glomerular filtration rate (GFR), not total renal blood flow
cardiac_output = 5.0  # L/min.
half_saturation_concen = 0.316e-9  # 0.316 pmol = 3.16e-10 mmol TPT in kidney.
# Only used when HALF_SATURATION_MODE = "table" in main.py.
n = 1  # Dimensionless Hill coefficient.

# Explicit time scales are required to turn dimensionless fractions into rates.
gut_absorption_turnover_rate = 1 / 360  # min^-1; ASSUMED ~6 h gut transit
gut_waste_turnover_rate = 1 / 360  # min^-1; same as absorption so 30% absorbed stays 30%
kidney_turnover_rate = 1.0  # min^-1; ASSUMED.

frac_filterable_Ca = 0.575  # filterable (ionized + complexed) fraction of blood Ca; the rest is protein-bound
k_blood_kidney_Ca = frac_filterable_Ca * blood_flow_kidney / Vol_blood_in_bone  # min^-1
# Bone flux: perfusion alone (cardiac_out_frac_bone * cardiac_output / V) would mean bone takes up ALL the calcium
# delivered to it (about 54 g Ca/day). Physiological bone formation/resorption is about 0.3-0.6 g/day, so the
# delivered calcium is multiplied by an extraction fraction (the share of delivered calcium that is taken up).
bone_Ca_flux_mg_per_day = 450.0  # mg/day; physiological bone exchange, mid-range of 300-600 mg/day.
Ca_blood_mg_per_L_baseline = 95.0  # mg/L; same blood calcium used for initial_blood_Ca below.
bone_perfusion_rate = cardiac_out_frac_bone * cardiac_output / Vol_blood_in_bone  # min^-1; perfusion-limited ceiling.
bone_Ca_extraction = (
    bone_Ca_flux_mg_per_day / 1440
    / (bone_perfusion_rate * Ca_blood_mg_per_L_baseline * Vol_blood_in_bone)
)  # dimensionless; about 0.008, i.e. under 1% of calcium delivered to bone is taken up.
k_blood_to_bone_Ca = bone_perfusion_rate * bone_Ca_extraction  # min^-1


# INITIAL CONDITIONS (mg)
Ca_intake_mg_per_day = np.random.uniform(1000, 1200)  # mg/day dietary calcium (same random draw as before)
Ca_intake_rate = Ca_intake_mg_per_day / 1440  # mg/min; food calcium spread evenly over the day
initial_gut = 0.0  # replaced below by the steady-state gut amount
initial_blood_Ca = 95 * Vol_blood_in_bone  # 95 mg/L * 8.4 L = 798 mg.
intial_bone_Ca = 1.0e6  # 1 kg = 1,000,000 mg.

k_bone_to_blood_Ca = (
    k_blood_to_bone_Ca * initial_blood_Ca / intial_bone_Ca
)  # min^-1; equal basal bone influx and outflux at the initial blood/bone amounts.

# Kidney calcium starts at steady state:
#   inflow k_blood_kidney_Ca * blood = outflow (reabsorption + urine) * kidney = kidney_turnover_rate * kidney
initial_kidney_Ca = k_blood_kidney_Ca * initial_blood_Ca / kidney_turnover_rate  # mg; about 6.5 mg.

# Calcitriol baseline: calcitriol.py starts blood calcitriol at its own steady state (28.8 pg/mL), so the
# gut-absorption effect is measured from THAT level. The 18-78 pg/mL draw above is kept only so the random
# sequence (and every draw after it) stays the same; its value is not used.
calcitriol_basal_level = calcitriol.Initial_calcitriol_blood / Vol_blood_in_bone  # mg/L; effect = 1 at baseline.

# HEALTHY STEADY STATE: intake, gut and kidney start at steady state, and the kidney is sized so that
# calcium absorbed from the gut = calcium lost in urine (bone in = bone out is already set above).
initial_intake_Ca = Ca_intake_rate / k_speed_to_gut_transfer  # mg; intake in = intake out
initial_gut = Ca_intake_rate / (basal_absorb_Ca * gut_absorption_turnover_rate + (1 - basal_absorb_Ca) * gut_waste_turnover_rate)  # mg; gut in = gut out
filtered_Ca_load = k_blood_kidney_Ca * initial_blood_Ca  # mg/min filtered by the kidney
basal_kidney_reabsorb_Ca = 1 - basal_absorb_Ca * Ca_intake_rate / filtered_Ca_load  # about 0.965; urine = absorbed

initial_calcium = [
    initial_intake_Ca,
    initial_gut,
    initial_blood_Ca,
    initial_kidney_Ca,
    intial_bone_Ca,
]

# HOMEOSTASIS: when blood Ca rises, the body lowers its own PTH and clears the extra Ca (mostly in urine).
# Satterwhite 2010 fit serum Ca with an indirect-response model whose return rate is k_out = 0.12 h^-1
# (half-life ~5.8 h). Without feedback our model returns ~3x slower, so add the missing part:
k_out_Ca_Satterwhite = 0.12 / 60  # min^-1
k_Ca_return_no_feedback = (k_blood_kidney_Ca * (1 - basal_kidney_reabsorb_Ca) + k_blood_to_bone_Ca)  # min^-1; how fast extra blood Ca leaves without feedback
k_Ca_feedback = max(k_out_Ca_Satterwhite - k_Ca_return_no_feedback, 0.0)  # min^-1; extra urine loss of EXCESS Ca
