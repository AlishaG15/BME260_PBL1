"""Calcium parameters. Units: mg calcium, mmol TPT, minutes, and liters.

Dynamic hormone effects are evaluated in main.py, using current amounts or
concentrations rather than undefined derivatives such as dmB and dmCatrB.
"""

import numpy as np

from tpt import Vol_blood_in_bone
import calcitriol


# PARAMETERS
cardiac_out_frac_bone = np.random.uniform(0.04, 0.10)  # Dimensionless.
calcitriol_sensitivity = np.random.uniform(0.2, 1.0)  # Original sampling retained.
calcitriol_basal_level = np.random.uniform(18, 78) * 1e-6  # mg/L.
# 18-78 pg/mL = 18e-6-78e-6 mg/L; use the same units for current calcitriol.
max_frac_renalreabs_TPT_impact = np.random.uniform(0.1, 1.0)  # Dimensionless.
ca_release_incr_TPT = np.random.uniform(1.0, 2.9)  # Original range; dimensionless.

basal_absorb_Ca = 0.3  # Dimensionless; assumed basal gut absorption fraction.
basal_kidney_reabsorb_Ca = 0.99  # Dimensionless; assumed basal reabsorbed fraction.
k_speed_to_gut_transfer = 0.05  # min^-1; assumed 20-minute intake-to-gut transfer.
blood_flow_kidney = 0.118924  # L/min; 118.924 mL/min = 0.118924 L/min.
cardiac_output = 5.0  # L/min; value from the PDF.
half_saturation_concen = 0.316e-9  # 0.316 pmol = 3.16e-10 mmol TPT in kidney.
# The table calls this a concentration, but its stated unit is an amount.
n = 1  # Dimensionless Hill coefficient; no replacement name supplied in the PDF.

# Explicit time scales are required to turn dimensionless fractions into rates.
gut_absorption_turnover_rate = 1 / 360  # min^-1; ASSUMED ~6 h gut transit
gut_waste_turnover_rate = 1 / 360  # min^-1; same as absorption so 30% absorbed stays 30%
kidney_turnover_rate = 1.0  # min^-1.
#Assumption: [equals to the original assumed kidney turnover of 1 min^-1].

# The original bone factor was 1 + E_TPT_Bo * TPT_bone (pmol), although the
# table labels E_TPT_Bo dimensionless. A reference amount is needed to make
# that expression dimensionless; its value is left blank rather than guessed.
bone_TPT_reference_mmol = None  # mmol TPT; needed only for the bone response.

k_blood_kidney_Ca = blood_flow_kidney / Vol_blood_in_bone  # min^-1.
# FIX (bone flux): the old rate cardiac_out_frac_bone * cardiac_output / V assumed bone takes up ALL the calcium
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
m_Ca_intake = np.random.uniform(1000, 1200)  # Initial food/intake pool, not mg/min.
initial_gut = 0.0
#Assumption: [equals to the initial kidney calcium amount]; both start empty.
initial_blood_Ca = 95 * Vol_blood_in_bone  # 95 mg/L * 8.4 L = 798 mg.
intial_bone_Ca = 1.0e6  # 1 kg = 1,000,000 mg; spelling follows the PDF.

k_bone_to_blood_Ca = (
    k_blood_to_bone_Ca * initial_blood_Ca / intial_bone_Ca
)  # min^-1; equal basal bone influx and outflux at the initial blood/bone amounts.

# FIX (kidney start): kidney calcium used to start at 0, so reabsorption and urine loss started at 0 and the
# control run spent its first hours filling the kidney. Start at steady state instead:
#   inflow k_blood_kidney_Ca * blood = outflow (reabsorption + urine) * kidney = kidney_turnover_rate * kidney
initial_kidney_Ca = k_blood_kidney_Ca * initial_blood_Ca / kidney_turnover_rate  # mg; about 11.3 mg.

# FIX (calcitriol baseline): calcitriol.py starts blood calcitriol at its own steady state (about 86.5 pg/mL), so
# measure the gut-absorption effect from THAT level; the sampled 18-78 pg/mL draw above is kept only so the
# random sequence (and every draw after it) stays identical, and is no longer used.
calcitriol_basal_level_sampled = calcitriol_basal_level  # mg/L; kept for reference, not used.
calcitriol_basal_level = calcitriol.Initial_calcitriol_blood / Vol_blood_in_bone  # mg/L; effect = 1 at baseline.

initial_calcium = [
    m_Ca_intake,
    initial_gut,
    initial_blood_Ca,
    initial_kidney_Ca,
    intial_bone_Ca,
]