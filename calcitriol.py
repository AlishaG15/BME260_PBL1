"""Calcitriol parameters. Units: mg, mmol TPT, minutes, and liters.

k_increase_calcitriol_produce_by_TPT is left as None because it is calibrated in main.py.
Disease-state changes (osteoporosis, renal failure) are applied in main.py, apply_conditions().
"""

import math

# PARAMETERS (min^-1)
# Midpoint of the document's listed fast-exchange value/range
k_kidney_to_blood_calcitriol = 0.2
k_blood_to_kidney_calcitriol = 0.2
k_liver_to_gut_calcitriol = 0.0  # ASSUMED 0; no independent value, lumped into elimination
k_gut_to_blood_calcitriol = 0.2
k_blood_to_liver_calcitriol = 0.2
K_liver_to_blood_calcitriol = 0.2

# Only used if INCLUDE_BLOOD_TO_GUT_CALCITRIOL = True in main.py (off by default)
k_blood_to_gut_calcitriol = 0.2

# HALF-LIFE (min)
# Midpoint of 5-8 hours = 6.5 hours = 390 minutes
calcitriol_half_life = 390.0

# BASAL PRODUCTION (mg/min)
# Midpoint: 3.1 pmol/min
# Molecular weight = 416.64 g/mol
PMOL_TO_MG = 416.64e-9
basal_calcitriol_produce_kidney = 3.1 * PMOL_TO_MG

# TPT-dependent production coefficient
# No literature value; calibrated in main.py so one dose raises blood calcitriol 80%
k_increase_calcitriol_produce_by_TPT = None



# INITIAL CONDITIONS (mg)
# Start at steady state (no TPT), so calcitriol only changes because of the drug.
# Set each balance to 0 with no TPT and k_liver_to_gut = 0:
#   liver:  k_BL*B = k_LB*L        ->  L = (k_BL/k_LB)*B
#   blood:  k_KB*K = k_BK*B        ->  K = (k_BK/k_KB)*B
#   kidney: P + k_BK*B = (k_KB + k_d)*K  ->  B = P*k_KB/(k_d*k_BK)
# Only the kidney degrades calcitriol, but at steady state the kidney holds only part of the total
# (kidney : blood : liver = k_BK/k_KB : 1 : k_BL/k_LB), so scale the kidney rate by total/kidney.
# Then the whole body clears calcitriol with the half-life entered above.
total_over_kidney = 1 + (k_kidney_to_blood_calcitriol / k_blood_to_kidney_calcitriol) * (
    1 + k_blood_to_liver_calcitriol / K_liver_to_blood_calcitriol)  # = 3 when all exchange rates are equal
k_degrade_calcitriol = math.log(2) / calcitriol_half_life * total_over_kidney  # min^-1

Initial_calcitriol_blood = (
    basal_calcitriol_produce_kidney * k_kidney_to_blood_calcitriol
    / (k_degrade_calcitriol * k_blood_to_kidney_calcitriol)
)
Initial_calcitriol_kidney = (k_blood_to_kidney_calcitriol / k_kidney_to_blood_calcitriol) * Initial_calcitriol_blood
initial_calcitriol_liver = (k_blood_to_liver_calcitriol / K_liver_to_blood_calcitriol) * Initial_calcitriol_blood
Initial_calcitriol_gut = 0.0  # nothing flows liver -> gut, so the gut stays empty

initial_calcitriol = [
    Initial_calcitriol_kidney,
    Initial_calcitriol_blood,
    initial_calcitriol_liver,
    Initial_calcitriol_gut,
]
