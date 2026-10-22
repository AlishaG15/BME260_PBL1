"""Calcitriol parameters. Units: mg, mmol TPT, minutes, and liters.

None means the value has intentionally been left blank. Enter new values in
the units shown here; no missing parameter values have been estimated.
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

# Listed in the document but absent from the original ODEs
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
# Not specified numerically in the document
k_increase_calcitriol_produce_by_TPT = None


# CONDITIONS: False keeps the healthy baseline.
osteoporosis = False
renal_failure = False
low_estrogen = False

if osteoporosis:
    print("Osteoporosis has no assumed calcitriol parameter changes.")

if renal_failure:
    if basal_calcitriol_produce_kidney is None or calcitriol_half_life is None:
        raise ValueError("Fill in healthy calcitriol production and half-life first.")
    basal_calcitriol_produce_kidney *= 14.2 / 31.8
    calcitriol_half_life /= 0.45 / 0.58

if low_estrogen:
    print("Low estrogen has no assumed calcitriol parameter changes.")


# INITIAL CONDITIONS (mg)
# Start at steady state (no TPT), so calcitriol only changes because of the drug.
# Set each balance to 0 with no TPT and k_liver_to_gut = 0:
#   liver:  k_BL*B = k_LB*L        ->  L = (k_BL/k_LB)*B
#   blood:  k_KB*K = k_BK*B        ->  K = (k_BK/k_KB)*B
#   kidney: P + k_BK*B = (k_KB + k_d)*K  ->  B = P*k_KB/(k_d*k_BK)
# Old reported values (pmol): kidney 153.5, blood 273.5, liver 891.5, gut 594.5
k_degrade_calcitriol = math.log(2) / calcitriol_half_life  # min^-1

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
