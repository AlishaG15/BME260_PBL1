import numpy as np



import math

# PARAMETERS (min^-1)
# Midpoint of the document's listed fast-exchange value/range
k_kidney_to_blood_calcitriol = 0.2
k_blood_to_kidney_calcitriol = 0.2
k_liver_to_gut_calcitriol = None  # No independent value; lumped elimination
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


# INITIAL CONDITIONS (mg)
# Midpoints of the reported initial mass ranges, converted from pmol to mg

Initial_calcitriol_kidney = 153.5 * PMOL_TO_MG
Initial_calcitriol_blood = 273.5 * PMOL_TO_MG
initial_calcitriol_liver = 891.5 * PMOL_TO_MG
Initial_calcitriol_GUT_PLACEHOLDER = 594.5 * PMOL_TO_MG

Initial_calcitriol_gut = Initial_calcitriol_GUT_PLACEHOLDER

initial_calcitriol = [
    Initial_calcitriol_kidney,
    Initial_calcitriol_blood,
    initial_calcitriol_liver,
    Initial_calcitriol_gut,
]


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
