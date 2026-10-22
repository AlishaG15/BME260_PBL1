"""Calcitriol parameters. Units: mg, mmol TPT, minutes, and liters.

None means the value has intentionally been left blank. Enter new values in
the units shown here; no missing parameter values have been estimated.
"""


# PARAMETERS (min^-1)
k_kidney_to_blood_calcitriol = None
k_blood_to_kidney_calcitriol = None
k_liver_to_gut_calcitriol = None
k_gut_to_blood_calcitriol = None
k_blood_to_liver_calcitriol = None
K_liver_to_blood_calcitriol = None  # Capital K follows the PDF's updated name.

# Listed in the PDF but absent from the original ODEs; kept blank and unused.
k_blood_to_gut_calcitriol = None

calcitriol_half_life = None  # min.
basal_calcitriol_produce_kidney = None  # mg/min.
k_increase_calcitriol_produce_by_TPT = None  # mg / (min * mmol TPT).
# If entering an old mg/(min*pmol) value, multiply it by 1e9 first.
# An 80% rise within one day is a calibration target, not a fitted value here.


# INITIAL CONDITIONS (mg)
Initial_calcitriol_kidney = None
Initial_calcitriol_blood = None
initial_calcitriol_liver = None
Initial_calcitriol_gut = None

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
    # Original condition adjustments now update the parameters actually used.

if low_estrogen:
    print("Low estrogen has no assumed calcitriol parameter changes.")
