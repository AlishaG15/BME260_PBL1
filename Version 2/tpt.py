"""TPT parameters. Units: mmol, mg, minutes, and liters unless noted.

Names follow the PDF's Updated Variable Names column. Spaces in names are
replaced by underscores. TPT state amounts are mmol, not mass in mg.
"""

import numpy as np


# PARAMETERS
body_weight = 70_000  # 70 kg = 70,000 g; body weight uses the table's g convention.
mol_weight_TPT = 4117.8  # 4117.8 g/mol = 4117.8 mg/mmol (not 4117.8 mg/mol).
blood_frac = 0.95  # Dimensionless bioavailability.
Vol_blood_in_bone = (0.12 / 1000) * body_weight  # 0.12 L/kg = 0.00012 L/g; 8.4 L.
#Assumption: [equals to V_Ca, the calcium blood-pool volume]; use this name for both.
# Despite the PDF name, this is the apparent TPT distribution volume.

# Draw each uncertain parameter once, then reuse it throughout the simulation.
k_diffuse_TPT = np.random.uniform(0.34, 1.4) / 60  # h^-1 -> min^-1.
# Conversion check: 0.69 h^-1 = 0.0115 min^-1; the PDF's 0.00115 is a typo.
frac_elim_kidney_TPT = np.random.uniform(0.42, 0.68)  # Dimensionless.
frac_elim_toliver_TPT = np.random.uniform(0.0, 0.10)  # Dimensionless.
k_blood_to_bone_TPT = np.random.uniform(0.8, 2.0) / 60  # h^-1 -> min^-1.
#Assumption: [equals to plasma_flow_to_bone_TPT in min^-1]; no separate alias.
bone_uptake_TPT = np.random.uniform(0.2, 0.5)  # Dimensionless extraction fraction.

k_elimination_TPT = 7.4 / 60  # 7.4 h^-1 = 0.123333... min^-1.
k_blood_to_kidney_TPT = frac_elim_kidney_TPT * k_elimination_TPT  # min^-1.
#Assumption: [equals to k_elimin_kidney_TPT]; use this rate for kidney degradation too.
k_kidney_to_blood_TPT = 0.0  # min^-1; negligible intact TPT return.
k_bloodtoliver_TPT = frac_elim_toliver_TPT * k_elimination_TPT  # min^-1.
#Assumption: [equals to k_elimin_liver_TPT]; use this rate for liver degradation too.
k_bone_to_blood_TPT = (1 - bone_uptake_TPT) * k_blood_to_bone_TPT  # min^-1.
TPT_to_bone_receptor = bone_uptake_TPT * k_blood_to_bone_TPT  # min^-1.
k_blood_to_other_tissue = (
    1 - frac_elim_kidney_TPT - frac_elim_toliver_TPT
) * k_elimination_TPT  # min^-1.

PTH_endogenous_blood_mmol_per_min = 0.0  # mmol/min
# Normal PTH is already built into the basal rates (e.g. 99% kidney Ca reabsorption),
# so we only track injected TPT. Literature range was 2.3-5.6 pmol/min.
TPT_half_life = 5.6  # min; retained original comparison value, not used in the ODEs.


# INITIAL CONDITIONS (mmol)
dose = 4857e-9  # 4857 pmol = 0.000004857 mmol; rounded original 20 microgram dose.
initial_TPT_injection = blood_frac * dose
initial_TPT_blood = 0.0
initial_TPT_kidney = 0.0
initial_TPT_liver = 0.0
initial_TPT_bone = 0.0

initial_TPT = [
    initial_TPT_injection,
    initial_TPT_blood,
    initial_TPT_kidney,
    initial_TPT_liver,
    initial_TPT_bone,
]
