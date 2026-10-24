"""TPT parameters. Units: mmol, mg, minutes, and liters unless noted.

Names follow the team's variable table. TPT state amounts are mmol, not mass in mg.
"""

import numpy as np


# PARAMETERS
body_weight = 70_000  # 70 kg = 70,000 g; body weight uses the table's g convention.
mol_weight_TPT = 4117.8  # 4117.8 g/mol = 4117.8 mg/mmol (not 4117.8 mg/mol).
blood_frac = 0.95  # Dimensionless bioavailability.
Vol_blood_in_bone = (0.12 / 1000) * body_weight  # 0.12 L/kg = 0.00012 L/g; 8.4 L.
# ASSUMPTION: this 8.4 L TPT volume is also used as the blood volume for calcium and calcitriol, so Ca
# is reported per 8.4 L (95 mg/L -> 798 mg). Real pools differ (plasma ~3 L, extracellular fluid ~14 L); because
# results are compared as TREATED minus CONTROL concentrations, one shared volume keeps the three models consistent.
# Despite the name, this is the apparent TPT distribution volume.

# Draw each uncertain parameter once, then reuse it throughout the simulation.
k_diffuse_TPT = np.random.uniform(0.34, 1.4) / 60  # h^-1 -> min^-1.
frac_elim_kidney_TPT = np.random.uniform(0.42, 0.68)  # Dimensionless.
frac_elim_toliver_TPT = np.random.uniform(0.0, 0.10)  # Dimensionless.
k_blood_to_bone_TPT = np.random.uniform(0.8, 2.0) / 60  # h^-1 -> min^-1.
bone_uptake_TPT = np.random.uniform(0.2, 0.5)  # Dimensionless extraction fraction.

k_elimination_TPT = 7.4 / 60  # 7.4 h^-1 = 0.123333... min^-1.
k_blood_to_kidney_TPT = frac_elim_kidney_TPT * k_elimination_TPT  # min^-1.
# ASSUMPTION: no source gives a separate kidney degradation rate for TPT, so kidney TPT is degraded
# at the same rate it arrives (half-life ~9-13 min). This sets the kidney TPT time course behind the kidney Hill terms.
k_kidney_degrade_TPT = k_blood_to_kidney_TPT  # min^-1; kidney -> degraded (same as arrival rate)
k_kidney_to_blood_TPT = 0.0  # min^-1; negligible intact TPT return.
k_bloodtoliver_TPT = frac_elim_toliver_TPT * k_elimination_TPT  # min^-1.
# ASSUMPTION: liver TPT is degraded at the same rate it arrives.
k_bone_to_blood_TPT = (1 - bone_uptake_TPT) * k_blood_to_bone_TPT  # min^-1.
TPT_to_bone_receptor = bone_uptake_TPT * k_blood_to_bone_TPT  # min^-1.
# k_elimination_TPT is ALL TPT clearance, so kidney, liver AND bone uptake must fit inside it.
# Bone permanently takes up TPT_to_bone_receptor (min^-1), so "other tissue" only gets what is left.
k_blood_to_other_tissue = (
    (1 - frac_elim_kidney_TPT - frac_elim_toliver_TPT) * k_elimination_TPT - TPT_to_bone_receptor
)  # min^-1

PTH_endogenous_blood_mmol_per_min = 0.0  # mmol/min
# ASSUMPTION: normal PTH (literature 2.3-5.6 pmol/min) is already built into the healthy basal rates,
# so only injected TPT is tracked. The body's PTH response to a Ca rise is represented by k_Ca_feedback in calcium.py.
TPT_half_life = 60.0  # min; apparent half-life after SC injection (Satterwhite 2010, ~1 h); comparison only, not used in the ODEs.
# (The 5.6 min value is the blood elimination half-life, ln2 / k_elimination_TPT.)


# INITIAL CONDITIONS (mmol)
dose = 4857e-9  # 4857 pmol = 0.000004857 mmol; 20 microgram dose (20 ug / 4117.8 g/mol).
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
