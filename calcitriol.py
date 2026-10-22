import numpy as np


# PARAMETERS

# Calcitriol rate constants
k_KB_Catr  = None # kidney -> blood
k_BK_Catr  = None # blood -> kidney
k_LG_Catr  = None # liver -> gut
k_GB_Catr  = None # gut -> blood
k_BL_Catr  = None # blood -> liver
k_LB_Catr  = None # liver -> blood
t_half_Catr = None # calcitriol half-life (min)
k_p_Catr   = None # production in kidney (mass/min)
k_TPT_Catr = None # extra production from TPT in kidney


# INITIAL CONDITIONS

m_CatrK0 = None # kidney
m_CatrB0 = None # blood
m_CatrL0 = None # liver
m_CatrG0 = None # gut

# 1 dose of tpt causes 80% increase in calcitriol

mCatr0 = [m_CatrK0, m_CatrB0, m_CatrL0, m_CatrG0]

# Insert Condition (change to true)
osteoporosis = False
renal_failure = False
low_estrogen = False

# Copy healthy values to patient
patient = {
    "k_LG_Catr": k_LG_Catr,  # liver -> gut
    "k_BL_Catr": k_BL_Catr,  # blood -> liver
    "k_LB_Catr": k_LB_Catr,  # liver -> blood
    "k_KB_Catr": k_KB_Catr,  # kidney -> blood
    "k_BK_Catr": k_BK_Catr,  # blood -> kidney
    "k_BG_Catr": k_BG_Catr,  # blood -> gut
    "k_GB_Catr": k_GB_Catr,  # gut -> blood
    "k_p_Catr": k_p_Catr,
    "k_TPT_Catr": k_TPT_Catr,
    "t_half_Catr": t_half_Catr,
}

# Osteoporosis
if osteoporosis:
    print("Osteoporosis has no assumed calcitriol parameter changes.")

# Renal Fail
if renal_failure:
    if k_p_Catr is None or t_half_Catr is None:
        raise ValueError("Enter healthy k_p_Catr and t_half_Catr values first.")
    patient["k_p_Catr"] = k_p_Catr * (14.2 / 31.8)
  # k_p_Catr is healthy production rate. The ratio 14.2/31.8 is renal failure rate with healthy rate. The patient produces about 45% as much calcitriol.
    print("Renal failure causes calcitriol production to decrease about 45% of healthy baseline.")
    patient["t_half_Catr"] = t_half_Catr / (0.45 / 0.58)
  # The ratio 0.45/0.58 is existing calcitriol being removed more slowly about 78% of healthy clearance
    print("Renal failure: half-life increased by about 29%.")

# Low Estrogen
if low_estrogen:
    print("Low estrogen has no assumed calcitriol parameter changes.")
  
# Show condition effects
for parameter, value in patient.items():
    print(f"{parameter} = {value}")
