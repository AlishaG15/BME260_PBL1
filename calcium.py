import numpy as np
from tpt import V_B


V_Ca = V_B   # volume for blood Ca (plasma/ECF), L; using V_B until we find a value


# PARAMETERS

# Sampled quantities (each drawn once, reused below)
f_abs     = np.random.uniform(0.10, 0.50)   # gut Ca absorption fraction
f_bone_Q  = np.random.uniform(0.04, 0.10)   # fraction of cardiac output to bone

# Calcium rate constants (1/min)
k_IG_Ca  = 0.05                          # intake -> gut; ASSUMED (~20 min gastric emptying)
k_GB_Ca  = f_abs * (1/360)               # gut -> blood (ASSUMED 6 h gut transit)
k_Gw_Ca  = (1 - f_abs) * (1/360)         # gut -> waste
k_BK_Ca  = 0.118924 / V_Ca               # blood -> kidney; Qki_in = 0.118924 L/min
k_KB_Ca  = 0.99 * 1.0                    # kidney -> blood; 99% reabsorbed x ASSUMED turnover of 1 1/min
k_Ku_Ca  = 0.01 * 1.0                    # kidney -> urine; remaining 1%
k_BBo_Ca = (f_bone_Q * 5.0) / V_Ca       # blood -> bone; fraction of ~5 L/min cardiac output
m_CaB0   = 95 * V_Ca                     # blood Ca, mg (95 mg/L x V_Ca); defined here so k_BoB_Ca can use it
m_CaBo0  = 1.0e6                         # bone Ca, mg (~1 kg)
k_BoB_Ca = k_BBo_Ca * m_CaB0 / m_CaBo0   # bone -> blood; steady-state flux balance

E_Catr   = None                          # how much calcitriol boosts gut Ca absorption
E_TPT_K  = None                          # how much TPT in kidney boosts Ca reabsorption
E_TPT_Bo = None                          # how much TPT in bone boosts Ca release from bone


# INITIAL CONDITIONS (mg)

m_CaI0  = np.random.uniform(1000, 1200)  # intake from food
m_CaG0  = 0                              # gut; ASSUMED empty at start
# m_CaB0 defined above (95 mg/L x V_Ca)
m_CaK0  = 0                              # kidney
# m_CaBo0 defined above (1.0e6 mg)

mCa0 = [m_CaI0, m_CaG0, m_CaB0, m_CaK0, m_CaBo0]
