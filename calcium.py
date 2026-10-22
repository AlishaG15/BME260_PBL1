import numpy as np
from tpt import V_B
from calcitriol import 

V_Ca = V_B   # volume for blood Ca (plasma/ECF), L; using V_B until we find a value


# PARAMETERS

# Sampled quantities (each drawn once, reused below)

f_bone_Q  = np.random.uniform(0.04, 0.10)   # fraction of cardiac output to bone
gamma= np.random.uniform(0.2, 1.0)          # calcitriol sensitivity parameter for gut Ca absorption                               #mass flow of calcitriol in blood, units?
D0= np.random.uniform(18e-9, 78e-9)  # reference calcitriol concentration, mg/mL
# Calcium rate constants (1/min)
E_Catr   =  1 + gamma*(dmCatrB/D0 - 1)          # how much calcitriol boosts gut Ca absorption
E_TPT_K  =  np.random.uniform(0.1, 1.0)    # how much TPT in kidney boosts Ca reabsorption
E_TPT_Bo =  np.random.uniform(1, 2.9)    # how much TPT in bone boosts Ca release from bone
k0_gut   = 0.3                          # basal gut Ca absorption fraction (ASSUMED; 0.3 = 30% of intake absorbed)
k0_kidney = 0.99  
k0_bone= 0.40# basal kidney Ca reabsorption fraction (ASSUMED; 0.99 = 99% of filtered Ca reabsorbed)
m_CaI0  = np.random.uniform(1000, 1200)
k_IG_Ca  = 0.05                          # intake -> gut; ASSUMED (~20 min gastric emptying)
k_GB_Ca  = k0_gut* E_Catr     # absorption from gut -> blood 
k_Gw_Ca  = (1 - k0_gut) * ()         # gut -> waste
k_BK_Ca  = 0.118924 / V_Ca              # blood -> kidney; Qki_in = 0.118924 L/min
k_KB_Ca  = k0_kidney*(1 + E_TPT_K*dmB/ (dmB + (( 0.316)*dmB)))                 # kidney -> blood; 99% reabsorbed x ASSUMED turnover of 1 1/min
k_Ku_Ca  = (1-k_KB_Ca)                 # kidney -> urine; remaining 1%
k_BBo_Ca = k0_bone*(1 + E_TPT_Bo*dmB/ (dmB + (( 0.316)*dmB)))       # blood -> bone; fraction of ~5 L/min cardiac output
m_CaB0   = 95 * V_Ca                     # blood Ca, mg (95 mg/L x V_Ca); defined here so k_BoB_Ca can use it
m_CaBo0  = 1.0e6                         # bone Ca, mg (~1 kg)
k_BoB_Ca = k_BBo_Ca * m_CaB0 / m_CaBo0   # bone -> blood; steady-state flux balance



# INITIAL CONDITIONS (mg)

  # intake from food
m_CaG0  = 0                              # gut; ASSUMED empty at start
# m_CaB0 defined above (95 mg/L x V_Ca)
m_CaK0  = 0                              # kidney
# m_CaBo0 defined above (1.0e6 mg)

mCa0 = [m_CaI0, m_CaG0, m_CaB0, m_CaK0, m_CaBo0]
