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


def CaModel(t, m, calcitriol_b, pth_blood):
    m_CaI, m_CaG, m_CaB, m_CaK, m_CaBo = m
    m_CatrG_t = np.interp(t, solCatr.t, m_CatrG) # calcitriol in gut at time t
    m_K_t     = np.interp(t, sol.t, m_K) # TPT in kidney at time t
    m_Bo_t    = np.interp(t, sol.t, m_Bo) # TPT in bone at time t

    k_GB_t  = k_GB_Ca*(1 + E_Catr*m_CatrG_t) # gut Ca absorption with calcitriol
    k_KB_t  = k_KB_Ca*(1 + E_TPT_K*m_K_t) # kidney Ca reabsorption with TPT
    k_BoB_t = k_BoB_Ca*(1 + E_TPT_Bo*m_Bo_t) # bone Ca release with TPT

    dm_CaI  = -k_IG_Ca*m_CaI # intake
    dm_CaG  = k_IG_Ca*m_CaI - (k_GB_t + k_Gw_Ca)*m_CaG # gut
    dm_CaB  = k_GB_t*m_CaG + k_KB_t*m_CaK + k_BoB_t*m_CaBo - (k_BK_Ca + k_BBo_Ca)*m_CaB # blood
    dm_CaK  = k_BK_Ca*m_CaB - (k_KB_t + k_Ku_Ca)*m_CaK # kidney
    dm_CaBo = k_BBo_Ca*m_CaB - k_BoB_t*m_CaBo # bone

    return [calcium_blood, calcium_gut, calcium_blood]


