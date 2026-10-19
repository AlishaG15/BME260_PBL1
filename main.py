import numpy as np
np.random.seed(0) # same random draws every run (must be before the imports below)
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from tpt import *
from calcium import *
from calcitriol import *

# Later: pick a state here (healthy, osteoporosis, renal failure, sex) and change parameters before solving


# MASS BALANCE

# TPT
def model(t, m):
    m_IC, m_B, m_K, m_L, m_Bo = m

    dm_IC = -k_a*m_IC # injection site
    dm_B  = k_a*m_IC + m_PTG + k_KB*m_K + k_BoB*m_Bo - (k_BK + k_BL + k_BBo + k_Bx)*m_B # blood
    dm_K  = k_BK*m_B - (k_KB + k_Kd)*m_K # kidney
    dm_L  = k_BL*m_B - k_Ld*m_L # liver
    dm_Bo = k_BBo*m_B - (k_BoB + k_Bou)*m_Bo # bone

    return [dm_IC, dm_B, dm_K, dm_L, dm_Bo]


# Calcitriol
def CatrModel(t, m):
    m_CatrK, m_CatrB, m_CatrL, m_CatrG = m
    m_K_t = np.interp(t, sol.t, m_K) # TPT in kidney at time t
    k_Kd_Catr = np.log(2) / t_half_Catr # kidney -> degraded, from half-life

    dm_CatrK = k_p_Catr + k_TPT_Catr*m_K_t + k_BK_Catr*m_CatrB - (k_KB_Catr + k_Kd_Catr)*m_CatrK # kidney
    dm_CatrB = k_KB_Catr*m_CatrK + k_LB_Catr*m_CatrL + k_GB_Catr*m_CatrG - (k_BK_Catr + k_BL_Catr)*m_CatrB # blood
    dm_CatrL = k_BL_Catr*m_CatrB - (k_LB_Catr + k_LG_Catr)*m_CatrL # liver
    dm_CatrG = k_LG_Catr*m_CatrL - k_GB_Catr*m_CatrG # gut

    return [dm_CatrK, dm_CatrB, dm_CatrL, dm_CatrG]


# Calcium
def CaModel(t, m):
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

    return [dm_CaI, dm_CaG, dm_CaB, dm_CaK, dm_CaBo]


# SIMULATION

t_start = 0
t_end   = 24*60    # 24 hr (min)
dt      = 1        # time step (min)
t       = np.arange(t_start, t_end + dt, dt)
# TPT is gone after ~240 min in the rat data

sol = solve_ivp(model, (t_start, t_end), m0, t_eval=t) # TPT
m_IC, m_B, m_K, m_L, m_Bo = sol.y

solCatr = solve_ivp(CatrModel, (t_start, t_end), mCatr0, t_eval=t) # calcitriol
m_CatrK, m_CatrB, m_CatrL, m_CatrG = solCatr.y

solCa = solve_ivp(CaModel, (t_start, t_end), mCa0, t_eval=t) # calcium
m_CaI, m_CaG, m_CaB, m_CaK, m_CaBo = solCa.y



# OUTPUTS

C_B    = m_B / V_B # TPT blood concentration, pmol/L
C_B_ngmL = C_B * MW_TPT / 1e6 # TPT blood concentration, ng/mL (same units as the data)
C_max  = np.max(C_B) # TPT peak concentration
t_max  = t[np.argmax(C_B)] # time of peak
AUC    = np.trapezoid(C_B, t) # area under curve

m_tot  = m_IC + m_B + m_K + m_L + m_Bo # total TPT

C_CaB  = m_CaB / V_Ca # Ca blood concentration


# PLOTS

# TPT mass
labels = ["Injection site", "Blood", "Kidney", "Liver", "Bone"]
plt.figure()
for i in range(5):
    plt.plot(t, sol.y[i], label=labels[i])
plt.xlabel("Time (min)")
plt.ylabel("TPT mass")
plt.legend()

# TPT blood concentration
plt.figure()
plt.plot(t, C_B)
plt.xlabel("Time (min)")
plt.ylabel("TPT blood concentration")

# Calcium mass
labelsCa = ["Intake", "Gut", "Blood", "Kidney", "Bone"]
plt.figure()
for i in range(5):
    plt.plot(t, solCa.y[i], label=labelsCa[i])
plt.xlabel("Time (min)")
plt.ylabel("Calcium mass")
plt.legend()

plt.show()
