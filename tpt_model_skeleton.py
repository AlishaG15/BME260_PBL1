import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# PARAMETERS

# TPT rate constants (1/min)
k_a   = None # injection site -> blood
k_BK  = None # blood -> kidney
k_KB  = None # kidney -> blood
k_Kd  = None # kidney -> degraded
k_BL  = None # blood -> liver
k_Ld  = None # liver -> waste
k_BBo = None # blood -> bone
k_BoB = None # bone -> blood
k_Bou = None # bone -> used up

F     = None # bioavailability
V_B   = None # blood volume
m_PTG = 0 # parathyroid input

# Calcium rate constants (1/min)
k_IG_Ca  = None # intake -> gut
k_GB_Ca  = None # gut -> blood
k_Gw_Ca  = None # gut -> waste
k_BK_Ca  = None # blood -> kidney
k_KB_Ca  = None # kidney -> blood
k_Ku_Ca  = None # kidney -> urine
k_BBo_Ca = None # blood -> bone
k_BoB_Ca = None # bone -> blood


# INITIAL CONDITIONS

# TPT
dose  = None # TPT dose

m_IC0 = None # injection site (F*dose)
m_B0  = 0 # blood
m_K0  = 0 # kidney
m_L0  = 0 # liver
m_Bo0 = 0 # bone

m0 = [m_IC0, m_B0, m_K0, m_L0, m_Bo0]

# Calcium
m_CaI0  = None # intake
m_CaG0  = None # gut
m_CaB0  = None # blood
m_CaK0  = 0 # kidney
m_CaBo0 = None # bone

mCa0 = [m_CaI0, m_CaG0, m_CaB0, m_CaK0, m_CaBo0]


# MASS BALANCE

# TPT
def model(t, m):
    m_IC, m_B, m_K, m_L, m_Bo = m

    dm_IC = -k_a*m_IC # injection site
    dm_B  = k_a*m_IC + m_PTG + k_KB*m_K + k_BoB*m_Bo - (k_BK + k_BL + k_BBo)*m_B # blood
    dm_K  = k_BK*m_B - (k_KB + k_Kd)*m_K # kidney
    dm_L  = k_BL*m_B - k_Ld*m_L # liver
    dm_Bo = k_BBo*m_B - (k_BoB + k_Bou)*m_Bo # bone

    return [dm_IC, dm_B, dm_K, dm_L, dm_Bo]


# Calcium
def CaModel(t, m):
    m_CaI, m_CaG, m_CaB, m_CaK, m_CaBo = m

    dm_CaI  = -k_IG_Ca*m_CaI # intake
    dm_CaG  = k_IG_Ca*m_CaI - (k_GB_Ca + k_Gw_Ca)*m_CaG # gut
    dm_CaB  = k_GB_Ca*m_CaG + k_KB_Ca*m_CaK + k_BoB_Ca*m_CaBo - (k_BK_Ca + k_BBo_Ca)*m_CaB # blood
    dm_CaK  = k_BK_Ca*m_CaB - (k_KB_Ca + k_Ku_Ca)*m_CaK # kidney
    dm_CaBo = k_BBo_Ca*m_CaB - k_BoB_Ca*m_CaBo # bone

    return [dm_CaI, dm_CaG, dm_CaB, dm_CaK, dm_CaBo]


# SIMULATION

t_start = 0
t_end   = 24*60    # 24 hr (min)
dt      = 1        # time step (min)
t       = np.arange(t_start, t_end + dt, dt)
# TPT is gone after ~240 min in the rat data

sol = solve_ivp(model, (t_start, t_end), m0, t_eval=t) # TPT
m_IC, m_B, m_K, m_L, m_Bo = sol.y

solCa = solve_ivp(CaModel, (t_start, t_end), mCa0, t_eval=t) # calcium
m_CaI, m_CaG, m_CaB, m_CaK, m_CaBo = solCa.y


# OUTPUTS

C_B    = m_B / V_B # TPT blood concentration
t_half = None # TPT half-life
C_max  = np.max(C_B) # TPT peak concentration
t_max  = t[np.argmax(C_B)] # time of peak
AUC    = np.trapezoid(C_B, t) # area under curve

m_tot  = m_IC + m_B + m_K + m_L + m_Bo # total TPT

C_CaB  = m_CaB / V_B # Ca blood concentration


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
