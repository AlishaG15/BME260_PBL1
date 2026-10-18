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

# Calcitriol rate constants
k_KB_Catr  = None # kidney -> blood
k_BK_Catr  = None # blood -> kidney
k_LG_Catr  = None # liver -> gut
k_GB_Catr  = None # gut -> blood
k_BL_Catr  = None # blood -> liver
k_LB_Catr  = None # liver -> blood
k_Kd_Catr  = None # kidney -> degraded
k_p_Catr   = None # production in kidney (mass/min)
k_TPT_Catr = None # extra production from TPT in kidney
E_Catr     = None # how much calcitriol boosts gut Ca absorption

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

# calcitriol 
m_CatrK0 = None # kidney
m_CatrB0 = None # blood
m_CatrL0 = None # liver
m_CatrG0 = None # gut

mCatr0 = [m_CatrK0, m_CatrB0, m_CatrL0, m_CatrG0]

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
    m_CatrG_t = np.interp(t, solCatr.t, m_CatrG) # calcitriol in gut at time t
    k_GB = k_GB_Ca*(1 + E_Catr*m_CatrG_t) # gut Ca absorption with calcitriol
    
    m_CaI, m_CaG, m_CaB, m_CaK, m_CaBo = m

    dm_CaI  = -k_IG_Ca*m_CaI # intake
    dm_CaG  = k_IG_Ca*m_CaI - (k_GB + k_Gw_Ca)*m_CaG # gut
    dm_CaB  = k_GB*m_CaG + k_KB_Ca*m_CaK + k_BoB_Ca*m_CaBo - (k_BK_Ca + k_BBo_Ca)*m_CaB # blood
    dm_CaK  = k_BK_Ca*m_CaB - (k_KB_Ca + k_Ku_Ca)*m_CaK # kidney
    dm_CaBo = k_BBo_Ca*m_CaB - k_BoB_Ca*m_CaBo # bone

    return [dm_CaI, dm_CaG, dm_CaB, dm_CaK, dm_CaBo]

# Calcitriol
def CatrModel(t, m):
    m_CatrK, m_CatrB, m_CatrL, m_CatrG = m
    m_K_t = np.interp(t, sol.t, m_K) #TPT in kidney at time T
    
    dm_CatrK = k_p_Catr + k_TPT_Catr*m_K_t + k_BK_Catr*m_CatrB - (k_KB_Catr + k_Kd_Catr)*m_CatrK # kidney
    dm_CatrB = k_KB_Catr*m_CatrK + k_LB_Catr*m_CatrL + k_GB_Catr*m_CatrG - (k_BK_Catr + k_BL_Catr)*m_CatrB # blood
    dm_CatrL = k_BL_Catr*m_CatrB - (k_LB_Catr + k_LG_Catr)*m_CatrL # liver
    dm_CatrG = k_LG_Catr*m_CatrL - k_GB_Catr*m_CatrG # gut

    return [dm_CatrK, dm_CatrB, dm_CatrL, dm_CatrG]

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
