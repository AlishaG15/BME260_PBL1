import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

f_bone_Q  = np.random.uniform(0.04, 0.10)   # fraction of cardiac output to bone
f_abs     = np.random.uniform(0.10, 0.50)   # gut Ca absorption fraction

ka_h      = np.random.uniform(0.34, 1.4)    # injection -> blood, 1/h
# Sampled quantities (each drawn once, reused below)

fL        = 0.000005   # fraction of elimination going to liver
Qbone_Vb  = 1.26     # bone plasma flow / V_B, 1/h
EB        = 0.316 # bone extraction ratio


# TPT rate constants (1/min)
ke    = 7.4 / 60                 # elimination rate constant (7.4 1/h) = 0.1233 1/min

k_a   = ka_h / 60                # injection site -> blood
k_BK  = fK * ke                  # blood -> kidney
k_KB  = 0.0                      # kidney -> blood (set to 0, <1.2% returned intact)
k_Kd  = k_BK                     # kidney -> degraded; ASSUMED equal to k_BK so kidney pool doesn't accumulate
k_BL  = fL * ke                  # blood -> liver
k_Ld  = k_BL                     # liver -> waste; 100% degraded (ASSUMED same constant)
k_BBo = Qbone_Vb / 60            # blood -> bone
k_BoB = (1 - EB) * k_BBo         # bone -> blood
k_Bou = EB * k_BBo               # bone -> used up (internalized)

F     = 0.95                     # bioavailability
V_B   = 8.4                      # blood volume, L (0.12 L/kg x 70 kg)
m_PTG = np.random.uniform(2.3, 5.6)  # parathyroid input, pmol/min

# Calcium rate constants (1/min)
k_IG_Ca  = 0.05                          # intake -> gut; ASSUMED (~20 min gastric emptying)
k_GB_Ca  = f_abs * (1/360)               # gut -> blood (ASSUMED 6 h gut transit)
k_Gw_Ca  = (1 - f_abs) * (1/360)         # gut -> waste
k_BK_Ca  = 0.118924 / V_B                # blood -> kidney; Qki_in = 0.118924 L/min
k_KB_Ca  = 0.99 * 1.0                    # kidney -> blood; 99% reabsorbed x ASSUMED turnover of 1 1/min
k_Ku_Ca  = 0.01 * 1.0                    # kidney -> urine; remaining 1%
k_BBo_Ca = (f_bone_Q * 5.0) / V_B        # blood -> bone; fraction of ~5 L/min cardiac output
m_CaB0   = 95 * V_B                      # blood Ca, mg (95 mg/L x V_B); defined here so k_BoB_Ca can use it
m_CaBo0  = 1.0e6                         # bone Ca, mg (~1 kg)
k_BoB_Ca = k_BBo_Ca * m_CaB0 / m_CaBo0   # bone -> blood; steady-state flux balance

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
dose  = 4857                # TPT dose, pmol (20 ug / 4117.8 g/mol); ASSUMED standard dose

m_IC0 = F * dose            # injection site (F*dose)
m_B0  = 0                   # blood
m_K0  = 0                   # kidney
m_L0  = 0                   # liver
m_Bo0 = 0                   # bone

m0 = [m_IC0, m_B0, m_K0, m_L0, m_Bo0]

# Calcium (mg)
m_CaI0  = np.random.uniform(1000, 1200)  # intake from food
m_CaG0  = 0                              # gut; ASSUMED empty at start
# m_CaB0 defined above (95 mg/L x V_B)
m_CaK0  = 0                              # kidney
# m_CaBo0 defined above (1.0e6 mg)

mCa0 = [m_CaI0, m_CaG0, m_CaB0, m_CaK0, m_CaBo0]

# calcitriol 
m_CatrK0 = None # kidney
m_CatrB0 = None # blood
m_CatrL0 = None # liver
m_CatrG0 = None # gut

mCatr0 = [m_CatrK0, m_CatrB0, m_CatrL0, m_CatrG0]

# MASS BALANCE

# TPT
def TPTmodel(t, m):
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

def simulation(state )
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
