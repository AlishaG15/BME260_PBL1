import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# PARAMETERS

# Rate constants [1/min]
k_a   = None   # injection site -> blood
k_BK  = None   # blood -> kidney
k_KB  = None   # kidney -> blood
k_Kd  = None   # kidney -> degraded/filtered
k_BL  = None   # blood -> liver
k_Ld  = None   # liver -> waste
k_BBo = None   # blood -> bone
k_BoB = None   # bone -> blood
k_Bou = None   # bone -> receptor-bound/used

F     = None   # bioavailability of SC injection
V_B   = None   # blood volume
m_PTG = None   # parathyroid gland input rate


# DOSE AND INITIAL CONDITIONS

dose = None

m_SC0 = F*dose   # initial mass at injection site
m_B0  = None   # initial mass in blood
m_K0  = None   # initial mass in kidney
m_L0  = None   # initial mass in liver
m_Bo0 = None   # initial mass in bone

m0 = [m_SC0, m_B0, m_K0, m_L0, m_Bo0]


# MODEL (mass balances)

def model(t, m):
    m_SC, m_B, m_K, m_L, m_Bo = m

    dm_SC = -k_a*m_SC   # injection site
    dm_B  = k_a*m_SC + m_PTG + k_KB*m_K + k_BoB*m_Bo - (k_BK + k_BL + k_BBo)*m_B   # blood
    dm_K  = k_BK*m_B - (k_KB + k_Kd)*m_K   # kidney
    dm_L  = k_BL*m_B - k_Ld*m_L   # liver
    dm_Bo = k_BBo*m_B - (k_BoB + k_Bou)*m_Bo   # bone

    return [dm_SC, dm_B, dm_K, dm_L, dm_Bo]


# SIMULATION

t_start = 0
t_end   = 24*60
dt      = 1
t       = np.arange(0,24*60+dt,dt)   # time points
# no TPT in system after 240min (in rat model)

sol = solve_ivp(model, (t_start, t_end), m0, t_eval=t)       # solve_ivp(model, (t_start, t_end), m0, t_eval=t)
m_SC, m_B, m_K, m_L, m_Bo = sol.y

# OUTPUTS

C_B = m_B / V_B        # blood concentration = m_B / V_B
t_half = None
C_max  = np.max(C_B)
t_max  = t[np.argmax(C_B)]
AUC    = np.trapezoid(C_B, t) #measures total drug exposure - how much drug the blood saw over the whole period

m_tot = m_SC + m_B + m_K + m_L + m_Bo

# PLOTS

# plot mass in each compartment vs time
# plot blood concentration vs time

labels = ["Injection site", "Blood", "Kidney", "Liver", "Bone"]

plt.figure()
for i in range(5):
    plt.plot(t, sol.y[i], label=labels[i])
plt.xlabel("Time [min]")
plt.ylabel("TPT mass")
plt.legend()

plt.figure()
plt.plot(t, C_B)
plt.xlabel("Time [min]")
plt.ylabel("Blood concentration")

plt.show()