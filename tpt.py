import numpy as np


# PARAMETERS

BW     = 70                      # body weight, kg
MW_TPT = 4117.8                  # TPT molecular weight, g/mol

# Sampled quantities (each drawn once, reused below)
ka_h      = np.random.uniform(0.34, 1.4)    # injection -> blood, 1/h
fK        = np.random.uniform(0.42, 0.68)   # fraction of elimination going to kidney
fL        = np.random.uniform(0.0, 0.10)    # fraction of elimination going to liver
Qbone_Vb  = np.random.uniform(0.8, 2.0)     # bone plasma flow / V_B, 1/h
EB        = np.random.uniform(0.2, 0.5)     # bone extraction ratio

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
k_Bx  = (1 - fK - fL) * ke       # blood -> other tissues (degraded); rest of elimination

F     = 0.95                     # bioavailability
V_B   = 0.12 * BW                # TPT volume of distribution, L (0.12 L/kg)
m_PTG = np.random.uniform(2.3, 5.6)  # parathyroid input, pmol/min
t_half = 5.6 # TPT half-life after SC injection (human ~1 h, FDA label); model output to check against


# INITIAL CONDITIONS

dose  = 4857                # TPT dose, pmol (20 ug / 4117.8 g/mol); ASSUMED standard dose

m_IC0 = F * dose            # injection site (F*dose)
m_B0  = 0                   # blood
m_K0  = 0                   # kidney
m_L0  = 0                   # liver
m_Bo0 = 0                   # bone

m0 = [m_IC0, m_B0, m_K0, m_L0, m_Bo0]
