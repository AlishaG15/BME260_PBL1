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

mCatr0 = [m_CatrK0, m_CatrB0, m_CatrL0, m_CatrG0]
