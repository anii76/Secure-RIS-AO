import numpy as np

from rician_fading import *

K = 1  # Rician factor
zeta = -30 # -30 dB 
M = 4  # antenna at BS
n = 1  # antenna at UE/EVE
N = 25  # RIS elements
dist_AU = np.sqrt((150 - 0)**2 + (0 - 0)**2)   # 150m 
dist_AE = np.sqrt((145 - 0)**2 + (0 - 0)**2)   # 145m
dist_AI = np.sqrt((145 - 0)**2 + (5 - 0)**2)   # 145.08...m
dist_IE = np.sqrt((145 - 145)**2 + (0 - 5)**2) # 5m
dist_IU = np.sqrt((150 - 145)**2 + (0 - 5)**2) # 10m
pl = 3
pl_AI = 2.2
r_factor = 0.95 # correlation factor

# 1000 Channel realisations
N_samples = int(input('Enter number of samples >>> '))


H_AU = np.array([Spatially_Correlated_Rician(M, dist_AU, pl, zeta, K, r_factor)
                for _ in range(N_samples)])
H_AE = np.array([Spatially_Correlated_Rician(M, dist_AE, pl, zeta, K ,r_factor)
                for _ in range(N_samples)])

#H_AU = np.array([Rician_Fading_Channels(M, 1, dist_AU, pl, K)
#                for _ in range(N_samples)])
#H_AE = np.array([Rician_Fading_Channels(M, 1, dist_AE, pl, K)
#                for _ in range(N_samples)])
H_AI = np.array([Rician_Fading_Channels(M, N, dist_AI, pl_AI, K)
                for _ in range(N_samples)])
H_IU = np.array([Rician_Fading_Channels(N, n, dist_IU, pl, K)
                for _ in range(N_samples)])
H_IE = np.array([Rician_Fading_Channels(N, n, dist_IE, pl, K)
                for _ in range(N_samples)])

Channels = {
    "H_AU": H_AU, 
    "H_AE": H_AE,
    "H_AI": H_AI,
    "H_IU": H_IU,
    "H_IE": H_IE
    }

np.save('Channels',Channels)
