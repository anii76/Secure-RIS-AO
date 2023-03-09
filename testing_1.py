import numpy as np

from rician_fading import Rician_Fading_Channels

K = 1 # Rician factor
eta = 1 #-30 dB , for the sake of problems (negative number under sqrt) keep it 1
M = 4 # antenna at BS
n = 1 # antenna at UE/EVE
N = 25 # RIS elements
dist_AU = np.sqrt((150 - 0)**2 + (0 - 0)**2)
dist_AE = np.sqrt((145 - 0)**2 + (0 - 0)**2)
dist_AI = np.sqrt((145 - 0)**2 + (5 - 0)**2)
dist_IE = np.sqrt((145 - 145)**2 + (0 - 5)**2)
dist_IU = np.sqrt((150 - 145)**2 + (0 - 5)**2)
pl = 3
pl_AI = 2.2

N_samples = 1000 # Channel realisations

H_AU = np.array([Rician_Fading_Channels(M,n,dist_AU,pl,eta,K) for _ in range(N_samples)])
H_AE = np.array([Rician_Fading_Channels(M,n,dist_AE,pl,eta,K) for _ in range(N_samples)])
H_AI = np.array([Rician_Fading_Channels(M,N,dist_AI,pl_AI,eta,K) for _ in range(N_samples)])
H_IU = np.array([Rician_Fading_Channels(N,n,dist_IU,pl,eta,K) for _ in range(N_samples)])
H_IE = np.array([Rician_Fading_Channels(N,n,dist_IE,pl,eta,K) for _ in range(N_samples)])