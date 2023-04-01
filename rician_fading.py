import numpy as np
from scipy.linalg import sqrtm
# all channal coefficients are in dB ?

#M is the number of transmitter's antennas
#N is the number of receiver' antennas
#dist is the distance between the transmitter and the receiver
#pl is path-loss exponent
#zeta is the pathloss at the reference distance
#K is the Rician factor
#r is correlation coefficient
#https://github.com/havutran/Rician-fading-by-Python
#https://dsp.stackexchange.com/questions/55719/channel-model-los-component-and-rician-k-factor


def Rician_Fading_Channels(M,N,dist,pl,K):
    path_loss = 1/(dist**pl)
    #d0 = 1; path_loss = zeta + 10 * pl * np.log10(dist/d0) #should i keep this or go back to path_loss = 1/(dist**pl)

    mu = np.sqrt(K/((K+1))) #direct path (mean)
    s = np.sqrt(1/(2*(K+1))) #scattered path (sigma) [Standard deviation]

    phi = np.random.uniform(0, 2*np.pi) # instead of AoA and doppler shift (or consider them all equal to 0 )
    Hw_LOS = mu * np.exp(1j * phi) #* np.exp(1j * f_D * cos(AoA) + phase_shift) (https://www.gaussianwaves.com/2020/08/rician-flat-fading-channel-simulation/)
    Hw_NLOS = s * (np.random.randn(N,M) + 1j * np.random.randn(N,M)) 
    Hw = Hw_LOS + Hw_NLOS # Rician channel
    H = np.sqrt(path_loss)*Hw #Rician channel with pathloss
    return H


def Spatially_Correlated_Rician(M,dist,pl,zeta,K,r):
    # Path loss
    d0 = 1 
    zeta = 10**(zeta/10)
    path_loss = zeta*(d0/dist)**pl

    # Small-scale fading component
    phi = np.random.uniform(0, 2*np.pi)
    g_LOS = np.sqrt(K/(K+1)) * np.exp(1j * phi) #because AoA is 0 (00,0),(145,0),(150,0) ?
    g_NLOS = np.sqrt(1/(2*(K+1))) * (np.random.randn(M) + 1j * np.random.randn(M))
    R = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            R[i,j] = r ** np.abs(i-j) #not sure of this line (applying correlation)
                                            #probably correct (see how to add variance to random distribution in numpy: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html#numpy.random.randn)
    B = sqrtm(R)
    small_scale_fading =  g_LOS + B @ g_NLOS

    # Channel coefficients
    H = np.sqrt(path_loss) * small_scale_fading  
                #converting to linear units
    return np.array([H])