import numpy as np

#M is the number of transmitter's antennas
#N is the number of receiver' antennas
#dist is the distance between the transmitter and the receiver
#pl is path-loss exponent
#K is the Rician factor
#eta is the pathloss at the reference distance

def Rician_Fading_Channels(M,N,dist,pl,eta,K):
    mu = np.sqrt(K/((K+1))) #direct path
    s = np.sqrt(1/(2*(K+1))) #scattered path
    Hw = mu + s*(np.random.randn(M,N)+1j*np.random.randn(M,N)) # Rician channel
    H = np.sqrt(eta/(dist**pl))*Hw #Rician channel with pathloss
    return H

# def Spacially_Correlated_Rician():