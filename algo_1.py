import numpy as np
from scipy.linalg import svd
import cvxpy as cp

from testing_1 import *

M = 4 # Number of antennas at BS
N = 25 # Number of RIS elements
noise_power = -80 #dBm
transmit_power = 20 #dBm

# I need to fix channel user 0 and eve 0 for the algorithm (one test) 
h_IU = H_IU[0].T # 1xN
h_AU = H_AU[0].T # 1xM
h_IE = H_IE[0].T # 1xN
h_AE = H_AE[0].T # 1xM
H_AI = H_AI[0].T # NxM

# epsilon 
epsilon = 1e-3

def secrecy_rate(w,q):
    Q = np.diag(q)
    ue_rate = np.log2(1 + (abs((h_IU @ Q @ H_AI + h_AU) @ w)/noise_power)**2)
    eve_rate = np.log2(1 + (abs((h_IE @ Q @ H_AI + h_AE) @ w)/noise_power)**2)
    return (ue_rate - eve_rate)[0][0] #if ue_rate > eve_rate else 0 # array shape (1,)

# matrices A and B (9),(8)
def A(q):
    Q = np.diag(q)
    A = ((h_IU @ Q @ H_AI + h_AU).T.conjugate() @ (h_IU @ Q @ H_AI + h_AU))/noise_power**2
    return A # shape (4,4)

def B(q):
    Q = np.diag(q)
    B = ((h_IE @ Q @ H_AI + h_AE).T.conjugate() @ (h_IE @ Q @ H_AI + h_AE))/noise_power**2
    return B # shape (4,4)

# calculate eigen values of matrix U
def U_max(B, A):
    # Il faut ajouter un check pour que soit la 1ere () inversible
    U = np.linalg.inv(B + np.identity(M)/transmit_power) @ (A + np.identity(M)/transmit_power) #shape (4,4)
    eigvals, eigvecs = np.linalg.eig(U) 
    max_idx = np.argmax(eigvals)
    eigen_U = eigvecs[:, max_idx]
    
    # to be normalized
    norm_r = np.linalg.norm(eigen_U.real)
    norm_i = np.linalg.norm(eigen_U.imag)
    normalized_eig = eigen_U.real / norm_r + (eigen_U.imag / norm_i)*1j
    
    return np.array([normalized_eig]).T # to have consistent shape of (4,1)

# solve problem (22) using SDR
def problem_22(w,q):
    h_U = (h_AU.conjugate() @ w.conjugate() @ w.T @ h_AU.T) / noise_power**2
    h_E = (h_AE.conjugate() @ w.conjugate() @ w.T @ h_AE.T) / noise_power**2

    # Constructing G_U and G_E  
    # G_U
    bloc_1 = np.diagflat(h_IU.conjugate()) @ H_AI.conjugate() @ w.conjugate() @ w.T @ H_AI.T @ np.diagflat(h_IU)
    bloc_2 = np.diagflat(h_IU.conjugate()) @ H_AI.conjugate() @ w.conjugate() @ w.T @ h_AU.T
    bloc_3 = h_AU.conjugate() @ w.conjugate() @ w.T @ H_AI.T @ np.diagflat(h_IU)
    bloc_4 = np.array([np.append(bloc_3,0)])

    G_U = np.concatenate((np.concatenate((bloc_1, bloc_2), axis=1), bloc_4), axis=0) # equation (17)

    # G_E
    bloc_1 = np.diagflat(h_IE.conjugate()) @ H_AI.conjugate() @ w.conjugate() @ w.T @ H_AI.T @ np.diagflat(h_IE)
    bloc_2 = np.diagflat(h_IE.conjugate()) @ H_AI.conjugate() @ w.conjugate() @ w.T @ h_AE.T
    bloc_3 = h_AE.conjugate() @ w.conjugate() @ w.T @ H_AI.T @ np.diagflat(h_IE)
    bloc_4 = np.array([np.append(bloc_3,0)])

    G_E = np.concatenate((np.concatenate((bloc_1, bloc_2), axis=1), bloc_4), axis=0) # equation (17)


     #E = [np.zeros((i,i)) for i in range(1,N+2)] # (a 3 dimentional matrix (i,j,n)) n : {1,26}
    E = np.zeros((N+1,N+1,N+1)) # 26
    for i in range(N+1):
        E[i][i][i] = 1
    #  s[:-1].T.conjugate() @ E[24] @ s[:-1] verifies == 1 (n=25)

    ######################
    # Problem parameters #
    ######################

    # s contains the element to optimise q
    s = np.append(q,1) # q ki ydkhol rah déjà q.T
    s = np.array([s]).T
    S = s @ s.T.conjugate() # S ~= ss^H (deta equal) # so a scalar not a matrix ? # after adjustments its a matrix of dim(s)xdim(s)
                            # np.linalg.matrix_rank(S) verifies 1

    µ = 1/(np.trace(G_E @ S) + h_E + 1) # µ = µ[0][0] 
    X = µ * S # la variable à optimiser;

    # Code for reference
    #objective_function = np.trace(G_U @ X) + µ * (h_U + 1)
    #constraint_1 = np.trace(G_E @ X) + µ * (h_E + 1) # == 1
    #constraint_2 = np.trace(E[n] @ X) # == µ, quelque soit n

    ## Defining optimisation variable X and µ (X.shape) <=> (N+1,N+1)
    Z_r = cp.Variable((N+1, N+1), PSD=True)
    Z_i = cp.Variable((N+1, N+1), PSD=True)

    µ_r = cp.Variable(nonneg=True)
    µ_i = cp.Variable(nonneg=True)

    # Not sure the utility of this 
    Z = cp.hstack([cp.vstack([Z_r, -Z_i]), cp.vstack([Z_i, Z_r])])

    ## Defining opjective function
    f = ( cp.trace(G_U.real @ Z_r - G_U.imag @ Z_i) + cp.imag(cp.trace(G_U.real @ Z_i + G_U.imag @ Z_r))+ µ_r * cp.real(h_U + 1) - µ_i * cp.imag(h_U + 1))

    ## Defining constraints
    # 1) tr(G_E @ x) + µ * (h_E + 1) == 1
    constraints = [ cp.trace(G_E.real @ Z_r - G_E.imag @ Z_i) + 1j * cp.trace(G_E.real @ Z_i + G_E.imag @ Z_r) + µ_r * cp.real(h_E + 1) - µ_i * cp.imag(h_E + 1) == 1 ]

    # 2) tr(E[n] @ X) == µ , ∀n
    for n in range(26):
        constraints.append(cp.trace(E[n] @ Z_r) == µ_r)
        constraints.append(cp.trace(E[n] @ Z_i) == µ_i)
    
    ## Defining optimization problem
    problem = cp.Problem(cp.Maximize(f), constraints)

    ## Solve problem
    res = problem.solve(solver='SCS')
    print(res)

    # Optimized values of X and µ
    X_ = (Z_r + 1j * Z_i).value
    µ_ = (µ_r + 1j * µ_i).value

    # Value of S
    S_ = X_/µ_

    # Normally rank(S) = 1, but resulted S is not of rank 1 
    rankS = np.linalg.matrix_rank(S_)

    return S_
    
# So we apply: The standard Gaussian randomization method (kyn bzzf; 9rinahum f ANUM (1CS))
# to obtain an approximate solution to the matrix decomposition problem
# S = ss^H , finding s helps us extract q.
def sgr(S):
    # Calculate s_approx and q_approx

    m = 5 # Number of random projections
    Y = np.random.rand(N+1,m)
    Z = S @ Y
    Q, R = np.linalg.qr(Z, mode='reduced')
    U, Sigma, VT = svd(Q.T @ S @ Q)

    # Extract the dominant singular vector
    s_approx = np.sqrt(Sigma[0]) * Q @ VT[0]

    # I'm quite confused how to get the value of q ?
    # ChatGPT told me that q = s[:-1] :))
    q_approx = s_approx[:-1]

    return q_approx


if __name__ == '__main__' :
    print("Program starts")

    h_AI_H = np.array([H_AI[0]]).T.conjugate() #0 or any row of H_AI

    k = 0 #index (i will have problem with index since list size is not predefined)
    w_0 = h_AI_H #h_AI^H Transpose Conjugate # Mx1
    q_0 = np.ones(N) # Nx1
    r_0 = secrecy_rate(w_0,q_0) #objective value
    print("w_0 :",w_0)
    print("q_0 :",q_0)
    print("r_0 :",r_0)

    # to keep track of all changes(hoping we don't surpass 100 iterations)
    w = [0]*100; w[k] = w_0
    q = [0]*100; q[k] = q_0
    r = [0]*100; r[k] = r_0

    # Transform this into repeat-until 
    while k < 100 :
        k = k + 1

        print("q[k-1] : ",q[k-1],"\nw[k-1] : ",w[k-1])
        # With q[k-1] : find normalized eigenvector corresponding to largest eigen value (µ_max)
        µ_max = U_max(A(q[k-1]),B(q[k-1])) 

        print(k,". µ_max : ",µ_max)
        
        w[k] = np.sqrt(transmit_power)*µ_max

        print(k,". w : ",w[k])

        # With w[k] : solve (problem (22)) => get(S) which it's rank(S)# 1
        S = problem_22(w[k],q[k-1])
        #             apply gussian randomisation over solution to obtain approx. q[k]
        q[k] = sgr(S)
        r[k] = secrecy_rate(w[k],q[k])
        print(k,". q : ",q[k])
        print(k,". Secrecy : ",r[k])

        if (r[k] - r[k-1])/r[k] <= epsilon :
            print("finished")
            break

# add try-except bloc to catch errors

