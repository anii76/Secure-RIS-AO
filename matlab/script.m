clear;
%clc;

K = 1; %K-factor
zeta = -30; %dBm
M = 4; % Num BS antennas
N = 25; % Num RIS elements : [9 25 40 64 72 88 104 120]~ 
n = 1;
dist_AU = sqrt((145 - 0)^2 + (0 - 0)^2);
dist_AE = sqrt((150 - 0)^2 + (0 - 0)^2);
dist_AI = sqrt((145 - 0)^2 + (5 - 0)^2);
dist_IE = sqrt((150 - 145)^2 + (0 - 5)^2);
dist_IU = sqrt((145 - 145)^2 + (0 - 5)^2);
pl = 3;
pl_AI = 2.2;
r_factor = 0.95;

P_t = 15; % transmit power in dBm [-5 0 5 10 15 20 25]
noise_dB = -80;

noisepow = db2pow(noise_dB);%10^((-80)/10); %dBm to mW
transmit_power = db2pow(P_t);%10^((P_t)/10);%dBm to mW

epsilon = 1e-03;

% 1000 Channel realisations
N_samples = 5;
sr = zeros(N_samples,5);

for l=1:N_samples

% Loop here through channel realizations
h_AU = RicianFadingChannel(M,n,dist_AU,pl,zeta,K);%SpatiallyCorrelatedRician(M,n,dist_AU,pl,zeta,K,r_factor);
h_AE = RicianFadingChannel(M,n,dist_AE,pl,zeta,K);%SpatiallyCorrelatedRician(M,n,dist_AE,pl,zeta,K,r_factor);
H_AI = RicianFadingChannel(M,N,dist_AI,pl_AI,zeta,K);
h_IU = RicianFadingChannel(N,n,dist_IU,pl,zeta,K);
h_IE = RicianFadingChannel(N,n,dist_IE,pl,zeta,K);

%% Algorithm
% Initialization
max_iter = 5;
w = zeros(M,max_iter);
q = zeros(N,max_iter);
r = zeros(1,max_iter);

k = 1; h_AI = H_AI(1,:);
w(:,1) = h_AI'/norm(h_AI);
q(:,1) = ones(N,1);
% secrecy rate
Q = diag(q(:,1));
ue_rate = log2(1 + abs((h_IU*Q*H_AI+h_AU)*w(:,1))^2/noisepow);
eve_rate = log2(1 + abs((h_IE*Q*H_AI+h_AE)*w(:,1))^2/noisepow);
r(1) = ue_rate - eve_rate;

while (k < max_iter)
    k = k + 1;
    % Calculating matrix A & B
    Q = diag(q(:,k-1));
    A = 1/noisepow * ((h_IU*Q*H_AI+h_AU)' * (h_IU*Q*H_AI+h_AU));
    B = 1/noisepow * ((h_IE*Q*H_AI+h_AE)' * (h_IE*Q*H_AI+h_AE));

    % Calculating Umax ----> probably issue here !
    U = (B + 1/transmit_power * eye(M))^-1 * (A + 1/transmit_power * eye(M));
    [eig_vec, eig_val] = eig(U);
    eig_vals = diag(eig_val);
    [max_eigval, max_eigval_idx] = max(eig_vals);
    u_max = eig_vec(:,max_eigval_idx) / norm(eig_vec(:,max_eigval_idx)); % When I use normalize() the optimization works, else Unbounded
                                                                         % When I use normalize() norm(w)^2 > P_t
    % Updating w(k)
    disp(u_max);
    disp(norm(u_max));
    w(:,k) = sqrt(transmit_power) * u_max;
    disp(w(:,k));
    %w(:,k) = sqrt(transmit_power) * w(:,1); %keeping initial value as it yields reasonable results
    
    % Solving problem (22)
    h_U = (conj(h_AU)*conj(w(:,k))*w(:,k).'*h_AU.')/noisepow; h_U = real(h_U);
    h_E = (conj(h_AE)*conj(w(:,k))*w(:,k).'*h_AE.')/noisepow; h_E = real(h_E);

    bloc_1 = diag(conj(h_IU))*conj(H_AI)*conj(w(:,k))*w(:,k).'*H_AI.'*diag(h_IU);
    bloc_2 = diag(conj(h_IU))*conj(H_AI)*conj(w(:,k))*w(:,k).'*h_AU.';
    bloc_3 = conj(h_AU)*conj(w(:,k))*w(:,k).'*H_AI.'*diag(h_IU);

    G_U = 1/noisepow * [bloc_1 bloc_2; bloc_3 0];

    bloc_1 = diag(conj(h_IE))*conj(H_AI)*conj(w(:,k))*w(:,k).'*H_AI.'*diag(h_IE);
    bloc_2 = diag(conj(h_IE))*conj(H_AI)*conj(w(:,k))*w(:,k).'*h_AE.';
    bloc_3 = conj(h_AE)*conj(w(:,k))*w(:,k).'*H_AI.'*diag(h_IE);

    G_E = 1/noisepow * [bloc_1 bloc_2; bloc_3 0];

    %obj , S = Problem_22(h_U,h_E,G_U,G_E,q(:,k),N);
    E = zeros(N+1,N+1,N+1);
    for i=1:N+1
        E(i,i,i) = 1;
    end
    
    % Do I need to initilize these ? 
    % I guess no (just copied 'em from paper for reference)
    %s = [q(:,k-1).' 1].';
    %S = s*s';
    %mu = 1/(trace(G_E*S)+h_E+1);
    %X = u*S;
    
    % Using CVX
    cvx_quiet(false);
    cvx_begin sdp
        variable X(N+1,N+1) hermitian semidefinite 
        variable u nonnegative
        maximize( real(trace( G_U * X )) + u * ( h_U + 1 ) )
        subject to
            real(trace( G_E * X )) + u * ( h_E + 1 ) == 1
            for i=1:N+1
                real(trace(E(:,:,i)*X)) == u
            end 
            X >= 0
    cvx_end
    
    S = X/u;

    % Apply Standard Gaussian randomization
    [D, Sigma, VT] = svd(S);
    d = D(:,1);
    s = Sigma(1,1);
    v = VT(:,1);

    s_approx = d * sqrt(s);
    q_approx = s_approx(2:end);
    q(:,k) = q_approx;

    % Calculate secrecy rate
    Q = diag(q(:,k));
    ue_rate = log2(1 + abs((h_IU*Q*H_AI+h_AU)*w(:,k))^2/noisepow);
    eve_rate = log2(1 + abs((h_IE*Q*H_AI+h_AE)*w(:,k))^2/noisepow);
    r(k) = ue_rate - eve_rate;

    if (r(k) - r(k-1))/r(k) < epsilon
        break;
    end

end

sr(l,:) = r; %max(r); %peut etre je dois prendre le dernier r!
end
% all observations are wrong, I was using w(:,1) in secrecy rate
% after fixing it, secrecy rate is outrageously high ! (ex: 0.1475    7.4035   25.4405)
% it seems that I have multiplied all rates by 10 !
% without RIS y = h*w
