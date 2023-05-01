function [H] = RicianFadingChannel(M,N,dist,pl,zeta,K)
d0 = 1;
PL0 = 10^(zeta/10);
path_loss = PL0*(d0/dist)^pl;

mu = sqrt(K/(K+1));
sigma = sqrt(1/(2*(K+1)));
Hw = (mu + sigma  * randn(N,M)) + 1i *(sigma * randn(N,M) + mu);
H = sqrt(path_loss)*Hw;
