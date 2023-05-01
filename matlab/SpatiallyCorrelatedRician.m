function [H] = SpatiallyCorrelatedRician(M,N,dist,pl,zeta,K,r)
d0 = 1;
PL0 = 10^(zeta/10);
path_loss = PL0*(d0/dist)^pl;

mean = sqrt(K/(K+1));
sigma = sqrt(1/(2*(K+1)));

g_LOS = mean + 1j * mean;
g_NLOS = sigma  * randn(M,N) + 1i *(sigma * randn(M,N));

R = zeros(M,M);
for i=1:M
    for j=1:M
        R(i,j) = r^abs(i-j);
    end
end

B = sqrtm(R); 
small_scale_fading = (g_LOS + B * g_NLOS).';

H = sqrt(path_loss) * small_scale_fading;