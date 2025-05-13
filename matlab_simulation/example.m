clear all;
N = 1000;
K = 512;
a = @(theta,K) exp(1j*2*pi*(0:K-1)*theta);
f = a(2/K, K);
A = f'*randi([0,1],1,N)+1*randn(K,N);
input.step = 100;
input.theoreshold = 1e-10;
input.Mat = A;
tic
output = getMaxEigVec(input);
toc
tic
[u,a,v] = svds(A,1);
toc
disp(norm(output.eigVecRow'*u(:,1)))
disp(norm(v(:,1)'*output.eigVecCol'))