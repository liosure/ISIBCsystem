clear all
N = 10;
M = 200;
K = 64;
L = 4;
codeMat = diag([0, ones(1,N-1)]);
electMat = zeros(L+1,N);
randomNumbers = [1,sort(randperm(N-1, L))+1];
for index = 1:L+1
    electMat(index,randomNumbers(index)) = 1;
end
c = electMat*codeMat;
a = @(theta,K) exp(1j*2*pi*theta*(0:K-1));
theta = 0.9*(rand(L+1,1)-0.5);%[    0.3981;    0.9316;    0.6989;    0.9024];
theta1 = 0.9*(rand(L+1,1)-0.5);
paraReal = containers.Map;
SNR = 100;
pnoise = sqrt(10^(-SNR/10)/2);
for i = 1:L
    paraReal(['BD',int2str(randomNumbers(i+1)-1)])=asin([theta(i+1),theta1(i+1)]*2)/pi*180;
end



f = a(theta, K);
data = 2*randi([0,1],1,M*N)-1;
A = f.'*(kron((1/sqrt(2)*(randn(L+1,1)+1j*randn(L+1,1))*ones(1,N)...
    +1/sqrt(2)*diag(randn(L+1,1)+1j*randn(L+1,1))*c),ones(1,M)).*(ones(L+1,1)*data))...
+pnoise*(randn(K,M*N)+1j*randn(K,M*N));
input.step = 100;
input.theoreshold = 1e-10;
u = zeros(K,N);
for i = 1:N
    input.Mat = A(:,(i-1)*M+1:(i*M));
    output = getMaxEigVec(input);
    u(:,i) = output.eigVecRow;
end
[v1,lambda,v2] = svd(u);
lambda = diag(lambda);
MDL = zeros(1,N-1);
for k=0:N-1
    Lk = prod(lambda(k+1:N).^(1/(K-k))) / (mean(lambda(k+1:N)) + eps);
    MDL(k+1) = -N*(K-k)*log(Lk) + 0.5*k*(2*K-k)*log(N);
end
[~,K_est] = min(MDL);
numSource = K_est-1;
Uforword = [eye(K-1),zeros(K-1,1)]*v1(:,1:numSource);
Ubackword = [zeros(K-1,1),eye(K-1)]*v1(:,1:numSource);
[~, eigval] = eig(Ubackword\Uforword);
thetaEst = angle(diag(eigval'))/pi/2;
eigvecEst =  a(-thetaEst, K);
omiMat = (eigvecEst*eigvecEst')\eigvecEst*u;
omiMatFir = omiMat./omiMat(:,1);
[maProbN, maProbAb] = mapEstimatior(omiMatFir,N,numSource);
maxAProb = [maProbN;maProbAb];
maxAProbRow = rowMapEst(maxAProb);
[~,idx1] = max(maxAProbRow,[],2);
pairResult = containers.Map;
for i = 1:numSource
    pairResult(['BD',int2str(idx1(i)-1)])=asin(thetaEst(i)*2)/pi*180;
end




f = a(theta1, K);
% data = 2*randi([0,1],1,M*N)-1;
A = f.'*(kron((1/sqrt(2)*(randn(L+1,1)+1j*randn(L+1,1))*ones(1,N)...
    +1/sqrt(2)*diag(randn(L+1,1)+1j*randn(L+1,1))*c),ones(1,M)).*(ones(L+1,1)*data))...
    +pnoise*(randn(K,M*N)+1j*randn(K,M*N));
input.step = 100;
input.theoreshold = 1e-10;
u = zeros(K,N);
for i = 1:N
    input.Mat = A(:,(i-1)*M+1:(i*M));
    output = getMaxEigVec(input);
    u(:,i) = output.eigVecRow;
end
[v1,lambda,v2] = svd(u);
lambda = diag(lambda);
MDL = zeros(1,N-1);
for k=0:N-1
    Lk = prod(lambda(k+1:N).^(1/(K-k))) / (mean(lambda(k+1:N)) + eps);
    MDL(k+1) = -N*(K-k)*log(Lk) + 0.5*k*(2*K-k)*log(N);
end
[~,K_est] = min(MDL);
numSource = K_est-1;
Uforword = [eye(K-1),zeros(K-1,1)]*v1(:,1:numSource);
Ubackword = [zeros(K-1,1),eye(K-1)]*v1(:,1:numSource);
[~, eigval] = eig(Ubackword\Uforword);
thetaEst1 = angle(diag(eigval'))/pi/2;
eigvecEst =  a(-thetaEst1, K);
omiMat = (eigvecEst*eigvecEst')\eigvecEst*u;
omiMatFir = omiMat./omiMat(:,1);
[maProbN, maProbAb] = mapEstimatior(omiMatFir,N,numSource);
maxAProb = [maProbN;maProbAb];
maxAProbRow = rowMapEst(maxAProb);
[~,idx2] = max(maxAProbRow,[],2);
% pairResult = containers.Map;
for i = 1:numSource
    if ~isKey(pairResult, ['BD',int2str(idx2(i)-1)])
            continue
    end
    pairResult(['BD',int2str(idx2(i)-1)])=[pairResult(['BD',int2str(idx2(i)-1)]),asin(thetaEst1(i)*2)/pi*180];
end

plotFunction(pairResult,paraReal,L,N-1,SNR)