clear all
t_sum = 0;
step_sum = 0;
N = 400;
K = 200;
a = @(theta,K) exp(1j*2*pi*(0:K-1)*theta);
 f = a(0.1, K);


t_sum = 0;
for i = 1:1000
    A = f'*randi([0,1],1,N)+1*randn(K,N);
    tic;
    E = A*A';
    [u,v] = eig(E);
    t = toc;
    t_sum = t_sum+t;
end
disp("EIG分解时间")
disp(t_sum/1000)

t_sum = 0;
for i = 1:1000
    A = f'*randi([0,1],1,N)+0.01*randn(K,N);
    tic;
    E = A*A';
    [u,v] = eigs(E,1);
    t = toc;
    t_sum = t_sum+t;
end
disp("EIGs分解时间")
disp(t_sum/1000)


t_sum = 0;
for i = 1:1000
    A = f'*randi([0,1],1,N)+0.01*randn(K,N);
    tic;
    [u,v] = svd(A);
    t = toc;
    t_sum = t_sum+t;
end
disp("SVD分解时间")
disp(t_sum/1000)

t_sum = 0;
for i = 1:1000
    A = f'*randi([0,1],1,N)+0.01*randn(K,N);
    tic;
    [u,v] = svds(A);
    t = toc;
    t_sum = t_sum+t;
end
disp("SVDs分解时间")
disp(t_sum/1000)

t_sum = 0;
for i = 1:1000
    A = f'*randi([0,1],1,N)+0.01*randn(K,N);
    tic;
    E = A*A';
    u = ones(K,1);
    u_temp = zeros(K,1);
    th = 1e-10;
    step = 0;
    while norm(u-u_temp)>th
        u_temp = u;
        u = E*u; 
        u = u/norm(u);
        u = u/u(end);
        u = u/norm(u);
        step = step+1;
    end
    t = toc;
    t_sum = t_sum+t;
    step_sum = step_sum+step;
end
disp("迭代")
disp(step_sum/1000)
disp(t_sum/1000)
