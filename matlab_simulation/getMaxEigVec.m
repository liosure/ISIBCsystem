function output = getMaxEigVec(input)
    Mat = input.Mat;
    if isfield(input,'theoreshold')
        th = input.theoreshold;
    else
        th = 1e-10;
    end
    if isfield(input,'step')
        stepLimit = input.step;
    else
        stepLimit = 1000;
    end
    E = Mat*Mat';
    [K,~] = size(E);
    u = [ones(K/2,1);zeros(K/2,1)];
    u_temp = zeros(K,1);
    step = 0;
    while norm(u-u_temp)>th && step<stepLimit
        u_temp = u;
        u = E*u; 
        u = u/norm(u);
        u = u/u(1);
        u = u/norm(u);
        step = step+1;
    end
    output.eigVecRow = u;
    output.eigValue = sqrt(norm(E*u));
    output.eigVecCol = output.eigVecRow'*Mat;
    output.eigVecCol = output.eigVecCol/output.eigVecCol(1);
    output.eigVecCol = output.eigVecCol/norm(output.eigVecCol);
end