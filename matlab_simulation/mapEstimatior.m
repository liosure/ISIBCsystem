function [outputN,outputAB] = mapEstimatior(omiMat,N,L)
%MAPESTIMATIOR 此处显示有关此函数的摘要
%   此处显示详细说明
    meanNormal = mean(omiMat);
    varianceNormal = 1/(L-1)*sum(abs(omiMat-meanNormal).^2);
    meanAbnormal = zeros(L,N);
    varianceAbnormal = zeros(L,N);
    maxAProbAbnormal = zeros(L,N);
    for l = 1:L
        idx = 1:L;
        idx(l) = [];
        meanAbnormal(l,:) = mean(omiMat(idx,:));
        varianceAbnormal(l,:) = 1/(L-1)*sum(abs(omiMat(idx,:)-meanAbnormal(l,:)).^2);
    end
    maxAProbNormal = log(N-L+1)-log(N)-L*log(pi*varianceNormal)...
        -sum(abs((omiMat-meanNormal)).^2)./varianceNormal;
    for l = 1:L
        idx = 1:L;
        idx(l) = [];
        maxAProbAbnormal(l,:) = log(L-1)-log(N)-log(L)-L*log(pi*varianceAbnormal(l,:))...
            -sum(abs((omiMat(idx,:)-meanAbnormal(l,:))).^2)./varianceAbnormal(l,:);
    end
    outputN = maxAProbNormal(:,2:end);
    outputAB = maxAProbAbnormal(:,2:end);
end

