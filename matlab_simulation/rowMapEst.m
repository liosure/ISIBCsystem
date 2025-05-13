function [aProbRowHypo] = rowMapEst(maxAProb)
%MAPESTIMATIOR 此处显示有关此函数的摘要
%   此处显示详细说明
    normMapMat = maxAProb-min(maxAProb);
    normMapMat = min(normMapMat,100);
    aProbMat = exp(normMapMat)./sum(exp(normMapMat));
    aProbMatLn = log(aProbMat(2:end,:));
    aProbMatFalseLn = max(log(1-aProbMat(2:end,:)),-100);
    [L,N] = size(maxAProb);
    aProbRowHypo = sum(aProbMatFalseLn,2)*ones(1,N+1)-[zeros(L-1,1),aProbMatFalseLn]+[zeros(L-1,1),aProbMatLn];
end

