function threshVec = calcGradThresh(initThresh,finalThresh,stepSize,timeConst,numIter)
% calculate an exponentially decaying threshold that starts at initThresh
% and finishes at finalThresh with a time constant of timeConst

if (timeConst > 0)
    timeVec = (1:numIter)*stepSize;
    normConst = (initThresh-finalThresh)/exp(-timeVec(1)/timeConst);
    threshVec = finalThresh + normConst*exp(-timeVec/timeConst);
else
    timeVec = (1:numIter)*stepSize;
    normConst = 0;
    threshVec = finalThresh + normConst*exp(-timeVec/timeConst);
end