function [ p ] = getFriedmanTest( leftIdx, rightIdx, testAUCMax )

topRank = 16;
repTimes = length(leftIdx);
leftBestPerf = reshape(testAUCMax(1:topRank, leftIdx), [topRank*repTimes, 1]);
rightBestPerf = reshape(testAUCMax(1:topRank, rightIdx), [topRank*repTimes, 1]);
popcorn = [leftBestPerf rightBestPerf];

leftMax = max(leftBestPerf);
rightMax = max(rightBestPerf);
leftMean = mean(leftBestPerf);
rightMean = mean(rightBestPerf);
if leftMean > rightMean
    disp('Left');
else
    disp('Right');
end

p = friedman(popcorn, repTimes);

end

