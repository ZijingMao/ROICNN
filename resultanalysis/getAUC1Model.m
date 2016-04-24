function [ testAUCAll, testAUCMax, testAUCIdx ] = getAUC1Model( setFilesList, filePath )

testAUCAll = zeros(length(setFilesList), 100);
testAUCMax = zeros(length(setFilesList), 1);
testAUCIdx = zeros(length(setFilesList), 1);
for fileIdx = 1:length(setFilesList)
    fileName = setFilesList{fileIdx};
    outputFeat = csvread([filePath fileName]);
    outputFeat = reshape(outputFeat, [3, 100])';

    % get the index of the max performance of validation
    [~, maxIdx] = max(outputFeat(:, 2));
    testAUC = outputFeat(maxIdx, 3);

    testAUCMax(fileIdx) = testAUC;
    testAUCAll(fileIdx, :) = outputFeat(:, 3);
    testAUCIdx(fileIdx) = maxIdx;

end

end