clc;clear;close;
dbstop if error;

% Testing with weight parameters learned from Supervised Tempotron Learning
load weightTempotron.mat;
load testSet.mat;

%nTest = 100; % remove this for full test
classindex = zeros(nTest,1);
Testlabel = cell2mat(test_labels);
nCls = 11;
for nPtns = 1:nTest
   [classindex(nPtns)] = GnrlTmptrClsTe(AllWeights,1, TmaxTest(nPtns),ptnTest(nPtns,:));
end
C = confusionmat(Testlabel,classindex);
accuracy = sum(Testlabel(1:nTest) == classindex)/length(classindex);
fprintf('Accuracy = %f\n',accuracy);